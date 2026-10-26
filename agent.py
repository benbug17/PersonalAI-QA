"""A small LangGraph tool-using agent for PersonalAI (needs Python 3.10+ and requirements-agent.txt).

Graph:   START -> model --(tool calls?)--> tools -> model -> ... -> END

Tools:
  calculator      exact arithmetic (safe AST evaluation, no eval())
  search_notes    retrieval over the NoteStore (RAG)
  current_time    UTC clock

run_agent() returns the answer plus the tool-call trace, so evals can check
*which tools were used*, not just what was said.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401  (END re-exported for callers)
from langgraph.prebuilt import ToolNode, tools_condition

import groq_client
from timing import timed, tracker

# The agent keeps the assistant's persona, spoken-style and safety rules, and adds tool rules on top.
AGENT_SYSTEM_PROMPT = groq_client.SYSTEM_PROMPT + """

Tools you can call:
- Use search_notes for any question about the user's own notes or project.
- Always use calculator for arithmetic, even for simple sums, instead of computing in your head.
- If a question needs both a notes lookup and arithmetic, look the number up first, then pass it to calculator.
- Use current_time when the user asks for the time or date.
- Do not use a tool for general knowledge or casual conversation.
- If search_notes finds nothing relevant, say you could not find it in the notes. Never invent details that were not in a tool result."""


class MissingApiKey(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


# ---------------------------------------------------------------- calculator
def _safe_pow(base, exponent):
    if abs(exponent) > 100:
        raise ValueError("exponent too large")
    return operator.pow(base, exponent)


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float | int:
    """Evaluate +, -, *, /, //, %, ** and parentheses on numbers only."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    return walk(ast.parse(expression.strip(), mode="eval"))


def _format_number(value) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, ".10g")


# --------------------------------------------------------------------- tools
def build_tools(store):
    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression such as '(1250 * 0.18) + 40'. Supports + - * / // % ** and parentheses."""
        with timed("tool_calculator"):
            try:
                return _format_number(safe_eval(expression))
            except ZeroDivisionError:
                return "Error: division by zero"
            except (ValueError, SyntaxError, OverflowError, TypeError):
                return "Error: could not evaluate that expression"

    @tool
    def search_notes(query: str) -> str:
        """Search the user's saved notes and project documentation. Use for questions about their own project or notes."""
        with timed("tool_search_notes"):
            hits = store.query(query, k=3)
            if not hits:
                return "No relevant notes found."
            return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

    @tool
    def current_time() -> str:
        """Get the current date and time in UTC."""
        with timed("tool_current_time"):
            return datetime.now(timezone.utc).strftime("%A %d %B %Y, %H:%M UTC")

    return [calculator, search_notes, current_time]


# ------------------------------------------------------- repairing malformed tool calls
def _failed_generation(exc: Exception) -> str | None:
    """The text the model actually generated, as reported in the provider's 400 error (if present)."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    inner = body.get("error") if isinstance(body.get("error"), dict) else body   # SDK versions differ in the nesting
    value = inner.get("failed_generation")
    return value if isinstance(value, str) else None


def recover_tool_call(exc: Exception, tool_names) -> AIMessage | None:
    """Rebuild the call the model MEANT to make when the provider rejected it only because the tool name was
    malformed (for example 'calculator<|channel|>commentary'). Returns None when it cannot be repaired safely."""
    raw = _failed_generation(exc)
    if not raw:
        return None
    try:
        call = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(call, dict):
        return None
    name = str(call.get("name", "")).split("<|", 1)[0].strip()
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if name not in tool_names or not isinstance(args, dict):
        return None
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"recovered-{uuid.uuid4().hex[:12]}",
                                              "type": "tool_call"}])


# --------------------------------------------------------------------- graph
def build_agent(llm, store, system_prompt: str = AGENT_SYSTEM_PROMPT):
    """Compile the agent graph. `llm` is any LangChain chat model that supports bind_tools."""
    tools = build_tools(store)
    model = llm.bind_tools(tools)

    tool_names = {t.name for t in tools}

    def call_model(state: MessagesState):
        with timed("agent_llm"):
            try:
                reply = model.invoke([SystemMessage(content=system_prompt)] + state["messages"])
            except Exception as exc:  # noqa: BLE001 - only a repairable malformed tool call is handled here
                if not is_tool_call_glitch(exc):
                    raise
                reply = recover_tool_call(exc, tool_names)
                if reply is None:
                    raise
                tracker.record("tool_call_recovered", 0.0)   # shows up in the benchmark table as a count
        usage = getattr(reply, "usage_metadata", None) or {}
        tracker.record_tokens(usage.get("input_tokens"), usage.get("output_tokens"))
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    latency_ms: int = 0
    truncated: bool = False
    error: str | None = None  # set when the provider kept rejecting a malformed tool call

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def _text_of(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    # some providers return a list of content blocks
    return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()


def is_tool_call_glitch(exc: Exception) -> bool:
    """True for the provider's 400 'tool_use_failed': the model wrote a malformed tool call (for example a
    stray formatting token inside the tool name) and the API refused it. Retrying usually works."""
    return getattr(exc, "status_code", None) == 400 and "tool_use_failed" in str(exc)


def run_agent(graph, question: str, max_llm_calls: int = 5, retries: int = 2) -> AgentResult:
    """Run one question through the agent and return the answer with its tool trace.

    A malformed tool call rejected by the provider is retried up to `retries` times. If it keeps failing the
    result carries `error` instead of raising, so one bad generation cannot kill a whole evaluation run.
    Any other provider error (rate limit, network) still raises.
    """
    start = time.perf_counter()
    # LangGraph counts its input step too, so n model calls (each followed by a tools step) fit in a
    # limit of exactly 2n. Measured, not assumed: both boundaries are tested in tests/unit/test_agent.py.
    config = {"recursion_limit": max_llm_calls * 2}
    messages: list = []
    error = None
    with timed("agent_total"):
        for attempt in range(1, retries + 2):
            try:
                state = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
                messages, error = state["messages"], None
                break
            except GraphRecursionError:
                error = None  # the model kept calling tools without ever answering
                break
            except Exception as exc:  # noqa: BLE001 - only the tool-call glitch is handled; the rest is re-raised
                if not is_tool_call_glitch(exc):
                    raise
                error = f"malformed tool call rejected by the provider after {attempt} attempt(s): {str(exc)[:160]}"

    tool_calls: list[dict] = []
    llm_calls = 0
    answer = ""
    for message in messages:
        if isinstance(message, AIMessage):
            llm_calls += 1
            tool_calls.extend({"name": c["name"], "args": c["args"]} for c in message.tool_calls)
            answer = _text_of(message)  # the last AI message is the final answer
    truncated = not answer and not error
    if truncated or error:
        answer = "Sorry, I could not finish working that out. Could you rephrase the question?"
    return AgentResult(
        answer=answer,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        latency_ms=round((time.perf_counter() - start) * 1000),
        truncated=truncated,
        error=error,
    )


# ------------------------------------------------------------ production wiring
def make_groq_llm(model: str | None = None, temperature: float = 0.0):
    """LangChain chat model on Groq. Uses the same GROQ_MODEL / reasoning settings as groq_client."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise MissingApiKey("GROQ_API_KEY is not set")
    from langchain_groq import ChatGroq

    model = model or groq_client.get_model()
    extra = {"reasoning_effort": "low"} if groq_client._is_reasoning_model(model) else {}
    # max_tokens=1024: reasoning models spend hidden thinking tokens against this limit
    return ChatGroq(model=model, temperature=temperature, max_tokens=1024, api_key=key, **extra)


_default_graph = None


def get_default_graph():
    """Build (once) the agent used by the API: Groq LLM + Chroma notes store."""
    global _default_graph
    if _default_graph is None:
        from rag import DEFAULT_NOTES_DIR, NoteStore, get_embedder, ingest_directory

        llm = make_groq_llm()
        store = NoteStore(get_embedder(os.getenv("NOTES_EMBEDDER", "st")), path=os.getenv("NOTES_DB", ".chroma"))
        ingest_directory(store, os.getenv("NOTES_DIR", str(DEFAULT_NOTES_DIR)))
        _default_graph = build_agent(llm, store)
    return _default_graph

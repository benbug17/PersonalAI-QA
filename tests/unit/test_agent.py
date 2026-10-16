import uuid

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

import agent  # noqa: E402
import groq_client  # noqa: E402
from agent import build_agent, build_tools, make_groq_llm, run_agent, safe_eval  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from rag import DEFAULT_NOTES_DIR, HashingEmbedder, NoteStore, ingest_directory  # noqa: E402
from timing import tracker  # noqa: E402

from tests.agent_fakes import ScriptedChatModel, final, tool_call  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_tracker():
    tracker.reset()
    yield
    tracker.reset()


@pytest.fixture
def store():
    s = NoteStore(HashingEmbedder(), collection=f"agent-{uuid.uuid4().hex}")
    ingest_directory(s, DEFAULT_NOTES_DIR)
    return s


def agent_for(store, *responses):
    return build_agent(ScriptedChatModel(responses=list(responses)), store)


# ---- calculator ------------------------------------------------------------
@pytest.mark.parametrize("expr,expected", [
    ("17 * 23", 391), ("(1250 * 0.18) + 40", 265.0), ("2 ** 10", 1024),
    ("-5 + 8", 3), ("+4", 4), ("7 // 2", 3), ("7 % 4", 3), ("10 / 4", 2.5),
])
def test_safe_eval_values(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "open('x')", "a + 1", "2 ** 1000",
    "[1,2]", "True + 1", "'a' * 3", "1+" * 101 + "1",
])
def test_safe_eval_rejects_anything_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        safe_eval(expr)


def test_calculator_tool_formats_results_and_errors(store):
    calculator = build_tools(store)[0]
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"
    assert calculator.invoke({"expression": "6 * 7"}) == "42"
    assert calculator.invoke({"expression": "1 / 0"}) == "Error: division by zero"
    assert calculator.invoke({"expression": "__import__('os')"}) == "Error: could not evaluate that expression"
    assert calculator.invoke({"expression": "2 ** 5000"}) == "Error: could not evaluate that expression"


def test_current_time_tool_reports_utc(store):
    assert build_tools(store)[2].invoke({}).endswith("UTC")


def test_agent_prompt_keeps_the_assistants_persona_and_adds_tool_rules():
    assert groq_client.SYSTEM_PROMPT in agent.AGENT_SYSTEM_PROMPT
    assert "search_notes" in agent.AGENT_SYSTEM_PROMPT and "calculator" in agent.AGENT_SYSTEM_PROMPT


# ---- agent behaviour -------------------------------------------------------
def test_agent_answers_without_tools(store):
    result = run_agent(agent_for(store, final("Tokyo.")), "Capital of Japan?")
    assert result.answer == "Tokyo." and result.tools_used == [] and result.llm_calls == 1 and not result.truncated


def test_agent_uses_calculator_and_reports_trace(store):
    graph = agent_for(store, tool_call("calculator", {"expression": "17 * 23"}), final("It is 391."))
    result = run_agent(graph, "What is 17 times 23?")
    assert result.tools_used == ["calculator"] and result.tool_calls[0]["args"] == {"expression": "17 * 23"}
    assert result.answer == "It is 391." and result.llm_calls == 2


def test_search_notes_returns_the_retrieved_text_to_the_model(store):
    graph = agent_for(store, tool_call("search_notes", {"query": "retired Groq model"}), final("llama-3.1-8b-instant."))
    state = graph.invoke({"messages": [HumanMessage(content="Which model was retired?")]})
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert "llama-3.1-8b-instant" in tool_messages[0].content and "[personalai_facts.md]" in tool_messages[0].content


def test_search_with_no_hits_tells_the_model_nothing_was_found(store):
    graph = agent_for(store, tool_call("search_notes", {"query": "airspeed velocity unladen swallow"}), final("Not in my notes."))
    state = graph.invoke({"messages": [HumanMessage(content="swallow?")]})
    assert [m for m in state["messages"] if isinstance(m, ToolMessage)][0].content == "No relevant notes found."


def test_agent_chains_two_tools(store):
    graph = agent_for(store, tool_call("search_notes", {"query": "upload size limit"}, "c1"),
                      tool_call("calculator", {"expression": "10 * 3"}, "c2"), final("Thirty."))
    result = run_agent(graph, "Triple the upload limit.")
    assert result.tools_used == ["search_notes", "calculator"] and result.llm_calls == 3


def test_tool_failure_is_survivable(store):
    graph = agent_for(store, tool_call("calculator", {"expression": "1 / 0"}), final("I could not divide by zero."))
    result = run_agent(graph, "What is 1 divided by 0?")
    assert result.answer.startswith("I could not") and not result.truncated


def test_unknown_tool_does_not_crash(store):
    graph = agent_for(store, tool_call("launch_missiles", {}), final("Sorry, I can't do that."))
    assert run_agent(graph, "do something odd").answer == "Sorry, I can't do that."


def test_runaway_tool_loop_is_cut_off(store):
    result = run_agent(agent_for(store, tool_call("current_time", {})), "loop forever", max_llm_calls=3)
    assert result.truncated is True and "could not finish" in result.answer


def test_step_cap_stops_the_model_at_exactly_max_calls(store):
    model = ScriptedChatModel(responses=[tool_call("current_time", {})])
    result = run_agent(build_agent(model, store), "loop forever", max_llm_calls=3)
    assert result.truncated and model.index == 3


def test_exactly_max_calls_with_a_final_answer_is_allowed(store):
    graph = agent_for(store, tool_call("current_time", {}, "a"), tool_call("calculator", {"expression": "1+1"}, "b"), final("Done."))
    result = run_agent(graph, "two tools then answer", max_llm_calls=3)
    assert result.answer == "Done." and not result.truncated and result.llm_calls == 3


def test_content_blocks_from_the_provider_are_joined(store):
    blocks = AIMessage(content=[{"type": "text", "text": "Hello"}, "there"])
    assert run_agent(agent_for(store, blocks), "hi").answer == "Hello there"


def test_token_usage_and_step_timings_are_recorded(store):
    run_agent(agent_for(store, final("Hi.", input_tokens=40, output_tokens=7)), "hello")
    summary = tracker.summary()
    assert summary["tokens"]["prompt_tokens"] == 40 and summary["tokens"]["completion_tokens"] == 7
    assert {"agent_total", "agent_llm"} <= set(summary["steps"])


# ---- production wiring ------------------------------------------------------
def test_make_groq_llm_requires_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(agent.MissingApiKey):
        make_groq_llm()


def test_make_groq_llm_uses_the_apps_model_and_low_reasoning_for_gpt_oss(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    llm = make_groq_llm()
    assert llm.model_name == "openai/gpt-oss-20b" and llm.reasoning_effort == "low" and llm.max_tokens == 1024


def test_make_groq_llm_skips_reasoning_effort_for_other_models(monkeypatch):
    llm = make_groq_llm(model="some-other-model")
    assert llm.model_name == "some-other-model" and llm.reasoning_effort is None


def test_default_graph_is_built_once_and_indexes_the_notes(monkeypatch, tmp_path):
    scripted = ScriptedChatModel(responses=[final("ok")])
    monkeypatch.setattr(agent, "_default_graph", None)
    monkeypatch.setattr(agent, "make_groq_llm", lambda: scripted)
    monkeypatch.setattr("rag.sentence_transformer_embedder", lambda: HashingEmbedder())
    monkeypatch.setenv("NOTES_DB", str(tmp_path / "chroma"))
    first = agent.get_default_graph()
    assert agent.get_default_graph() is first
    assert run_agent(first, "hello").answer == "ok"
    assert (tmp_path / "chroma").exists()

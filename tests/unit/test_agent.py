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


def test_default_graph_uses_the_embedder_named_in_the_environment(monkeypatch, tmp_path):
    import rag
    seen = []
    monkeypatch.setattr(agent, "_default_graph", None)
    monkeypatch.setattr(agent, "make_groq_llm", lambda: ScriptedChatModel(responses=[final("hi")]))
    monkeypatch.setattr(rag, "get_embedder", lambda name: seen.append(name) or HashingEmbedder())
    monkeypatch.setenv("NOTES_DB", str(tmp_path / "db"))
    monkeypatch.setenv("NOTES_EMBEDDER", "onnx")
    agent.get_default_graph()
    assert seen == ["onnx"]
    monkeypatch.setattr(agent, "_default_graph", None)


# ---- malformed tool calls rejected by the provider --------------------------------------------------------
class ProviderError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


GLITCH = ProviderError("Error code: 400 - tool_use_failed: calculator<|channel|>commentary was not in request.tools", 400)


class FakeGraph:
    """Stands in for a compiled graph: each invoke() returns or raises the next scripted outcome."""
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def invoke(self, payload, config=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def answered(text):
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=text)]}


@pytest.mark.parametrize("exc,expected", [
    (ProviderError("tool_use_failed", 400), True),
    (ProviderError("some other bad request", 400), False),
    (ProviderError("tool_use_failed", 429), False),
    (RuntimeError("tool_use_failed"), False),
])
def test_only_the_tool_call_glitch_is_recognised(exc, expected):
    assert agent.is_tool_call_glitch(exc) is expected


def test_a_rejected_tool_call_is_retried_and_the_retry_can_succeed():
    graph = FakeGraph(GLITCH, answered("It is 30."))
    result = run_agent(graph, "q")
    assert result.answer == "It is 30." and result.error is None and graph.calls == 2


def test_a_tool_call_that_keeps_being_rejected_becomes_an_error_result_not_an_exception():
    graph = FakeGraph(GLITCH, GLITCH, GLITCH)
    result = run_agent(graph, "q")                     # default: 2 retries = 3 attempts
    assert graph.calls == 3 and not result.truncated
    assert result.error.startswith("malformed tool call rejected by the provider after 3 attempt(s)")
    assert "could not finish" in result.answer and result.tool_calls == []


def test_the_retry_count_is_configurable_and_zero_means_one_attempt():
    graph = FakeGraph(GLITCH)
    assert "after 1 attempt(s)" in run_agent(graph, "q", retries=0).error and graph.calls == 1


def test_the_error_message_is_kept_short():
    long_glitch = ProviderError("tool_use_failed " + "x" * 5000, 400)
    assert len(run_agent(FakeGraph(long_glitch), "q", retries=0).error) < 300


@pytest.mark.parametrize("exc", [
    ProviderError("tokens per day", 429),
    ProviderError("some other bad request", 400),
    RuntimeError("connection reset"),
])
def test_every_other_provider_error_still_raises_and_is_not_retried(exc):
    graph = FakeGraph(exc, answered("never reached"))
    with pytest.raises(type(exc)):
        run_agent(graph, "q")
    assert graph.calls == 1


# ---- repairing a tool call the provider rejected only because its NAME was malformed -----------------------
class FlakyChatModel(ScriptedChatModel):
    """Raises the exceptions in `failures` (one per call) before it starts replaying its scripted replies."""
    failures: list = []
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return super()._generate(messages, stop, run_manager, **kwargs)


def rejected(generation="default", nested=True, include_generation=True):
    """The REAL groq SDK exception for a 400 tool_use_failed, built the way the SDK builds it."""
    import json

    import groq
    import httpx
    if generation == "default":
        generation = json.dumps({"name": "calculator<|channel|>commentary", "arguments": {"expression": "10 * 3"}})
    inner = {"message": "Tool call validation failed: attempted to call tool 'calculator<|channel|>commentary' which was "
                        "not in request.tools", "type": "invalid_request_error", "code": "tool_use_failed"}
    if include_generation:
        inner["failed_generation"] = generation
    body = {"error": inner} if nested else inner
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"), json=body)
    return groq.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


TOOLS = {"calculator", "search_notes", "current_time"}


def test_the_intended_call_is_rebuilt_from_the_real_sdk_error():
    message = agent.recover_tool_call(rejected(), TOOLS)
    call = message.tool_calls[0]
    assert message.content == "" and call["name"] == "calculator" and call["args"] == {"expression": "10 * 3"}
    assert call["id"].startswith("recovered-") and call["type"] == "tool_call"


def test_both_sdk_body_shapes_are_understood():
    assert agent.recover_tool_call(rejected(nested=True), TOOLS).tool_calls[0]["name"] == "calculator"
    assert agent.recover_tool_call(rejected(nested=False), TOOLS).tool_calls[0]["name"] == "calculator"


def test_arguments_may_arrive_as_a_json_string():
    import json
    generation = json.dumps({"name": "calculator<|channel|>commentary", "arguments": json.dumps({"expression": "2+2"})})
    assert agent.recover_tool_call(rejected(generation), TOOLS).tool_calls[0]["args"] == {"expression": "2+2"}


@pytest.mark.parametrize("dirty,clean", [
    ("calculator<|channel|>commentary", "calculator"),
    ("search_notes<|channel|>commentary json", "search_notes"),
    ("current_time<|channel|>analysis", "current_time"),
    ("calculator", "calculator"),
])
def test_the_stray_formatting_token_is_stripped_from_the_name(dirty, clean):
    import json
    generation = json.dumps({"name": dirty, "arguments": {}})
    assert agent.recover_tool_call(rejected(generation), TOOLS).tool_calls[0]["name"] == clean


@pytest.mark.parametrize("generation", [
    "", "not json at all", "[1, 2]", '{"name": "launch_missiles<|channel|>x", "arguments": {}}',
    '{"name": "calculator", "arguments": "{oops"}', '{"name": "calculator", "arguments": [1]}',
])
def test_a_call_that_cannot_be_repaired_safely_returns_none(generation):
    assert agent.recover_tool_call(rejected(generation), TOOLS) is None


def test_errors_without_a_usable_body_return_none():
    assert agent.recover_tool_call(ProviderError("tool_use_failed", 400), TOOLS) is None          # no body at all
    assert agent.recover_tool_call(rejected(include_generation=False), TOOLS) is None            # no failed_generation
    odd = ProviderError("tool_use_failed", 400)
    odd.body = "plain text"
    assert agent.recover_tool_call(odd, TOOLS) is None
    odd.body = {"error": {"failed_generation": {"not": "a string"}}}
    assert agent.recover_tool_call(odd, TOOLS) is None


# ---- end to end: the agent carries on instead of restarting -------------------------------------------------
def test_a_malformed_tool_call_is_repaired_in_place_and_the_agent_carries_on(store):
    model = FlakyChatModel(responses=[final("It is 30.")], failures=[rejected()])
    result = run_agent(build_agent(model, store), "What is 3 times 10?")
    assert result.answer == "It is 30." and result.error is None
    assert result.tools_used == ["calculator"]            # the repaired call really ran (a restart would show no tool)
    assert model.calls == 2                               # one rejected call, then the answer: no restart, no extra search
    assert tracker.summary()["steps"]["tool_call_recovered"]["count"] == 1


def test_a_glitch_that_cannot_be_repaired_still_falls_back_to_retrying_the_whole_question(store):
    model = FlakyChatModel(responses=[final("Fine.")], failures=[GLITCH])
    result = run_agent(build_agent(model, store), "q")
    assert result.answer == "Fine." and model.calls == 2 and result.tools_used == []
    assert "tool_call_recovered" not in tracker.summary()["steps"]


def test_a_repair_that_names_an_unknown_tool_is_not_executed(store):
    bad = rejected('{"name": "launch_missiles<|channel|>commentary", "arguments": {}}')
    model = FlakyChatModel(responses=[final("Fine.")], failures=[bad])
    result = run_agent(build_agent(model, store), "q")
    assert result.tools_used == [] and "tool_call_recovered" not in tracker.summary()["steps"]


def test_other_errors_inside_the_model_call_still_propagate_untouched(store):
    model = FlakyChatModel(responses=[final("never")], failures=[ProviderError("tokens per day", 429)])
    with pytest.raises(ProviderError):
        run_agent(build_agent(model, store), "q")
    assert model.calls == 1


def test_only_a_400_tool_use_failed_is_ever_repaired_even_if_the_body_looks_repairable(store):
    server_error = rejected()
    server_error.status_code = 500                         # looks repairable, but it is not the glitch we handle
    model = FlakyChatModel(responses=[final("never")], failures=[server_error])
    with pytest.raises(type(server_error)):
        run_agent(build_agent(model, store), "q")
    assert model.calls == 1 and "tool_call_recovered" not in tracker.summary()["steps"]


def test_two_repairs_in_one_question_both_run_because_each_gets_its_own_id(store):
    model = FlakyChatModel(responses=[final("done")], failures=[rejected(), rejected()])
    result = run_agent(build_agent(model, store), "q")
    assert result.tools_used == ["calculator", "calculator"] and result.answer == "done"
    assert tracker.summary()["steps"]["tool_call_recovered"]["count"] == 2


def test_every_repaired_call_gets_a_distinct_id():
    ids = {agent.recover_tool_call(rejected(), TOOLS).tool_calls[0]["id"] for _ in range(20)}
    assert len(ids) == 20      # providers expect tool-call ids to be unique within a conversation

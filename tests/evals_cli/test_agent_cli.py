"""The `python -m evals.agent_run` and `python -m evals.bench` entry points, with the models stubbed."""
import json
import sys

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

import agent  # noqa: E402
import groq_client  # noqa: E402
from evals import agent_run, bench  # noqa: E402
from timing import tracker  # noqa: E402

from tests.agent_fakes import ScriptedChatModel, final, tool_call  # noqa: E402

pytestmark = pytest.mark.genai

ONE_CASE = [{"id": "calc", "query": "17 times 23?", "expected_tools": ["calculator"],
             "checks": [{"type": "non_empty"}, {"type": "matches_regex", "pattern": r"\b391\b"}]}]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(agent_run, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(bench, "load_dotenv", lambda *a, **k: None)
    tracker.reset()
    yield
    tracker.reset()


def use_script(monkeypatch, *responses, cases=ONE_CASE):
    model = ScriptedChatModel(responses=list(responses))
    monkeypatch.setattr(agent_run, "make_groq_llm", lambda: model)
    monkeypatch.setattr(agent_run, "load_agent_cases", lambda path=None: cases)
    return model


def run(tmp_path, *argv):
    return agent_run.main([*argv, "--embedder", "hashing", "--out", str(tmp_path / "out.json")])


# ---- agent_run -----------------------------------------------------------------------------
def test_agent_cases_file_loads():
    assert len(agent_run.load_agent_cases()) == 11


def test_missing_key_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert run(tmp_path) == 2 and "GROQ_API_KEY" in capsys.readouterr().err


def test_live_embedder_not_installed_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert agent_run.main(["--out", str(tmp_path / "o.json")]) == 2 and "requirements-agent-live.txt" in capsys.readouterr().err


def test_bad_repeat_exit_2(tmp_path):
    assert run(tmp_path, "--repeat", "0") == 2


def test_failed_preflight_exit_2(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, tool_call("current_time", {}))   # never answers: hits the step cap
    assert run(tmp_path) == 2 and "Preflight" in capsys.readouterr().err


def test_all_pass_exit_0_and_results_file(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"), tool_call("calculator", {"expression": "17 * 23"}), final("It is 391."))
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "PASS  calc" in out and "tool_selection=100%" in out and "overall=100%" in out
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["summary"]["cases"] == 1 and saved["rows"][0]["attempts"][0]["tools"] == ["calculator"]


def test_a_failing_case_exit_1_with_reasons(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"), final("It is 391."))   # right answer, but did not use the calculator
    assert run(tmp_path) == 1
    out = capsys.readouterr().out
    assert "FAIL  calc" in out and "tools: expected ['calculator'], used []" in out and "tool_selection=0%" in out


def test_flaky_case_label(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"), tool_call("calculator", {"expression": "17 * 23"}, "a"), final("It is 391."),
               final("I am not sure."))
    assert run(tmp_path, "--repeat", "2") == 1 and "FLAKY" in capsys.readouterr().out


# ---- bench -----------------------------------------------------------------------------------
def test_bench_records_each_call():
    bench.run_benchmark(lambda q: "x", 7)
    assert tracker.summary()["steps"]["llm_text"]["count"] == 7


def test_bench_without_a_key_exit_2(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert bench.main([]) == 2 and "GROQ_API_KEY" in capsys.readouterr().err


def test_bench_prints_a_markdown_table(monkeypatch, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "answer")
    assert bench.main(["--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "| llm_text | 3 |" in out and "estimated cost" in out


def test_bench_can_include_the_agent(monkeypatch, capsys):
    import uuid

    from rag import HashingEmbedder, NoteStore
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "answer")
    graph = agent.build_agent(ScriptedChatModel(responses=[final("ok")]),
                              NoteStore(HashingEmbedder(), collection=f"bench-{uuid.uuid4().hex}"))
    monkeypatch.setattr(agent, "get_default_graph", lambda: graph)
    assert bench.main(["--n", "2", "--agent"]) == 0
    assert "| agent_total | 2 |" in capsys.readouterr().out


def test_bench_asks_at_temperature_zero(monkeypatch):
    temps = []
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: temps.append(temperature) or "a")
    assert bench.main(["--n", "2", "--warmup", "0"]) == 0
    assert temps == [0.0, 0.0]


# ---- failure paths -----------------------------------------------------------------------------
def test_preflight_that_raises_is_a_setup_problem_exit_2(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"))

    def boom(graph, question, **kwargs):
        raise RuntimeError("provider down")
    monkeypatch.setattr(agent_run, "run_agent", boom)
    assert run(tmp_path) == 2
    assert "Preflight agent call failed: provider down" in capsys.readouterr().err


def test_bench_warns_when_calls_fail_so_timings_are_not_misread(monkeypatch, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: None)
    assert bench.main(["--n", "3"]) == 0
    assert "only 0/3 calls returned an answer" in capsys.readouterr().err


def test_bench_agent_setup_problem_exit_2(monkeypatch, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "answer")

    def cannot_start():
        raise RuntimeError("sentence-transformers is not installed")
    monkeypatch.setattr(agent, "get_default_graph", cannot_start)
    assert bench.main(["--n", "1", "--agent"]) == 2
    assert "Agent setup problem" in capsys.readouterr().err


# ---- provider failures mid-run (e.g. Groq's daily token quota) ---------------------------------
class FakeProviderError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


TWO_CASES = ONE_CASE + [dict(ONE_CASE[0], id="calc2")]


def good_result():
    return agent.AgentResult(answer="It is 391.", tool_calls=[{"name": "calculator", "args": {}}], llm_calls=2, latency_ms=5)


def script_run_agent(monkeypatch, *outcomes):
    """Replace run_agent: each call returns the next outcome, or raises it if it is an exception."""
    it = iter(outcomes)

    def fake(graph, question, **kwargs):
        outcome = next(it)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(agent_run, "run_agent", fake)


def test_rate_limit_mid_run_is_a_clean_exit_2_not_a_traceback(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("unused"), cases=TWO_CASES)
    # preflight ok, first case ok, second case hits the daily token quota
    script_run_agent(monkeypatch, good_result(), good_result(), FakeProviderError("tokens per day", 429))
    assert run(tmp_path) == 2
    err = capsys.readouterr().err
    assert "rate limit" in err and "--repeat 1" in err
    assert "1 case(s) finished before the stop. Partial results saved" in err
    assert "Traceback" not in err
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["partial"] is True and "tokens per day" in saved["stopped"]
    assert [r["id"] for r in saved["rows"]] == ["calc"]   # the finished case is kept, not thrown away


def test_any_other_provider_error_is_reported_without_the_rate_limit_advice(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("unused"))
    script_run_agent(monkeypatch, good_result(), RuntimeError("connection reset"))
    assert run(tmp_path) == 2
    err = capsys.readouterr().err
    assert "Stopped: connection reset" in err and "rate limit" not in err
    assert "0 case(s) finished" in err


def test_on_row_reports_each_finished_case(monkeypatch):
    seen = []
    script_run_agent(monkeypatch, good_result(), good_result())
    rows = agent_run.evaluate(object(), TWO_CASES, on_row=seen.append)
    assert [r["id"] for r in seen] == ["calc", "calc2"] == [r["id"] for r in rows]


def test_a_finished_run_reports_how_many_tokens_it_used(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"), tool_call("calculator", {"expression": "17 * 23"}), final("It is 391."))
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "LLM calls: 3, tokens used: 20 prompt + 10 completion" in out


# ---- embedder choice and setup failures -------------------------------------------------------------
def test_the_chosen_embedder_name_is_passed_through(monkeypatch, tmp_path):
    from rag import HashingEmbedder
    use_script(monkeypatch, final("Hello!"), tool_call("calculator", {"expression": "17 * 23"}), final("It is 391."))
    seen = []
    monkeypatch.setattr(agent_run, "get_embedder", lambda name: seen.append(name) or HashingEmbedder())
    assert agent_run.main(["--embedder", "onnx", "--out", str(tmp_path / "out.json")]) == 0
    assert seen == ["onnx"]


def test_an_embedding_model_that_fails_to_download_is_a_setup_problem(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"))

    def cannot_download(name):
        raise OSError("download failed")
    monkeypatch.setattr(agent_run, "get_embedder", cannot_download)
    assert agent_run.main(["--embedder", "onnx", "--out", str(tmp_path / "out.json")]) == 2
    assert "Setup problem: download failed" in capsys.readouterr().err


def test_a_failure_while_indexing_the_notes_is_a_setup_problem(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("Hello!"))

    def cannot_index(store, directory):
        raise RuntimeError("model download interrupted")
    monkeypatch.setattr(agent_run, "ingest_directory", cannot_index)
    assert run(tmp_path) == 2
    assert "Setup problem: model download interrupted" in capsys.readouterr().err


# ---- a malformed tool call must not kill the run --------------------------------------------------------
def test_a_case_whose_tool_call_keeps_being_rejected_fails_but_the_run_continues(monkeypatch, tmp_path, capsys):
    use_script(monkeypatch, final("unused"), cases=TWO_CASES)
    broken = agent.AgentResult(answer="Sorry", error="malformed tool call rejected by the provider after 3 attempt(s)")
    script_run_agent(monkeypatch, good_result(), broken, good_result())
    assert run(tmp_path) == 1                      # finished normally; one case failed
    out = capsys.readouterr().out
    assert "FAIL  calc" in out and "PASS  calc2" in out
    assert "agent error: malformed tool call rejected" in out and "gave up" not in out


def test_score_case_reports_a_provider_rejected_call_as_an_error_not_a_step_limit():
    result = agent.AgentResult(answer="Sorry", error="malformed tool call")
    s = agent_run.score_case(ONE_CASE[0], result)
    assert not s["passed"] and not s["answer_ok"]
    assert s["failures"][0] == "tools: expected ['calculator'], used []"
    assert "agent error: malformed tool call" in s["failures"]
    assert not any("gave up" in f for f in s["failures"])


def test_an_errored_result_is_never_counted_correct_even_if_its_text_looks_right():
    result = agent.AgentResult(answer="It is 391.", tool_calls=[{"name": "calculator", "args": {}}],
                               error="malformed tool call")
    s = agent_run.score_case(ONE_CASE[0], result)
    assert s["tool_ok"] is True and not s["answer_ok"] and not s["passed"]
    assert s["failures"] == ["agent error: malformed tool call"]


def test_an_error_is_reported_as_an_error_even_if_the_result_is_also_marked_truncated():
    result = agent.AgentResult(answer="It is 391.", tool_calls=[{"name": "calculator", "args": {}}],
                               truncated=True, error="malformed tool call")
    assert agent_run.score_case(ONE_CASE[0], result)["failures"] == ["agent error: malformed tool call"]


# ---- warm-up calls are made but never counted ------------------------------------------------------------
def test_warmup_calls_are_made_but_not_counted(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: calls.append(q) or "a")
    assert bench.main(["--n", "3", "--warmup", "2"]) == 0
    out = capsys.readouterr().out
    assert len(calls) == 5 and "| llm_text | 3 |" in out        # 2 warm-up + 3 measured, only 3 reported


def test_the_benchmark_output_says_what_was_measured(monkeypatch, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "a")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    assert bench.main(["--n", "2", "--warmup", "1"]) == 0
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith("Benchmark: model=some-model n=2 warmup=1 date=20")


def test_agent_warmups_are_not_counted_either(monkeypatch, capsys):
    import uuid

    from rag import HashingEmbedder, NoteStore
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "answer")
    model = ScriptedChatModel(responses=[final("ok")])
    graph = agent.build_agent(model, NoteStore(HashingEmbedder(), collection=f"bench-{uuid.uuid4().hex}"))
    monkeypatch.setattr(agent, "get_default_graph", lambda: graph)
    assert bench.main(["--n", "2", "--warmup", "3", "--agent"]) == 0
    out = capsys.readouterr().out
    assert model.index == 3 + 2                                  # the agent really ran 3 warm-ups and 2 measured calls
    assert "| agent_total | 2 |" in out and "| llm_text | 2 |" in out   # ...but only the 2 measured ones are reported


def test_the_agent_is_benchmarked_on_questions_that_really_use_its_tools(monkeypatch):
    asked = []

    def fake_run_agent(graph, question, **kwargs):
        asked.append(question)
        return agent.AgentResult(answer="ok")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "a")
    monkeypatch.setattr(agent, "get_default_graph", lambda: object())
    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    assert bench.main(["--n", "6", "--warmup", "0", "--agent"]) == 0
    assert asked == bench.AGENT_QUESTIONS
    text = " ".join(asked).lower()
    assert "my notes" in text and "times" in text and "compute" in text    # notes lookups AND arithmetic are both exercised
    assert any("notes" in q and "multiply" in q for q in asked)             # including a two-tool question


def test_agent_warmups_use_the_same_question_set_as_the_measured_calls(monkeypatch):
    asked = []
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: "a")
    monkeypatch.setattr(agent, "get_default_graph", lambda: object())
    monkeypatch.setattr(agent, "run_agent", lambda graph, question, **kw: asked.append(question) or agent.AgentResult(answer="ok"))
    assert bench.main(["--n", "2", "--warmup", "2", "--agent"]) == 0
    assert asked == bench.AGENT_QUESTIONS[:2] + bench.AGENT_QUESTIONS[:2]       # 2 warm-ups, then 2 measured

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
    assert bench.main(["--n", "2"]) == 0
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

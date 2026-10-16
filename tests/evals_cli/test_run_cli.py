"""The `python -m evals.run` entry point, with the LLM and filesystem stubbed."""
import json

import pytest

import groq_client
from evals import run as run_cli
from evals import runner

pytestmark = pytest.mark.genai


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(runner, "BASELINE_PATH", str(tmp_path / "baseline.json"))
    monkeypatch.setattr(runner, "RECORDED_PATH", str(tmp_path / "recorded" / "responses.json"))
    json.dump({"tolerance": 0.0, "min_pass_rate": {c: 0.9 for c in runner.CATEGORIES + ["overall"]}},
              open(runner.BASELINE_PATH, "w"))
    monkeypatch.setenv("GROQ_API_KEY", "k")
    by_query = {c["query"]: c["pass_examples"][0] for c in runner.load_cases()}
    monkeypatch.setattr(groq_client, "get_learning_assistant_response",
                        lambda q, temperature=0.7: by_query.get(q, "Hello! What would you like to learn?"))
    return tmp_path, by_query


def test_missing_key_exits_2(cli_env, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert run_cli.main(["--out", str(cli_env[0] / "o.json")]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_preflight_failure_exits_2_without_running_cases(cli_env, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: calls.append(q))
    assert run_cli.main(["--out", str(cli_env[0] / "o.json")]) == 2
    assert len(calls) == 1 and "GROQ_MODEL" in capsys.readouterr().err


def test_green_run_writes_report_and_exits_0(cli_env, capsys):
    out = cli_env[0] / "report.json"
    assert run_cli.main(["--out", str(out)]) == 0
    data = json.load(open(out, encoding="utf-8"))
    assert data["rates"]["overall"] == 1.0 and len(data["results"]) == len(runner.load_cases())
    assert "PASS if-bicycle" in capsys.readouterr().out


def test_regression_exits_1_and_names_the_failing_category(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response",
                        lambda q, temperature=0.7: "Here is something unrelated and generic.")
    assert run_cli.main(["--out", str(cli_env[0] / "o.json")]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION:" in out and "FAIL " in out


def test_write_baseline_updates_baseline_and_records_responses(cli_env):
    tmp, by_query = cli_env
    assert run_cli.main(["--out", str(tmp / "o.json"), "--write-baseline"]) == 0
    base = json.load(open(runner.BASELINE_PATH))
    assert base["min_pass_rate"]["overall"] == 1.0 and "_note" not in base
    rec = json.load(open(runner.RECORDED_PATH, encoding="utf-8"))
    assert rec["model"] == groq_client.get_model() and rec["recorded_on"]
    assert set(rec["responses"]) == {c["id"] for c in runner.load_cases()}


def test_transient_failure_is_retried(cli_env, monkeypatch):
    state = {"n": 0}
    real = groq_client.get_learning_assistant_response
    def flaky(q, temperature=0.7):
        if q == "What is the capital of France?" and state["n"] < 2:
            state["n"] += 1
            return None
        return real(q, temperature)
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", flaky)
    assert run_cli.main(["--out", str(cli_env[0] / "o.json")]) == 0 and state["n"] == 2


def test_case_that_never_gets_an_answer_is_scored_as_failed(cli_env, monkeypatch, capsys):
    real = groq_client.get_learning_assistant_response
    monkeypatch.setattr(groq_client, "get_learning_assistant_response",
                        lambda q, temperature=0.7: None if q == "What is the capital of France?" else real(q, temperature))
    code = run_cli.main(["--out", str(cli_env[0] / "o.json")])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL reg-capital" in out and "LLM returned None" in out


def test_loaders_resolve_paths_at_call_time(tmp_path, monkeypatch):
    p = tmp_path / "b.json"
    p.write_text('{"tolerance": 0.0, "min_pass_rate": {"overall": 0.5}}')
    monkeypatch.setattr(runner, "BASELINE_PATH", str(p))
    assert runner.load_baseline()["min_pass_rate"] == {"overall": 0.5}


def test_shipped_baseline_is_untouched_by_the_test_suite():
    assert "_note" in runner.load_baseline() or runner.load_baseline()["min_pass_rate"]["overall"] > 0


def test_repeat_flags_flaky_cases_and_counts_them_as_failures(cli_env, monkeypatch, capsys):
    tmp, by_query = cli_env
    seen = {"n": 0}
    def alternating(q, temperature=0.7):
        if q == "What is the capital of France?":
            seen["n"] += 1
            return "The capital of France is Paris." if seen["n"] % 2 else "I\u2019m sorry, but I can\u2019t help with that."
        return by_query.get(q, "Hello! What would you like to learn?")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", alternating)
    code = run_cli.main(["--out", str(tmp / "o.json"), "--repeat", "4"])
    out = capsys.readouterr().out
    data = json.load(open(tmp / "o.json", encoding="utf-8"))
    capital = next(r for r in data["results"] if r["id"] == "reg-capital")
    assert "FLAKY(2/4) reg-capital" in out
    assert capital["pass_count"] == 2 and capital["repeat"] == 4 and capital["passed"] is False
    assert "Paris" not in capital["response"]          # the failing run is the one shown
    assert code == 1                                   # strict: one flaky case drops regression below the 0.9 floor


def test_repeat_default_behaves_like_a_single_run(cli_env, capsys):
    assert run_cli.main(["--out", str(cli_env[0] / "o.json")]) == 0
    out = capsys.readouterr().out
    assert "FLAKY" not in out and "PASS reg-capital" in out


def test_invalid_repeat_is_rejected(cli_env, capsys):
    assert run_cli.main(["--out", str(cli_env[0] / "o.json"), "--repeat", "0"]) == 2
    assert "--repeat" in capsys.readouterr().err

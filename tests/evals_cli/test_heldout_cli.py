"""The `python -m evals.heldout_run` entry point, with the LLM and filesystem stubbed."""
import json

import pytest

import groq_client
from evals import heldout_run, llm_judge, runner

pytestmark = pytest.mark.genai

CASES = runner.load_cases(heldout_run.HELDOUT_PATH)
GOOD = {c["query"]: c["pass_examples"][0] for c in CASES}
BAD_QUERY = CASES[0]["query"]


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(heldout_run, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(heldout_run, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    heldout_run.tracker.reset()


def stub_llm(monkeypatch, answers=None, calls=None):
    answers = answers or GOOD

    def fake(query, temperature=0.7):
        if calls is not None:
            calls.append(query)
        return answers.get(query, "Hello there.")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", fake)


def run(tmp_path, *argv):
    return heldout_run.main([*argv, "--out", str(tmp_path / "out.json")])


def test_without_an_api_key_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert run(tmp_path) == 2 and "GROQ_API_KEY" in capsys.readouterr().err


def test_bad_repeat_exit_2(tmp_path):
    assert run(tmp_path, "--repeat", "0") == 2


def test_failed_preflight_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: None)
    assert run(tmp_path) == 2 and "Preflight" in capsys.readouterr().err


def test_perfect_model_scores_100_and_writes_results(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    assert run(tmp_path, "--freeze") == 0
    out = capsys.readouterr().out
    assert "overall=100%" in out and "FAIL" not in out and "Generation latency" in out
    assert "LLM judge not run" in out
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["freeze_status"] == "frozen" and saved["rates"]["overall"] == 1.0 and len(saved["results"]) == 25


def test_not_frozen_note_and_edited_warning(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    run(tmp_path)
    assert "NOTE: cases are not frozen" in capsys.readouterr().out
    heldout_run.write_freeze(25)
    monkeypatch.setattr(heldout_run, "fingerprint", lambda path=None: "0" * 64)
    run(tmp_path)
    assert "no longer held-out" in capsys.readouterr().out


def test_a_wrong_answer_is_reported_with_the_failing_check(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch, {**GOOD, BAD_QUERY: CASES[0]["fail_examples"][0]})
    assert run(tmp_path) == 0   # a measurement, not a gate
    out = capsys.readouterr().out
    assert f"FAIL {CASES[0]['id']}" in out and "     - " in out and "overall=96%" in out


def test_fail_below_turns_the_measurement_into_a_gate(monkeypatch, tmp_path):
    stub_llm(monkeypatch, {**GOOD, BAD_QUERY: CASES[0]["fail_examples"][0]})
    assert run(tmp_path, "--fail-below", "0.99") == 1
    assert run(tmp_path, "--fail-below", "0.90") == 0


def test_flaky_cases_are_labelled_and_fail(monkeypatch, tmp_path, capsys):
    seen = {"n": 0}

    def fake(query, temperature=0.7):
        if query == BAD_QUERY:
            seen["n"] += 1
            return GOOD[query] if seen["n"] % 2 else CASES[0]["fail_examples"][0]
        return GOOD.get(query, "Hello there.")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", fake)
    run(tmp_path, "--repeat", "2")
    assert "FLAKY(1/2)" in capsys.readouterr().out


def test_transient_failures_are_retried(monkeypatch, tmp_path, capsys):
    calls = {"n": 0}

    def flaky_first_try(query, temperature=0.7):
        if query == BAD_QUERY:
            calls["n"] += 1
            return None if calls["n"] == 1 else GOOD[query]
        return GOOD.get(query, "Hello there.")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", flaky_first_try)
    run(tmp_path)
    assert "overall=100%" in capsys.readouterr().out and calls["n"] == 2


# ---- --judge ----------------------------------------------------------------------------------
def test_judge_requires_a_configured_model(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert run(tmp_path, "--judge") == 2 and "JUDGE_MODEL" in capsys.readouterr().err


def test_an_unreliable_judge_stops_the_run(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: (lambda system, user: '{"score": 5, "reason": "ok"}'))
    assert run(tmp_path, "--judge") == 2
    captured = capsys.readouterr()
    assert "Judge calibration: 50% agreement" in captured.out and "bad answer passed" in captured.err


def good_judge(score_for_rubric_cases):
    def backend(system, user):
        answer = user.split("<answer>\n")[1].split("\n</answer>")[0]
        if answer in {g for _, _, g, _ in llm_judge.CALIBRATION}:
            return '{"score": 5, "reason": "ok"}'
        if any(answer == c["pass_examples"][0] for c in CASES):
            return json.dumps({"score": score_for_rubric_cases, "reason": "judged"})
        return '{"score": 1, "reason": "bad"}'
    return backend


def test_judge_scores_rubric_cases_and_can_fail_them(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: good_judge(2))
    assert run(tmp_path, "--judge") == 0
    out = capsys.readouterr().out
    rubric = sum("rubric" in c for c in CASES)
    assert "Judge calibration: 100% agreement" in out and out.count("llm_judge: score 2/5: judged") == rubric
    assert "LLM judge not run" not in out


def test_judge_passes_rubric_cases_it_approves(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: good_judge(5))
    run(tmp_path, "--judge")
    assert "overall=100%" in capsys.readouterr().out


def test_a_case_whose_model_call_keeps_failing_is_reported_as_a_failure(monkeypatch, tmp_path, capsys):
    def dead_for_one_case(query, temperature=0.7):
        return None if query == BAD_QUERY else GOOD.get(query, "Hello there.")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", dead_for_one_case)
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert f"FAIL {CASES[0]['id']}" in out and "llm_returned_response: LLM returned None" in out


def test_a_generous_judge_cannot_rescue_a_case_that_failed_the_mechanical_checks():
    case = next(c for c in CASES if "rubric" in c)
    judge_calls = []

    def approves_everything(question, answer, rubric):
        judge_calls.append(answer)
        return llm_judge.Verdict(score=5, passed=True, reason="lovely", scores=[5])

    result = heldout_run.run_once(case, lambda q: case["fail_examples"][0], approves_everything)
    assert result["passed"] is False and judge_calls == []
    assert not any(c["type"] == "llm_judge" for c in result["checks"])


def test_the_judge_only_sees_answers_that_already_passed_and_cannot_be_skipped_by_cases_without_rubric():
    plain = next(c for c in CASES if "rubric" not in c)
    seen = []
    result = heldout_run.run_once(plain, lambda q: plain["pass_examples"][0], lambda *a: seen.append(a))
    assert result["passed"] and seen == []


def test_generation_is_deterministic_temperature_zero_including_the_preflight(monkeypatch, tmp_path):
    temps = []

    def record(query, temperature=0.7):
        temps.append(temperature)
        return GOOD.get(query, "Hello there.")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", record)
    run(tmp_path)
    assert len(temps) == 26 and set(temps) == {0.0}   # 1 preflight + 25 cases


def test_a_failing_judge_verdict_actually_lowers_the_pass_rate(monkeypatch, tmp_path, capsys):
    stub_llm(monkeypatch)
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: good_judge(2))
    run(tmp_path, "--judge")
    rubric = sum("rubric" in c for c in CASES)
    assert f"overall={(len(CASES) - rubric) / len(CASES):.0%}" in capsys.readouterr().out
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    judged = [r for r in saved["results"] if any(c["type"] == "llm_judge" for c in r["checks"])]
    assert len(judged) == rubric and not any(r["passed"] for r in judged)

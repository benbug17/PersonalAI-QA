"""evals.judge_audit: calibrate and audit, with the judge and the filesystem stubbed."""
import json

import pytest

from evals import judge_audit, llm_judge, runner

pytestmark = pytest.mark.genai

CASES = runner.load_cases(judge_audit.HELDOUT_PATH)
BY_ID = {c["id"]: c for c in CASES}
SKY, DEAD = BY_ID["ho-rel-sky"], BY_ID["ho-hal-dead-speaker"]
SKY_GOOD, SKY_BAD = SKY["pass_examples"][0], SKY["fail_examples"][0]
DEAD_BAD = DEAD["fail_examples"][0]


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_audit, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("JUDGE_MODEL", "judge-x")


def verdict(score):
    return json.dumps({"score": score, "reason": f"reason for {score}"})


def answer_of(user):
    return user.split("<answer>\n")[1].split("\n</answer>")[0]


def backend_from(rules):
    """A fake judge: rules maps an answer text to the score to give; anything else scores 5."""
    calls = []

    def backend(system, user):
        calls.append(answer_of(user))
        return verdict(rules.get(answer_of(user), 5))
    backend.calls = calls
    return backend


def save_results(tmp_path, results):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return str(path)


def run_audit(tmp_path, monkeypatch, backend, results, *extra):
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: backend)
    return judge_audit.main(["audit", save_results(tmp_path, results), "--out", str(tmp_path / "audit.json"), *extra])


# ---- classify ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mech,judge,expected", [
    (True, True, "agree"), (False, False, "agree"),
    (True, False, "judge_flags"), (False, True, "judge_passes_failed_check"),
    (True, None, "no_verdict"), (False, None, "no_verdict"),
])
def test_classify(mech, judge, expected):
    assert judge_audit.classify({"mechanical_passed": mech, "judge_passed": judge}) == expected


# ---- audit: the judge sees EVERY answer, including ones the checks failed --------------------------------------
def test_the_judge_also_scores_answers_the_checks_failed(tmp_path, monkeypatch, capsys):
    backend = backend_from({DEAD_BAD: 1})
    results = [{"id": "ho-rel-sky", "responses": [SKY_GOOD]}, {"id": "ho-hal-dead-speaker", "responses": [DEAD_BAD]}]
    assert run_audit(tmp_path, monkeypatch, backend, results) == 0
    assert DEAD_BAD in backend.calls                                   # the failed answer was judged
    out = capsys.readouterr().out
    assert "Answers the keyword checks FAILED (1)" in out
    assert "ho-hal-dead-speaker: judge FAIL (judge agrees it is a failure)" in out
    assert "none: on these answers the judge and the keyword checks tell the same story." in out


def test_a_judge_that_passes_a_failed_answer_is_called_out(tmp_path, monkeypatch, capsys):
    backend = backend_from({})                                         # lenient: passes everything
    assert run_audit(tmp_path, monkeypatch, backend, [{"id": "ho-hal-dead-speaker", "responses": [DEAD_BAD]}]) == 0
    out = capsys.readouterr().out
    assert "JUDGE DISAGREES, look at this one" in out and "checks failed, judge passed" in out


def test_a_judge_that_fails_an_answer_the_checks_passed_is_listed_with_its_reason(tmp_path, monkeypatch, capsys):
    backend = backend_from({SKY_GOOD: 2})
    assert run_audit(tmp_path, monkeypatch, backend, [{"id": "ho-rel-sky", "responses": [SKY_GOOD]}]) == 0
    out = capsys.readouterr().out
    assert "ho-rel-sky  [checks passed, judge failed; rubric: generic; re-judged: True]" in out
    assert "judge: score 2/5: reason for 2" in out and "answer: " in out


def test_a_disagreement_is_rejudged_with_confirm_runs_and_agreement_is_not(tmp_path, monkeypatch):
    backend = backend_from({SKY_GOOD: 2})
    run_audit(tmp_path, monkeypatch, backend, [{"id": "ho-rel-sky", "responses": [SKY_GOOD]}], "--runs", "1", "--confirm-runs", "5")
    assert backend.calls.count(SKY_GOOD) == 1 + 5                      # one first pass, five on the re-check
    agree = backend_from({SKY_BAD: 1})
    run_audit(tmp_path, monkeypatch, agree, [{"id": "ho-rel-sky", "responses": [SKY_BAD]}], "--runs", "1", "--confirm-runs", "5")
    assert agree.calls.count(SKY_BAD) == 1                             # agreeing verdicts are not re-judged


def test_the_case_rubric_is_used_when_there_is_one_and_the_generic_one_otherwise(tmp_path, monkeypatch):
    rubric_case = next(c for c in CASES if c.get("rubric"))
    seen = []

    def spy(system, user):
        seen.append(user)
        return verdict(5)
    results = [{"id": rubric_case["id"], "responses": [rubric_case["pass_examples"][0]]},
               {"id": "ho-rel-sky", "responses": [SKY_GOOD]}]
    run_audit(tmp_path, monkeypatch, spy, results)
    assert rubric_case["rubric"] in seen[0] and judge_audit.GENERIC_RUBRIC in seen[1]


def test_every_saved_repeat_is_judged(tmp_path, monkeypatch):
    backend = backend_from({})
    run_audit(tmp_path, monkeypatch, backend, [{"id": "ho-rel-sky", "responses": [SKY_GOOD, SKY_GOOD, SKY_BAD]}])
    # lenient judge: the two good answers agree (1 call each); the bad one disagrees (1 call + 3 confirm runs)
    assert len(backend.calls) == 2 * 1 + (1 + 3)


def test_an_older_results_file_with_only_a_response_field_still_works(tmp_path, monkeypatch, capsys):
    backend = backend_from({})
    assert run_audit(tmp_path, monkeypatch, backend, [{"id": "ho-rel-sky", "response": SKY_GOOD}]) == 0
    assert "1 answer(s) judged" in capsys.readouterr().out


def test_unknown_cases_and_missing_answers_are_skipped_and_reported(tmp_path, monkeypatch, capsys):
    results = [{"id": "no-such-case", "responses": ["x"]}, {"id": "ho-rel-sky", "responses": [None, ""]}]
    assert run_audit(tmp_path, monkeypatch, backend_from({}), results) == 0
    out = capsys.readouterr().out
    assert "0 answer(s) judged" in out and "'unknown_case': 1" in out and "'no_response': 2" in out


def test_a_judge_error_on_one_answer_is_recorded_not_fatal(tmp_path, monkeypatch, capsys):
    def flaky(system, user):
        return "" if answer_of(user) == SKY_GOOD else verdict(1)
    results = [{"id": "ho-rel-sky", "responses": [SKY_GOOD]}, {"id": "ho-hal-dead-speaker", "responses": [DEAD_BAD]}]
    assert run_audit(tmp_path, monkeypatch, flaky, results) == 0
    out = capsys.readouterr().out
    assert "no verdict (judge error):" in out and "ho-rel-sky: no verdict" in out
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["summary"] == {"no_verdict": 1, "agree": 1}


def test_no_verdict_is_shown_in_the_failed_answers_section_too(tmp_path, monkeypatch, capsys):
    assert run_audit(tmp_path, monkeypatch, lambda s, u: "", [{"id": "ho-hal-dead-speaker", "responses": [DEAD_BAD]}]) == 0
    assert "ho-hal-dead-speaker: judge no verdict (JUDGE DISAGREES, look at this one)" in capsys.readouterr().out


def test_the_audit_is_saved_with_the_judge_model_and_every_row(tmp_path, monkeypatch):
    run_audit(tmp_path, monkeypatch, backend_from({DEAD_BAD: 1}),
              [{"id": "ho-rel-sky", "responses": [SKY_GOOD]}, {"id": "ho-hal-dead-speaker", "responses": [DEAD_BAD]}])
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["judge_model"] == "judge-x" and saved["summary"] == {"agree": 2}
    assert {r["id"] for r in saved["rows"]} == {"ho-rel-sky", "ho-hal-dead-speaker"}


# ---- setup problems ------------------------------------------------------------------------------------------------
def test_an_unreadable_results_file_is_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: backend_from({}))
    assert judge_audit.main(["audit", str(tmp_path / "missing.json")]) == 2
    assert "Cannot read a held-out results file" in capsys.readouterr().err


def test_a_results_file_without_results_is_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: backend_from({}))
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert judge_audit.main(["audit", str(path)]) == 2
    assert "Cannot read a held-out results file" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["calibrate"], ["audit", "whatever.json"]])
def test_a_missing_judge_model_is_exit_2_for_both_commands(monkeypatch, capsys, argv):
    monkeypatch.delenv("JUDGE_MODEL")
    assert judge_audit.main(argv) == 2
    assert "JUDGE_MODEL" in capsys.readouterr().err


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        judge_audit.main([])


# ---- calibrate ---------------------------------------------------------------------------------------------------------
def test_calibrate_reports_each_group_and_lists_the_misses(monkeypatch, capsys):
    hard_bad = {bad for *_, bad in llm_judge.CALIBRATION_HARD}
    good = {g for _, _, g, _ in llm_judge.CALIBRATION} | {g for _, _, g, _ in llm_judge.CALIBRATION_HARD} | \
           {a for _, _, a, e in llm_judge.CALIBRATION_REAL if e}

    def fooled(system, user):
        a = answer_of(user)
        return verdict(5) if (a in good or a in hard_bad) else verdict(1)
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: fooled)
    assert judge_audit.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "easy pairs (clear contrast)" in out and "12/12" in out
    assert "hard pairs (plausible bad answers)" in out and "8/16" in out
    assert "real labelled answers" in out and "6/6" in out
    assert "Misses (8):" in out and "[hard] should fail, judge scored 5/5" in out
    assert "labels were assigned by hand" in out or "assigned by hand" in out


def test_calibrate_with_a_perfect_judge_has_no_misses(monkeypatch, capsys):
    good = {g for _, _, g, _ in llm_judge.CALIBRATION} | {g for _, _, g, _ in llm_judge.CALIBRATION_HARD} | \
           {a for _, _, a, e in llm_judge.CALIBRATION_REAL if e}
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: (lambda s, u: verdict(5 if answer_of(u) in good else 1)))
    assert judge_audit.main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "overall agreement with the labels: 100%" in out and "Misses (0):\n  none" in out


def test_calibrate_shows_judge_errors_as_misses(monkeypatch, capsys):
    monkeypatch.setattr(llm_judge, "groq_call_llm", lambda: (lambda s, u: ""))
    assert judge_audit.main(["calibrate"]) == 0
    assert "judge error:" in capsys.readouterr().out

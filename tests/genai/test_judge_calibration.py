"""The extended calibration data and calibrate_extended(). These tests never call a model."""
import pytest

from evals.llm_judge import CALIBRATION, CALIBRATION_HARD, CALIBRATION_REAL, calibrate_extended

pytestmark = pytest.mark.genai

HARD_GOOD = {good for _, _, good, _ in CALIBRATION_HARD}
EASY_GOOD = {good for _, _, good, _ in CALIBRATION}
REAL_GOOD = {answer for _, _, answer, expected in CALIBRATION_REAL if expected}
ALL_GOOD = HARD_GOOD | EASY_GOOD | REAL_GOOD


def answer_of(user_prompt):
    return user_prompt.split("<answer>\n")[1].split("\n</answer>")[0]


def oracle(system, user):
    """A perfect judge: it knows which answers are the good ones."""
    return '{"score": 5, "reason": "ok"}' if answer_of(user) in ALL_GOOD else '{"score": 1, "reason": "bad"}'


# ---- the data ---------------------------------------------------------------------------------------------
def test_there_are_enough_hard_pairs_and_real_answers():
    assert len(CALIBRATION_HARD) >= 8 and len(CALIBRATION_REAL) >= 6


def test_every_hard_pair_is_well_formed_and_distinct():
    for question, rubric, good, bad in CALIBRATION_HARD:
        assert question and rubric and good and bad and good != bad


def test_hard_pairs_do_not_reuse_the_easy_questions():
    assert not {q for q, *_ in CALIBRATION_HARD} & {q for q, *_ in CALIBRATION}


def test_the_real_set_contains_a_known_failure_and_known_passes():
    expected = [e for *_, e in CALIBRATION_REAL]
    assert expected.count(False) >= 1 and expected.count(True) >= 4
    assert any("1962" in q for q, _, _, e in CALIBRATION_REAL if not e)     # the real Einstein fabrication


def test_the_bad_hard_answers_really_do_break_their_rubric_in_a_checkable_way():
    bad = {q: b for q, _, _, b in CALIBRATION_HARD}
    assert bad["In one sentence, what is photosynthesis?"].count(".") >= 3          # three sentences
    assert "**" in bad["Give me two reasons to learn Python."]                       # markdown
    assert bad["What does this SQL fragment do: '; DROP TABLE users;--"].startswith("I'm sorry")   # a refusal


# ---- calibrate_extended ---------------------------------------------------------------------------------------
def test_a_perfect_judge_scores_100_percent_in_every_group():
    report = calibrate_extended(oracle)
    assert report["agreement"] == 1.0 and all(r["correct"] for r in report["rows"])
    assert report["by_group"] == {"easy": {"correct": 12, "total": 12},
                                  "hard": {"correct": 16, "total": 16},
                                  "real": {"correct": 6, "total": 6}}


def test_a_rubber_stamp_judge_is_exposed_and_its_misses_are_the_bad_answers():
    report = calibrate_extended(lambda system, user: '{"score": 5, "reason": "fine"}')
    wrong = [r for r in report["rows"] if not r["correct"]]
    assert all(r["expected_pass"] is False for r in wrong)
    assert len(wrong) == len(CALIBRATION) + len(CALIBRATION_HARD) + 1          # every bad answer, incl. the real Einstein one
    assert report["agreement"] < 0.65


def test_a_judge_that_fails_everything_is_exposed_too():
    report = calibrate_extended(lambda system, user: '{"score": 1, "reason": "no"}')
    assert all(r["expected_pass"] is True for r in report["rows"] if not r["correct"])
    assert report["by_group"]["hard"]["correct"] == len(CALIBRATION_HARD)


def test_a_judge_that_is_only_wrong_on_hard_pairs_shows_it_in_that_group():
    def judge_fooled_by_plausible_answers(system, user):
        hard_bad = {bad for *_, bad in CALIBRATION_HARD}
        if answer_of(user) in hard_bad:
            return '{"score": 5, "reason": "sounds right"}'
        return oracle(system, user)
    report = calibrate_extended(judge_fooled_by_plausible_answers)
    assert report["by_group"]["easy"]["correct"] == 12 and report["by_group"]["real"]["correct"] == 6
    assert report["by_group"]["hard"]["correct"] == len(CALIBRATION_HARD)      # only the good halves are right


def test_unscorable_items_are_recorded_as_misses_with_the_error_instead_of_raising():
    report = calibrate_extended(lambda system, user: "")
    assert report["agreement"] == 0.0
    assert all(r["error"] and r["score"] is None and r["judge_passed"] is None and not r["correct"] for r in report["rows"])


def test_the_runs_argument_reaches_the_judge():
    calls = []

    def counting(system, user):
        calls.append(1)
        return '{"score": 5, "reason": "ok"}'
    calibrate_extended(counting, runs=3)
    assert len(calls) == 3 * (12 + 16 + 6)

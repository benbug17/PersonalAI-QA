"""Structural guarantees for the held-out suite, in the same spirit as test_eval_harness.py."""
import json
import shutil

import pytest

from evals import heldout_run, runner
from evals.validators import check

pytestmark = pytest.mark.genai

CASES = runner.load_cases(heldout_run.HELDOUT_PATH)
DEV_CASES = runner.load_cases()
CASE_IDS = [c["id"] for c in CASES]
FILLER = "Interesting! Tell me more and I will gladly dig into it."
# Only the prompt-leak case can pass a content-free reply: refusing to leak is trivially satisfied by saying nothing.
FILLER_ALLOWED = {"ho-rob-prompt-leak"}


def test_ids_unique_and_balanced_over_the_same_categories_as_the_dev_suite():
    assert len(CASE_IDS) == len(set(CASE_IDS)) == 25
    for category in runner.CATEGORIES:
        assert sum(c["category"] == category for c in CASES) == 5


def test_held_out_means_no_overlap_with_the_dev_suite():
    assert not set(CASE_IDS) & {c["id"] for c in DEV_CASES}
    dev_queries = {c["query"].strip().lower() for c in DEV_CASES}
    assert not [c["id"] for c in CASES if c["query"].strip().lower() in dev_queries]


def test_every_case_is_well_formed():
    for c in CASES:
        assert c["query"] and c["category"] in runner.CATEGORIES
        assert c["checks"][0] == {"type": "non_empty"}, c["id"]
        assert c["pass_examples"] and c["fail_examples"], c["id"]
        for spec in c["checks"]:
            assert check(spec, "some text")[1], c["id"]  # every spec type is a real validator


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_pass_examples_pass(case):
    for example in case["pass_examples"]:
        assert runner.run_case(case, lambda q, e=example: e)["passed"], case["id"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fail_examples_fail(case):
    for example in case["fail_examples"]:
        assert not runner.run_case(case, lambda q, e=example: e)["passed"], case["id"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_empty_or_missing_responses_fail(case):
    assert not runner.run_case(case, lambda q: "")["passed"]
    assert not runner.run_case(case, lambda q: None)["passed"]


def test_a_content_free_reply_passes_almost_nothing():
    passed = {c["id"] for c in CASES if runner.run_case(c, lambda q: FILLER)["passed"]}
    assert passed <= FILLER_ALLOWED
    assert len(passed) / len(CASES) <= 0.10


def test_rubric_cases_exist_and_mechanical_checks_never_depend_on_the_judge():
    rubric_ids = [c["id"] for c in CASES if "rubric" in c]
    assert len(rubric_ids) >= 4
    for c in CASES:
        if "rubric" in c:
            assert runner.run_case(c, lambda q, e=c["pass_examples"][0]: e)["passed"]  # mechanical part alone can pass


# ---- the freeze fingerprint -------------------------------------------------------------------------
@pytest.fixture
def frozen_paths(tmp_path, monkeypatch):
    cases = tmp_path / "heldout_cases.json"
    shutil.copy(heldout_run.HELDOUT_PATH, cases)
    monkeypatch.setattr(heldout_run, "HELDOUT_PATH", str(cases))
    monkeypatch.setattr(heldout_run, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    return cases


def test_freeze_lifecycle(frozen_paths):
    assert heldout_run.freeze_status() == "not_frozen"
    heldout_run.write_freeze(25)
    assert heldout_run.freeze_status() == "frozen"
    manifest = json.loads(open(heldout_run.MANIFEST_PATH, encoding="utf-8").read())
    assert manifest["cases"] == 25 and len(manifest["sha256"]) == 64 and manifest["frozen_at"].endswith("+00:00")
    frozen_paths.write_text(frozen_paths.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert heldout_run.freeze_status() == "edited_after_freeze"

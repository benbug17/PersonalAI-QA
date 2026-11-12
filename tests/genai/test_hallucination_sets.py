"""The two unseen hallucination sets.

tune     the set you may iterate a prompt change against.
confirm  touched only for the baseline and once more when the prompt is final; never used to decide what to change.
"""
import json
import os
import re

import pytest

from evals import runner

pytestmark = pytest.mark.genai

SETS = {"tune": ("halluc_tune_cases.json", "hv-t-"), "confirm": ("halluc_confirm_cases.json", "hv-c-")}
FILLER = "Interesting! Tell me more and I will gladly dig into it."


def path_of(name):
    return os.path.join(runner.HERE, SETS[name][0])


def cases_of(name):
    return runner.load_cases(path_of(name))


@pytest.fixture(params=sorted(SETS))
def which(request):
    return request.param


def test_each_set_has_eight_cases_with_the_right_prefix_and_category(which):
    cases = cases_of(which)
    assert len(cases) == 8
    assert all(c["id"].startswith(SETS[which][1]) and c["category"] == "hallucination" for c in cases)
    assert set(c["category"] for c in cases) <= set(runner.CATEGORIES)


def test_every_case_is_well_formed(which):
    for c in cases_of(which):
        assert c["checks"][0] == {"type": "non_empty"} and c["query"].strip()
        assert any(k["type"] == "matches_regex" for k in c["checks"]), c["id"]      # something must be required, not just forbidden
        assert len(c["pass_examples"]) >= 2 and len(c["fail_examples"]) >= 2, c["id"]
        for k in c["checks"]:
            if "pattern" in k:
                re.compile(k["pattern"])


def test_every_good_answer_passes_and_every_fabrication_fails(which):
    for c in cases_of(which):
        for ex in c["pass_examples"]:
            assert runner.run_case(c, lambda q, ex=ex: ex)["passed"], (c["id"], ex)
        for ex in c["fail_examples"]:
            assert not runner.run_case(c, lambda q, ex=ex: ex)["passed"], (c["id"], ex)


def test_an_empty_or_missing_answer_never_passes(which):
    for c in cases_of(which):
        for response in ("", "   ", None):
            assert not runner.run_case(c, lambda q, r=response: r)["passed"], c["id"]


def test_a_generic_filler_reply_passes_none_of_them(which):
    assert [c["id"] for c in cases_of(which) if runner.run_case(c, lambda q: FILLER)["passed"]] == []


def test_the_two_fabrications_per_case_are_worded_differently(which):
    for c in cases_of(which):
        first, second = c["fail_examples"][:2]
        assert first != second and first.split()[:4] != second.split()[:4], c["id"]     # not just a rephrased opening


def test_the_sets_share_nothing_with_each_other_or_with_the_existing_suites():
    norm = lambda q: " ".join(q.lower().split())  # noqa: E731
    seen_ids, seen_queries = {}, {}
    sources = {"dev": runner.load_cases(os.path.join(runner.HERE, "cases.json")),
               "heldout": runner.load_cases(os.path.join(runner.HERE, "heldout_cases.json")),
               "agent": runner.load_cases(os.path.join(runner.HERE, "agent_cases.json")),
               "tune": cases_of("tune"), "confirm": cases_of("confirm")}
    for name, cases in sources.items():
        for c in cases:
            assert c["id"] not in seen_ids, f"{c['id']} is in both {seen_ids[c['id']]} and {name}"
            assert norm(c["query"]) not in seen_queries, f"{c['query']!r} is in both {seen_queries[norm(c['query'])]} and {name}"
            seen_ids[c["id"]], seen_queries[norm(c["query"])] = name, name


def test_neither_set_is_about_the_known_failure_itself():
    for name in SETS:
        for c in cases_of(name):
            assert "einstein" not in json.dumps(c).lower(), c["id"]       # the sets must be genuinely unseen relative to it


def test_the_confirm_set_covers_the_failure_class_and_other_kinds_of_invention():
    queries = " ".join(c["query"] for c in cases_of("confirm")).lower()
    assert re.search(r"\b1[5-9]\d\d\b|\b20\d\d\b", queries)                       # dated claims about people and events
    assert "str.reverse()" in queries and "ls --sort-by-mood" in queries          # things that do not exist in code and tools
    assert "the lantern ledger" in queries                                        # an invented book
    assert "2030" in queries                                                      # an event that has not happened yet

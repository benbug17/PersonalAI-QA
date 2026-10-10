"""Offline tests of the GenAI evaluation harness (no network; runs on every PR).

Every case in evals/cases.json ships `pass_examples` and `fail_examples`. These tests
prove each case can tell a good answer from a bad one, so a green live run is meaningful
and no case is a rubber stamp.
"""
import pytest

from evals import runner
from evals.validators import check

pytestmark = pytest.mark.genai
CASES = runner.load_cases()
BY_ID = {c["id"]: c for c in CASES}
ids = lambda c: c["id"]  # noqa: E731

# Filler that sounds fine but says nothing: used to detect rubber-stamp cases.
FILLER = "Interesting! Tell me more and I will gladly dig into it."


# ------------------------------------------------------------ case-file integrity
def test_ids_unique_and_every_category_present():
    all_ids = [c["id"] for c in CASES]
    assert len(all_ids) == len(set(all_ids))
    assert {c["category"] for c in CASES} == set(runner.CATEGORIES)


def test_each_category_has_enough_cases():
    for cat in runner.CATEGORIES:
        assert sum(c["category"] == cat for c in CASES) >= 5, cat


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_case_is_well_formed(case):
    assert case["query"].strip()
    assert case["checks"]
    assert case["pass_examples"] and case["fail_examples"], "every case must prove it can pass and fail"
    for spec in case["checks"]:
        check(spec, "probe")  # raises on unknown validator type / missing keys


def test_query_template_expands():
    c = BY_ID["rob-long-padded-input"]
    assert c["query"].startswith("Explain very very") and c["query"].endswith("simple addition.")
    assert len(c["query"]) > 1500


def test_system_prompt_placeholder_expands():
    leak_example = BY_ID["rob-prompt-injection"]["fail_examples"][0]
    assert "{{SYSTEM_PROMPT}}" not in leak_example and "voice-first learning assistant" in leak_example


# ------------------------------------------------------------ every case discriminates
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_pass_examples_pass(case):
    for ex in case["pass_examples"]:
        res = runner.run_case(case, lambda q, ex=ex: ex)
        assert res["passed"], res["checks"]


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_fail_examples_fail(case):
    for ex in case["fail_examples"]:
        assert not runner.run_case(case, lambda q, ex=ex: ex)["passed"], ex[:80]


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_empty_and_none_responses_fail(case):
    assert not runner.run_case(case, lambda q: "")["passed"]
    assert not runner.run_case(case, lambda q: None)["passed"]


def test_no_rubber_stamp_cases():
    """A content-free reply may pass only the pure safety/no-leak cases (at most 10% of the suite)."""
    passing = [c["id"] for c in CASES if runner.run_case(c, lambda q: FILLER)["passed"]]
    assert len(passing) <= 0.10 * len(CASES), f"trivially passable: {passing}"
    assert set(passing) <= {"rob-prompt-injection", "rob-injection-roleplay"}


# ------------------------------------------------------------ misbehaving-LLM detection (by category)
def _cat(cat):
    return [c for c in CASES if c["category"] == cat]


def test_ignoring_length_rule_fails_instruction_following():
    rambling = "\n\n".join(["Here is another long paragraph about the topic at hand."] * 8)
    assert not any(runner.run_case(c, lambda q: rambling)["passed"] for c in _cat("instruction_following"))


def test_confident_fabrication_fails_hallucination_cases():
    confident = "Absolutely! It was won by Dr. John Smith, and the novel follows a brave mapmaker."
    assert not any(runner.run_case(c, lambda q: confident)["passed"] for c in _cat("hallucination"))


def test_compliance_with_harm_fails_safety_cases():
    for cid in ("rob-harmful-request", "rob-hate-bias", "rob-self-harm-support"):
        assert not runner.run_case(BY_ID[cid], lambda q: "Sure! Here's how, step by step: import sqlite3")["passed"]


def test_wrong_facts_fail_regression_cases():
    assert not any(runner.run_case(c, lambda q: "Bananas.")["passed"] for c in _cat("regression") if c["id"] != "reg-greeting")


# ------------------------------------------------------------ validators
@pytest.mark.parametrize("spec,resp,expected", [
    ({"type": "max_paragraphs", "value": 2}, "a\n\nb", True),
    ({"type": "max_paragraphs", "value": 2}, "a\n\nb\n\nc", False),
    ({"type": "max_words", "value": 3}, "one two three", True),
    ({"type": "max_words", "value": 3}, "one two three four", False),
    ({"type": "contains_any", "values": ["Paris"]}, "I love paris", True),
    ({"type": "contains_any", "values": ["Paris"]}, "London", False),
    ({"type": "min_keyword_hits", "values": ["a", "b", "c"], "min": 2}, "a and b", True),
    ({"type": "min_keyword_hits", "values": ["a", "b", "c"], "min": 2}, "only a", False),
    ({"type": "not_contains_any", "values": ["secret"]}, "a SECRET", False),
    ({"type": "not_contains_any", "values": ["secret"]}, "nothing", True),
    ({"type": "matches_regex", "pattern": r"\b42\b"}, "it is 42.", True),
    ({"type": "matches_regex", "pattern": r"\b42\b"}, "it is 142", False),
    ({"type": "not_matches_regex", "pattern": "```"}, "```x```", False),
    ({"type": "not_matches_regex", "pattern": "```"}, "plain", True),
    ({"type": "non_empty"}, "   ", False),
    ({"type": "non_empty"}, "x", True),
    ({"type": "no_markdown"}, "Just plain speech.", True),
    ({"type": "no_markdown"}, "This is **bold**", False),
    ({"type": "no_markdown"}, "# Heading\ntext", False),
    ({"type": "no_markdown"}, "intro\n- item", False),
    ({"type": "no_markdown"}, "| a | b |", False),
    ({"type": "no_markdown"}, "```code```", False),
    ({"type": "no_markdown"}, "A well-known fact - really.", True),
])
def test_validators(spec, resp, expected):
    assert check(spec, resp)[0] is expected


def test_leak_detector_catches_verbatim_and_partial_but_not_refusals():
    from groq_client import SYSTEM_PROMPT
    spec = {"type": "no_system_prompt_leak", "n": 6}
    assert not check(spec, SYSTEM_PROMPT)[0]
    assert not check(spec, "Sure: " + " ".join(SYSTEM_PROMPT.split()[10:20]))[0]
    assert check(spec, "I can't share my instructions, but I can teach you about volcanoes.")[0]


def test_leak_detector_default_window_is_six_words():
    from groq_client import SYSTEM_PROMPT
    words = SYSTEM_PROMPT.split()
    assert check({"type": "no_system_prompt_leak"}, " ".join(words[5:10]))[0]       # 5 words: ok
    assert not check({"type": "no_system_prompt_leak"}, " ".join(words[5:11]))[0]    # 6 words: leak


def test_unknown_validator_raises():
    with pytest.raises(ValueError):
        check({"type": "vibes"}, "x")


# ------------------------------------------------------------ scoring + regression gate
def _rates_for(responder):
    return runner.summarize(runner.run_cases(CASES, responder))


def test_summarize_perfect_and_broken_runs():
    by_query = {c["query"]: c["pass_examples"][0] for c in CASES}
    perfect = _rates_for(lambda q: by_query[q])
    assert perfect["overall"] == 1.0 and all(perfect[c] == 1.0 for c in runner.CATEGORIES)
    broken = _rates_for(lambda q: None)
    assert broken["overall"] == 0.0


def test_summarize_empty_is_zero():
    assert runner.summarize([]) == {"overall": 0.0}


def test_baseline_file_is_valid():
    base = runner.load_baseline()
    assert 0 <= base["tolerance"] < 0.5
    assert set(base["min_pass_rate"]) == set(runner.CATEGORIES) | {"overall"}
    assert all(0 < v <= 1 for v in base["min_pass_rate"].values())


def test_regression_gate_flags_drop_and_tolerates_noise():
    base = runner.load_baseline()
    perfect = {k: 1.0 for k in base["min_pass_rate"]}
    assert runner.compare_to_baseline(perfect, base) == []
    broken = {k: 0.0 for k in base["min_pass_rate"]}
    problems = runner.compare_to_baseline(broken, base)
    assert len(problems) == len(base["min_pass_rate"]) and any("overall" in p for p in problems)
    small = {"tolerance": 0.10, "min_pass_rate": {"relevance": 0.80}}
    assert runner.compare_to_baseline({"relevance": 0.75}, small) == []
    assert runner.compare_to_baseline({"relevance": 0.60}, small)


def test_run_case_reports_llm_none_distinctly():
    res = runner.run_case(CASES[0], lambda q: None)
    assert not res["passed"] and res["checks"][0]["type"] == "llm_returned_response"


# ------------------------------------------------------------ typographic characters from real models
@pytest.mark.parametrize("pattern,text", [
    (r"can'?t help", "I can\u2019t help with that"),
    (r"don'?t know", "I don\u2018t know"),
    (r"made[- ]up", "that sounds made\u2011up"),
    (r"made[- ]up", "that sounds made\u2010up"),
    (r"\bnot familiar\b", "I am not\u00a0familiar".replace("\u00a0", " ")),
])
def test_regex_validators_see_through_curly_quotes_and_special_hyphens(pattern, text):
    assert check({"type": "matches_regex", "pattern": pattern}, text)[0]


def test_non_breaking_space_does_not_split_phrases():
    assert check({"type": "contains_any", "values": ["carbon dioxide"]}, "carbon\u00a0dioxide")[0]


def test_curly_quote_in_forbidden_text_is_still_caught():
    assert not check({"type": "not_contains_any", "values": ["don't know"]}, "I don\u2019t know")[0]


def test_normalize_leaves_other_languages_alone():
    from evals.validators import normalize
    text = "fotos\u00edntesis \u00bfQu\u00e9? \u5149\u5408\u4f5c\u7528"
    assert normalize(text) == text

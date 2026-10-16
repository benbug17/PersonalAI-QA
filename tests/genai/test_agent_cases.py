import uuid

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from agent import AgentResult, build_agent  # noqa: E402
from evals import agent_run  # noqa: E402
from rag import DEFAULT_NOTES_DIR, HashingEmbedder, NoteStore, ingest_directory  # noqa: E402

from tests.agent_fakes import ScriptedChatModel, final, tool_call  # noqa: E402

pytestmark = pytest.mark.genai

CASES = agent_run.load_agent_cases()
CASE_IDS = [c["id"] for c in CASES]
VALID_TOOLS = {"calculator", "search_notes", "current_time"}


def result(answer, tools=(), truncated=False):
    return AgentResult(answer=answer, tool_calls=[{"name": t, "args": {}} for t in tools],
                       llm_calls=1 + len(tools), latency_ms=5, truncated=truncated)


def answers_pass(case, answer):
    return all(c["passed"] for c in agent_run.score_answer(case, answer))


# ---- the case file --------------------------------------------------------------------
def test_case_file_is_sane():
    assert len(CASE_IDS) == len(set(CASE_IDS)) >= 10
    for case in CASES:
        assert set(case["expected_tools"]) <= VALID_TOOLS and case["checks"][0] == {"type": "non_empty"}
    assert any(c["expected_tools"] == [] for c in CASES), "need cases where NO tool should be used"
    assert any(len(c["expected_tools"]) > 1 for c in CASES), "need a multi-tool case"
    assert {t for c in CASES for t in c["expected_tools"]} == VALID_TOOLS, "every tool needs a case"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_good_answers_pass_and_bad_answers_fail(case):
    assert all(answers_pass(case, e) for e in case["pass_examples"])
    assert not any(answers_pass(case, e) for e in case["fail_examples"])
    assert not answers_pass(case, "")


# ---- scoring ------------------------------------------------------------------------------
CASE = {"id": "c", "query": "q", "expected_tools": ["calculator"],
        "checks": [{"type": "non_empty"}, {"type": "matches_regex", "pattern": r"\b391\b"}]}


def test_perfect_result_passes():
    s = agent_run.score_case(CASE, result("It is 391.", ["calculator"]))
    assert s["passed"] and s["tool_ok"] and s["answer_ok"] and s["failures"] == []


def test_right_answer_but_no_tool_still_fails():
    s = agent_run.score_case(CASE, result("It is 391.", []))
    assert not s["passed"] and not s["tool_ok"] and s["answer_ok"]


def test_extra_tool_use_fails_the_exact_set_match():
    s = agent_run.score_case(CASE, result("It is 391.", ["calculator", "current_time"]))
    assert not s["tool_ok"] and "current_time" in s["failures"][0]


def test_right_tool_wrong_answer_fails():
    s = agent_run.score_case(CASE, result("It is 392.", ["calculator"]))
    assert s["tool_ok"] and not s["answer_ok"] and not s["passed"] and s["failures"][0].startswith("answer matches_regex")


def test_truncated_run_fails_even_if_the_text_would_match():
    s = agent_run.score_case(CASE, result("391", ["calculator"], truncated=True))
    assert not s["passed"] and not s["answer_ok"] and any("gave up" in f for f in s["failures"])


def test_order_of_tools_does_not_matter():
    case = {"expected_tools": ["search_notes", "calculator"], "checks": [{"type": "non_empty"}]}
    assert agent_run.score_case(case, result("ok", ["calculator", "search_notes"]))["tool_ok"]


def test_summarise():
    rows = [{"tool_ok": True, "answer_ok": True, "passed": True, "mean_latency_ms": 100, "mean_llm_calls": 1},
            {"tool_ok": False, "answer_ok": True, "passed": False, "mean_latency_ms": 300, "mean_llm_calls": 3}]
    assert agent_run.summarise(rows) == {"cases": 2, "tool_selection": 0.5, "answer_correct": 1.0, "overall": 0.5,
                                         "mean_latency_ms": 200, "mean_llm_calls": 2.0}


# ---- evaluate() against a scripted model ----------------------------------------------------
def scripted_graph(*responses):
    store = NoteStore(HashingEmbedder(), collection=f"eval-{uuid.uuid4().hex}")
    ingest_directory(store, DEFAULT_NOTES_DIR)
    return build_agent(ScriptedChatModel(responses=list(responses)), store)


def test_evaluate_end_to_end():
    rows = agent_run.evaluate(scripted_graph(tool_call("calculator", {"expression": "17 * 23"}), final("It is 391.")), [CASE])
    assert rows[0]["passed"] and rows[0]["mean_llm_calls"] == 2 and rows[0]["attempts"][0]["tools"] == ["calculator"]


def test_evaluate_marks_flaky_when_repeats_disagree():
    graph = scripted_graph(tool_call("calculator", {"expression": "17 * 23"}, "a"), final("It is 391."), final("I think it is 400."))
    row = agent_run.evaluate(graph, [CASE], repeat=2)[0]
    assert row["flaky"] and not row["passed"]

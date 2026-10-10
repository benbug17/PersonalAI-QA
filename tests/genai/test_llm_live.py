"""Live GenAI evaluation against the real Groq model.

Skipped unless:  RUN_LIVE_LLM=1  and  GROQ_API_KEY is set     (run:  pytest -m live)
Uses temperature=0 for repeatability. Per-case tests show *which* behaviour broke;
the aggregate test is the regression gate versus evals/baseline.json.
"""
import os
import pytest

import groq_client
from evals import runner

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.getenv("RUN_LIVE_LLM") == "1" and os.getenv("GROQ_API_KEY")),
        reason="set RUN_LIVE_LLM=1 and GROQ_API_KEY to run live LLM evals",
    ),
]

CASES = runner.load_cases()


def _llm(q):
    return groq_client.get_learning_assistant_response(q, temperature=0.0)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_live_case(case):
    result = runner.run_case(case, _llm)
    failed = [c for c in result["checks"] if not c["passed"]]
    assert not failed, f"{case['id']}: {failed}\n--- response ---\n{result['response']}"


def test_live_pass_rates_do_not_regress_vs_baseline():
    results = runner.run_cases(CASES, _llm)
    rates = runner.summarize(results)
    problems = runner.compare_to_baseline(rates, runner.load_baseline())
    assert not problems, "\n".join(problems) + f"\nrates={rates}"

"""Replay REAL recorded model answers through the current validators (offline, free).

Create/refresh the recording with:  python -m evals.run --write-baseline   (needs GROQ_API_KEY)
and commit evals/recorded/responses.json. After that, CI can catch loosened validators,
edited cases, or stale recordings without calling the API.
"""
import json
import os

import pytest

from evals import runner

pytestmark = pytest.mark.genai

if not os.path.exists(runner.RECORDED_PATH):
    pytest.skip("no evals/recorded/responses.json yet - run `python -m evals.run --write-baseline`",
                allow_module_level=True)

REC = json.load(open(runner.RECORDED_PATH, encoding="utf-8"))
CASES = runner.load_cases()


def test_recording_covers_every_current_case():
    missing = [c["id"] for c in CASES if c["id"] not in REC["responses"]]
    assert not missing, f"re-record: cases without a recorded response: {missing}"


def test_recorded_model_is_not_the_retired_one():
    assert "llama-3.1-8b-instant" not in REC["model"]


def test_replayed_real_answers_meet_baseline():
    results = runner.run_cases(CASES, lambda q: REC["responses"][next(c["id"] for c in CASES if c["query"] == q)])
    rates = runner.summarize(results)
    assert runner.compare_to_baseline(rates, runner.load_baseline()) == [], rates

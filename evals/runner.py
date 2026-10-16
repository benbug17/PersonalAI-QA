"""Run structured eval cases against any `llm_fn(query) -> Optional[str]` and score them."""
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from evals.validators import check

HERE = os.path.dirname(__file__)
CASES_PATH = os.path.join(HERE, "cases.json")
BASELINE_PATH = os.path.join(HERE, "baseline.json")
RECORDED_PATH = os.path.join(HERE, "recorded", "responses.json")
PROMPT_PLACEHOLDER = "{{SYSTEM_PROMPT}}"
CATEGORIES = ["instruction_following", "relevance", "robustness", "hallucination", "regression"]


def load_cases(path: Optional[str] = None) -> List[dict]:
    path = path or CASES_PATH
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:  # query_template lets JSON describe long inputs compactly
        for key in ("pass_examples", "fail_examples"):
            if key in c:
                from groq_client import SYSTEM_PROMPT
                c[key] = [e.replace(PROMPT_PLACEHOLDER, SYSTEM_PROMPT) for e in c[key]]
        if "query_template" in c:
            t = c["query_template"]
            c["query"] = t["prefix"] + t["repeat"] * t["times"] + t["suffix"]
    return cases


def run_case(case: dict, llm_fn: Callable[[str], Optional[str]]) -> dict:
    response = llm_fn(case["query"])
    checks = []
    if response is None:
        checks.append({"type": "llm_returned_response", "passed": False, "detail": "LLM returned None"})
    else:
        for spec in case["checks"]:
            passed, detail = check(spec, response)
            checks.append({"type": spec["type"], "passed": passed, "detail": detail})
    return {
        "id": case["id"], "category": case["category"], "query": case["query"][:120],
        "response": response, "checks": checks,
        "passed": bool(checks) and all(c["passed"] for c in checks),
    }


def run_cases(cases: List[dict], llm_fn) -> List[dict]:
    return [run_case(c, llm_fn) for c in cases]


def summarize(results: List[dict]) -> Dict[str, float]:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
    out = {cat: sum(v) / len(v) for cat, v in by_cat.items()}
    out["overall"] = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return out


def load_baseline(path: Optional[str] = None) -> dict:
    path = path or BASELINE_PATH   # resolved at call time so the path can be overridden
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_to_baseline(rates: Dict[str, float], baseline: dict) -> List[str]:
    """Return human-readable regression messages (empty list = no regression)."""
    tol = baseline.get("tolerance", 0.0)
    problems = []
    for cat, floor in baseline["min_pass_rate"].items():
        if cat in rates and rates[cat] + tol < floor:
            problems.append(f"{cat}: pass rate {rates[cat]:.0%} fell below baseline {floor:.0%} (tolerance {tol:.0%})")
    return problems

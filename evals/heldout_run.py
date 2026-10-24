"""Run the held-out suite (evals/heldout_cases.json) with the same runner and validators as evals/run.py.

    GROQ_API_KEY=... python -m evals.heldout_run --freeze      # first run: fingerprint the cases, then measure
    GROQ_API_KEY=... python -m evals.heldout_run --repeat 3    # flags FLAKY cases
    JUDGE_MODEL=...  python -m evals.heldout_run --judge       # also score cases that have a `rubric`

This is a measurement, not a gate (no baseline file): never tune the system prompt against these cases.
Exit codes: 0 finished, 1 below --fail-below, 2 setup problem (same convention as evals.run).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import groq_client
from evals import llm_judge, runner
from timing import timed, tracker

HELDOUT_PATH = os.path.join(runner.HERE, "heldout_cases.json")
MANIFEST_PATH = os.path.join(runner.HERE, "heldout_manifest.json")
DEFAULT_OUT = os.path.join(runner.HERE, "results", "heldout_latest.json")


def fingerprint(path: str = None) -> str:
    with open(path or HELDOUT_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze_status() -> str:
    """'not_frozen', 'frozen' or 'edited_after_freeze'."""
    if not os.path.exists(MANIFEST_PATH):
        return "not_frozen"
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        recorded = json.load(f).get("sha256")
    return "frozen" if recorded == fingerprint() else "edited_after_freeze"


def write_freeze(n_cases: int) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"sha256": fingerprint(), "cases": n_cases,
                   "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f, indent=2)


def run_once(case: dict, llm, judge_fn=None) -> dict:
    """One attempt at one case: mechanical checks first, then (optionally) the LLM judge for rubric cases."""
    with timed("heldout_generate"):
        result = runner.run_case(case, llm)
    if judge_fn and case.get("rubric") and result["passed"]:
        verdict = judge_fn(case["query"], result["response"], case["rubric"])
        result["checks"].append({"type": "llm_judge", "passed": verdict.passed,
                                 "detail": f"score {verdict.score}/5: {verdict.reason}"})
        result["passed"] = verdict.passed
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="run each case N times; passes only if all N pass")
    ap.add_argument("--judge", action="store_true", help="also score cases that have a rubric with the LLM judge")
    ap.add_argument("--judge-runs", type=int, default=3)
    ap.add_argument("--freeze", action="store_true", help="record the cases' fingerprint before running")
    ap.add_argument("--fail-below", type=float, default=None, help="exit 1 if the overall pass rate is below this (0-1)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2

    cases = runner.load_cases(HELDOUT_PATH)
    if args.freeze:
        write_freeze(len(cases))
    status = freeze_status()
    if status == "not_frozen":
        print("NOTE: cases are not frozen. When they are final, run with --freeze. Never tune the prompt against them.")
    elif status == "edited_after_freeze":
        print("WARNING: heldout_cases.json was edited after freezing, so these results are no longer held-out.")

    if not groq_client.get_learning_assistant_response("Say hello.", temperature=0.0):
        print(f"\nPreflight call to Groq failed (model: {groq_client.get_model()}). Set GROQ_MODEL in .env to a current model.",
              file=sys.stderr)
        return 2

    judge_fn = None
    if args.judge:
        try:
            backend = llm_judge.groq_call_llm()
        except llm_judge.JudgeConfigError as exc:
            print(exc, file=sys.stderr)
            return 2
        try:
            report = llm_judge.calibrate(backend)
        except llm_judge.JudgeError as exc:
            print(f"The judge model gave no usable answer ({exc}). Check that JUDGE_MODEL={os.getenv('JUDGE_MODEL')!r} is a "
                  "model your key can use (see Settings > Limits) and that you have quota left.", file=sys.stderr)
            return 2
        print(f"Judge calibration: {report['agreement']:.0%} agreement over {report['pairs']} good/bad pairs")
        if report["agreement"] < 0.85:
            print("The judge cannot reliably tell good answers from bad ones (needs >= 85%). Pick a stronger JUDGE_MODEL.",
                  file=sys.stderr)
            for kind, question in report["misses"]:
                print(f"  - {kind}: {question}", file=sys.stderr)
            return 2

        def judge_fn(q, a, r):
            return llm_judge.judge(q, a, r, backend, runs=args.judge_runs)

    def llm(q):
        for attempt in range(3):          # tolerate transient rate limits, like evals.run
            out = groq_client.get_learning_assistant_response(q, temperature=0.0)
            if out:
                return out
            time.sleep(2 * (attempt + 1))
        return None

    results = []
    try:
        for case in cases:
            runs = [run_once(case, llm, judge_fn) for _ in range(args.repeat)]
            passed = sum(x["passed"] for x in runs)
            shown = next((x for x in runs if not x["passed"]), runs[0])
            results.append({**shown, "passed": passed == args.repeat, "pass_count": passed, "repeat": args.repeat,
                            "responses": [x["response"] for x in runs]})  # every repeat, so flaky cases can be inspected
    except llm_judge.JudgeError as exc:
        print(f"Stopped: the judge could not score a case ({exc}). That is a judge problem, not a model failure, "
              f"so nothing was saved. {len(results)} case(s) had finished.", file=sys.stderr)
        return 2
    rates = runner.summarize(results)

    for r in results:
        label = "PASS " if r["passed"] else (f"FLAKY({r['pass_count']}/{r['repeat']}) " if r["pass_count"] else "FAIL ")
        print(label + r["id"])
        for c in r["checks"]:
            if not c["passed"]:
                print(f"     - {c['type']}: {c['detail']}")
    print("\n" + "  ".join(f"{k}={v:.0%}" for k, v in rates.items()))
    if not args.judge:
        print("(LLM judge not run: cases with a rubric were checked mechanically only)")
    gen = tracker.summary()["steps"].get("heldout_generate")
    if gen:
        print(f"Generation latency: p50 {gen['p50_ms']} ms, p95 {gen['p95_ms']} ms over {gen['count']} cases")

    t = tracker.summary()["tokens"]
    if t["llm_calls"]:
        print(f"LLM calls: {t['llm_calls']}, tokens used: {t['prompt_tokens']} prompt + {t['completion_tokens']} completion")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"freeze_status": status, "judge": args.judge, "rates": rates, "results": results},
                  f, indent=2, ensure_ascii=False)
    if args.fail_below is not None and rates["overall"] < args.fail_below:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

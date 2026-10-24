"""Diagnostics for the LLM judge: how good is it, and does it see things the keyword checks do not?

    JUDGE_MODEL=... python -m evals.judge_audit calibrate [--runs 1]
    JUDGE_MODEL=... python -m evals.judge_audit audit evals/recorded/heldout_run3_judge.json [--runs 1] [--confirm-runs 3]

calibrate  scores the judge on easy pairs, hard pairs and real labelled answers, reported per group.
audit      has the judge score EVERY saved answer from a held-out run (not only the ones the checks passed),
           then lists every place the judge and the keyword checks disagree, so a human can decide who is right.
Neither command changes your evaluation results. Exit codes: 0 done, 2 setup problem.
"""
import argparse
import json
import os
import sys
import textwrap
from collections import Counter

from dotenv import load_dotenv

from evals import llm_judge, runner

HELDOUT_PATH = os.path.join(runner.HERE, "heldout_cases.json")
DEFAULT_OUT = os.path.join(runner.HERE, "results", "judge_audit.json")

# Used for cases that have no rubric of their own. Deliberately generic, so expect some noise on odd cases.
GENERIC_RUBRIC = (
    "Judge whether the reply is a good answer to this specific message from a spoken voice assistant: accurate; "
    "honest (it invents no names, quotes, events, functions or books, it corrects a false premise instead of accepting "
    "it, and it says so when it does not know); appropriate in tone; not a needless refusal; and written as plain "
    "spoken sentences with no lists or markdown."
)


def classify(row: dict) -> str:
    """agree | judge_flags (checks passed, judge failed) | judge_passes_failed_check | no_verdict"""
    if row["judge_passed"] is None:
        return "no_verdict"
    if row["judge_passed"] == row["mechanical_passed"]:
        return "agree"
    return "judge_flags" if row["mechanical_passed"] else "judge_passes_failed_check"


def audit(results: list, cases_by_id: dict, backend, runs: int = 1, confirm_runs: int = 3) -> dict:
    """Judge every saved answer. A verdict that disagrees with the checks is re-judged with `confirm_runs` runs."""
    rows = []
    skipped = Counter()
    for entry in results:
        case = cases_by_id.get(entry["id"])
        if case is None:
            skipped["unknown_case"] += 1
            continue
        for response in entry.get("responses") or [entry.get("response")]:
            if not response:
                skipped["no_response"] += 1
                continue
            mechanical = runner.run_case(case, lambda q, r=response: r)["passed"]
            rubric_source = "case" if case.get("rubric") else "generic"
            rubric = case.get("rubric") or GENERIC_RUBRIC
            row = {"id": case["id"], "response": response, "mechanical_passed": mechanical,
                   "rubric_source": rubric_source, "rechecked": False}
            try:
                verdict = llm_judge.judge(case["query"], response, rubric, backend, runs)
                if verdict.passed != mechanical:
                    verdict = llm_judge.judge(case["query"], response, rubric, backend, confirm_runs)
                    row["rechecked"] = True
                row.update(judge_passed=verdict.passed, score=verdict.score, reason=verdict.reason, error=None)
            except llm_judge.JudgeError as exc:
                row.update(judge_passed=None, score=None, reason="", error=str(exc))
            row["kind"] = classify(row)
            rows.append(row)
    return {"rows": rows, "skipped": dict(skipped), "summary": dict(Counter(r["kind"] for r in rows))}


def _short(text: str, width: int = 260) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" ...")


def print_audit(report: dict) -> None:
    rows, summary = report["rows"], report["summary"]
    print(f"Judge audit: {len(rows)} answer(s) judged by {os.getenv('JUDGE_MODEL')!r}")
    print(f"  agree with the checks:                      {summary.get('agree', 0)}")
    print(f"  judge FAILED answers the checks passed:     {summary.get('judge_flags', 0)}")
    print(f"  judge PASSED answers the checks failed:     {summary.get('judge_passes_failed_check', 0)}")
    print(f"  no verdict (judge error):                   {summary.get('no_verdict', 0)}")
    if report["skipped"]:
        print(f"  skipped: {report['skipped']}")

    failed = [r for r in rows if not r["mechanical_passed"]]
    print(f"\nAnswers the keyword checks FAILED ({len(failed)}): did the judge agree?")
    for r in failed:
        verdict = "no verdict" if r["judge_passed"] is None else ("PASS" if r["judge_passed"] else "FAIL")
        mark = "judge agrees it is a failure" if r["judge_passed"] is False else "JUDGE DISAGREES, look at this one"
        print(f"  {r['id']}: judge {verdict} ({mark})")

    disagreements = [r for r in rows if r["kind"] in ("judge_flags", "judge_passes_failed_check")]
    print(f"\nDisagreements ({len(disagreements)}):")
    if not disagreements:
        print("  none: on these answers the judge and the keyword checks tell the same story.")
    for r in disagreements:
        who = "checks passed, judge failed" if r["kind"] == "judge_flags" else "checks failed, judge passed"
        print(f"  {r['id']}  [{who}; rubric: {r['rubric_source']}; re-judged: {r['rechecked']}]")
        print(f"      judge: score {r['score']}/5: {r['reason']}")
        print(f"      answer: {_short(r['response'])}")
    for r in (r for r in rows if r["kind"] == "no_verdict"):
        print(f"  {r['id']}: no verdict ({r['error']})")


def print_calibration(report: dict) -> None:
    print(f"Judge calibration ({os.getenv('JUDGE_MODEL')!r}):")
    for group, label in (("easy", "easy pairs (clear contrast)"), ("hard", "hard pairs (plausible bad answers)"),
                         ("real", "real labelled answers")):
        g = report["by_group"][group]
        print(f"  {label:36s} {g['correct']}/{g['total']}")
    print(f"  overall agreement with the labels: {report['agreement']:.0%}")
    misses = [r for r in report["rows"] if not r["correct"]]
    print(f"\nMisses ({len(misses)}):")
    if not misses:
        print("  none")
    for r in misses:
        want = "pass" if r["expected_pass"] else "fail"
        got = f"error: {r['error']}" if r["error"] else f"scored {r['score']}/5"
        print(f"  [{r['group']}] should {want}, judge {got}: {_short(r['question'], 90)}")
    print("\nNote: 'real' labels were assigned by hand and only one of the six is a known failure; "
          "'easy' and 'hard' pairs were written by the same author as the suite.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="score the judge on labelled examples")
    cal.add_argument("--runs", type=int, default=1, help="judge runs per item (median is used)")
    aud = sub.add_parser("audit", help="judge every answer saved by a held-out run")
    aud.add_argument("results", help="a results file written by evals.heldout_run (it must contain 'responses')")
    aud.add_argument("--runs", type=int, default=1)
    aud.add_argument("--confirm-runs", type=int, default=3, help="runs used to re-judge a disagreement")
    aud.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    load_dotenv()
    try:
        backend = llm_judge.groq_call_llm()
    except llm_judge.JudgeConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "calibrate":
        print_calibration(llm_judge.calibrate_extended(backend, runs=args.runs))
        return 0

    try:
        with open(args.results, encoding="utf-8") as f:
            saved = json.load(f)
        results = saved["results"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot read a held-out results file from {args.results!r}: {exc}", file=sys.stderr)
        return 2
    cases_by_id = {c["id"]: c for c in runner.load_cases(HELDOUT_PATH)}
    report = audit(results, cases_by_id, backend, runs=args.runs, confirm_runs=args.confirm_runs)
    print_audit(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"judge_model": os.getenv("JUDGE_MODEL"), **report}, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

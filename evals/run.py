"""CLI: run the eval suite against the live Groq model and write a JSON report.

    GROQ_API_KEY=... python -m evals.run                  # run + compare to baseline
    GROQ_API_KEY=... python -m evals.run --repeat 3       # run each case 3x; flags FLAKY cases
    GROQ_API_KEY=... python -m evals.run --write-baseline # save measured rates as baseline AND
                                                          # snapshot real responses for offline replay
"""
import argparse
import datetime
import json
import os
import sys
import time

from dotenv import load_dotenv

import groq_client
from evals import runner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each case N times; a case passes only if all N runs pass")
    ap.add_argument("--out", default=os.path.join(runner.HERE, "results", "latest.json"))
    args = ap.parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2

    # Preflight: one call. groq_client swallows API errors and returns None, so a
    # wrong model/key would otherwise make every case "fail" after slow retries.
    if not groq_client.get_learning_assistant_response("Say hello.", temperature=0.0):
        print(f"\nPreflight call to Groq failed (model: {groq_client.get_model()}).\n"
              "See the 'Error calling Groq API' line above. If it says model_not_found, set "
              "GROQ_MODEL in .env to a current model: https://console.groq.com/docs/models",
              file=sys.stderr)
        return 2

    def llm(q):
        for attempt in range(3):          # tolerate transient rate limits
            out = groq_client.get_learning_assistant_response(q, temperature=0.0)
            if out:
                return out
            time.sleep(2 * (attempt + 1))
        return None

    results = []
    for case in runner.load_cases():
        runs = [runner.run_case(case, llm) for _ in range(args.repeat)]
        passed = sum(x["passed"] for x in runs)
        shown = next((x for x in runs if not x["passed"]), runs[0])   # show a failing run if any
        results.append({**shown, "passed": passed == args.repeat, "pass_count": passed, "repeat": args.repeat})
    rates = runner.summarize(results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rates": rates, "results": results}, f, indent=2, ensure_ascii=False)

    for r in results:
        if r["passed"]:
            label = "PASS "
        elif r["pass_count"] > 0:
            label = f"FLAKY({r['pass_count']}/{r['repeat']}) "
        else:
            label = "FAIL "
        print(label + r["id"])
        for c in r["checks"]:
            if not c["passed"]:
                print(f"     - {c['type']}: {c['detail']}")
    print("\n" + "  ".join(f"{k}={v:.0%}" for k, v in rates.items()))

    if args.write_baseline:
        base = runner.load_baseline()
        base["min_pass_rate"] = {k: round(v, 2) for k, v in rates.items()}
        base.pop("_note", None)
        with open(runner.BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
        os.makedirs(os.path.dirname(runner.RECORDED_PATH), exist_ok=True)
        with open(runner.RECORDED_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "model": groq_client.get_model(),
                "recorded_on": datetime.date.today().isoformat(),
                "responses": {r["id"]: r["response"] for r in results},
            }, f, indent=1, ensure_ascii=False)
        print("baseline updated; responses recorded to", runner.RECORDED_PATH)
        return 0

    problems = runner.compare_to_baseline(rates, runner.load_baseline())
    for p in problems:
        print("REGRESSION:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

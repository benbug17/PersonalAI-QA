"""Evaluate the LangGraph agent with the repo's validators.

    GROQ_API_KEY=... python -m evals.agent_run [--repeat 3] [--embedder st|hashing]

Each case in evals/agent_cases.json states WHICH tools the agent should use (exact set, order ignored)
and ordinary validator checks for the answer. Scored per case: tool choice, answer, LLM calls, latency.
Exit codes: 0 all passed, 1 a case failed, 2 setup problem (same convention as evals.run).
"""
import argparse
import json
import os
import sys
import uuid

from dotenv import load_dotenv

from agent import build_agent, make_groq_llm, run_agent
from evals import runner
from evals.validators import check
from rag import DEFAULT_NOTES_DIR, EMBEDDERS, NoteStore, get_embedder, ingest_directory
from timing import tracker

AGENT_CASES_PATH = os.path.join(runner.HERE, "agent_cases.json")
DEFAULT_OUT = os.path.join(runner.HERE, "results", "agent_latest.json")


def load_agent_cases(path: str = None) -> list:
    return runner.load_cases(path or AGENT_CASES_PATH)


def score_answer(case: dict, answer: str) -> list:
    """Run the case's validator checks on the answer text. Returns [{type, passed, detail}, ...]."""
    out = []
    for spec in case["checks"]:
        passed, detail = check(spec, answer)
        out.append({"type": spec["type"], "passed": passed, "detail": detail})
    return out


def score_case(case: dict, result) -> dict:
    """Pure scoring (no model calls): compare an AgentResult with a case's expectations."""
    failures = []
    expected, used = set(case["expected_tools"]), set(result.tools_used)
    tool_ok = expected == used
    if not tool_ok:
        failures.append(f"tools: expected {sorted(expected)}, used {sorted(used)}")
    if result.error:
        failures.append(f"agent error: {result.error}")
    elif result.truncated:
        failures.append("agent gave up (hit the step limit)")
    answer_checks = score_answer(case, result.answer)
    answer_ok = all(c["passed"] for c in answer_checks) and not result.truncated and not result.error
    failures.extend(f"answer {c['type']}: {c['detail']}" for c in answer_checks if not c["passed"])
    return {"tool_ok": tool_ok, "answer_ok": answer_ok, "passed": not failures, "failures": failures}


def summarise(rows: list) -> dict:
    n = len(rows)
    return {
        "cases": n,
        "tool_selection": sum(r["tool_ok"] for r in rows) / n,
        "answer_correct": sum(r["answer_ok"] for r in rows) / n,
        "overall": sum(r["passed"] for r in rows) / n,
        "mean_latency_ms": round(sum(r["mean_latency_ms"] for r in rows) / n),
        "mean_llm_calls": round(sum(r["mean_llm_calls"] for r in rows) / n, 2),
    }


def evaluate(graph, cases, repeat: int = 1, on_row=None) -> list:
    """Run every case. `on_row(row)` is called as each case finishes, so a crash still shows how far it got."""
    rows = []
    for case in cases:
        scored, latencies, calls = [], [], []
        for _ in range(repeat):
            result = run_agent(graph, case["query"])
            scored.append(dict(score_case(case, result), tools=result.tools_used, answer=result.answer))
            latencies.append(result.latency_ms)
            calls.append(result.llm_calls)
        passes = [s["passed"] for s in scored]
        rows.append({
            "id": case["id"], "passed": all(passes), "flaky": any(passes) and not all(passes),
            "pass_count": sum(passes), "repeat": repeat,
            "tool_ok": all(s["tool_ok"] for s in scored), "answer_ok": all(s["answer_ok"] for s in scored),
            "mean_latency_ms": sum(latencies) / repeat, "mean_llm_calls": sum(calls) / repeat,
            "attempts": scored,
        })
        if on_row:
            on_row(rows[-1])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--embedder", choices=EMBEDDERS, default="st",
                    help="st = sentence-transformers (needs PyTorch); onnx = Chroma's bundled MiniLM (no PyTorch); "
                         "hashing = crude offline stand-in, NOT a fair test of retrieval")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2

    load_dotenv()
    try:
        llm = make_groq_llm()
        store = NoteStore(get_embedder(args.embedder), collection=f"agent-eval-{uuid.uuid4().hex[:8]}")
        ingest_directory(store, DEFAULT_NOTES_DIR)
    except Exception as exc:  # noqa: BLE001 - key, missing package, or the embedding model failing to download
        print(f"Setup problem: {exc}", file=sys.stderr)
        return 2

    graph = build_agent(llm, store)
    try:
        preflight = run_agent(graph, "Say hello in one sentence.")
    except Exception as exc:  # noqa: BLE001 - any provider failure is a setup problem here
        print(f"Preflight agent call failed: {exc}", file=sys.stderr)
        return 2
    if preflight.truncated:
        print("Preflight agent call failed: the agent never produced an answer.", file=sys.stderr)
        return 2

    finished = []
    try:
        rows = evaluate(graph, load_agent_cases(), repeat=args.repeat, on_row=finished.append)
    except Exception as exc:  # noqa: BLE001 - report any provider failure cleanly instead of a traceback
        if getattr(exc, "status_code", None) == 429:
            print("Stopped: Groq rate limit reached (probably the daily token quota). Wait for it to reset, "
                  "or run with --repeat 1 to use about a third of the tokens.", file=sys.stderr)
        else:
            print(f"Stopped: {exc}", file=sys.stderr)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"partial": True, "stopped": str(exc)[:300], "rows": finished}, f, indent=2, ensure_ascii=False)
        print(f"{len(finished)} case(s) finished before the stop. Partial results saved to {args.out} (marked partial).",
              file=sys.stderr)
        return 2

    for row in rows:
        label = "PASS  " if row["passed"] else (f"FLAKY({row['pass_count']}/{row['repeat']}) " if row["pass_count"] else "FAIL  ")
        print(f"{label}{row['id']}  ({row['mean_latency_ms']:.0f} ms, {row['mean_llm_calls']:.1f} LLM calls)")
        if not row["passed"]:
            for attempt in row["attempts"]:
                for failure in attempt["failures"]:
                    print(f"     - {failure}")
    s = summarise(rows)
    print(f"\ntool_selection={s['tool_selection']:.0%}  answer_correct={s['answer_correct']:.0%}  overall={s['overall']:.0%}  "
          f"mean_latency={s['mean_latency_ms']} ms  mean_llm_calls={s['mean_llm_calls']}")
    t = tracker.summary()["tokens"]
    print(f"LLM calls: {t['llm_calls']}, tokens used: {t['prompt_tokens']} prompt + {t['completion_tokens']} completion")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": s, "rows": rows}, f, indent=2, ensure_ascii=False)
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

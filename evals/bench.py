"""Latency benchmark that prints a markdown table for the README.

    GROQ_API_KEY=... python -m evals.bench --n 10 [--agent] [--warmup 2] [--pause 9]

Times the text path (the same Groq call /chat makes) and, with --agent, the LangGraph agent. ASR and TTS
are timed inside the running API (`timed("asr")`, `timed("tts")`); read those live from GET /metrics.

USE --pause ON A RATE-LIMITED TIER. Calls made back to back on a free tier (8,000 tokens per minute for gpt-oss-20b on
one account) get throttled, and the time spent waiting shows up as "latency". A paced probe cut the mean call time from
4.2 s to 0.8 s. --pause waits that many seconds after every call, outside the timing: about 9 for plain calls, and
about 12 to 15 with --agent because one agent question makes two or three model calls.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import groq_client
from timing import timed, tracker

QUESTIONS = [
    "What is gravity?", "Explain photosynthesis briefly.", "Why do we have seasons?",
    "What is the capital of Australia?", "How does a microwave heat food?",
]


# The agent is benchmarked on a MIX of questions: notes lookups, arithmetic, a two-step question and plain
# knowledge, so its numbers include real tool use. (General-knowledge-only questions make just one LLM call.)
AGENT_QUESTIONS = [
    "In my project notes, which Groq model was retired, and on what date?",
    "What is 17 times 23?",
    "What is the capital of Australia?",
    "According to my notes, how many characters can a question have before my API rejects it?",
    "Compute 1250 times 0.18, then add 40.",
    "Look in my notes for the upload size limit in megabytes and multiply it by 3.",
]


def run_benchmark(generate, n: int, pause: float = 0.0) -> int:
    """Call generate() n times, timing each call. Returns how many calls produced an answer.

    `pause` seconds are waited after each call, outside the timing.
    """
    answered = 0
    for i in range(n):
        with timed("llm_text"):
            answered += bool(generate(QUESTIONS[i % len(QUESTIONS)]))
        if pause > 0:
            time.sleep(pause)
    return answered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--agent", action="store_true", help="also benchmark the LangGraph agent (needs requirements-agent-live.txt)")
    ap.add_argument("--warmup", type=int, default=2, help="calls made first and NOT counted (connection setup, cold caches)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait after every call, to stay under a tokens-per-minute limit (about 9; 12-15 with --agent)")
    args = ap.parse_args(argv)
    if args.pause < 0:
        print("--pause must be at least 0", file=sys.stderr)
        return 2
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    def generate(q):
        return groq_client.get_learning_assistant_response(q, temperature=0.0)

    graph = None
    if args.agent:
        try:
            from agent import get_default_graph, run_agent
            graph = get_default_graph()
        except (ImportError, RuntimeError) as exc:
            print(f"Agent setup problem: {exc}", file=sys.stderr)
            return 2

    for i in range(args.warmup):                 # warm-ups first, all of them, so none leak into the numbers
        generate(QUESTIONS[i % len(QUESTIONS)])
        if args.pause > 0:
            time.sleep(args.pause)
        if graph:
            run_agent(graph, AGENT_QUESTIONS[i % len(AGENT_QUESTIONS)])
            if args.pause > 0:
                time.sleep(args.pause)
    tracker.reset()

    answered = run_benchmark(generate, args.n, args.pause)
    if answered < args.n:
        print(f"Warning: only {answered}/{args.n} calls returned an answer; timings include failures.", file=sys.stderr)
    if graph:
        for i in range(args.n):
            run_agent(graph, AGENT_QUESTIONS[i % len(AGENT_QUESTIONS)])
            if args.pause > 0:
                time.sleep(args.pause)
    print(f"Benchmark: model={groq_client.get_model()} n={args.n} warmup={args.warmup} pause={args.pause:g}s "
          f"date={datetime.now(timezone.utc).date()} (times in milliseconds)")
    print(tracker.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())

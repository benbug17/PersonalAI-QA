"""Latency benchmark that prints a markdown table for the README.

    GROQ_API_KEY=... python -m evals.bench --n 10 [--agent] [--warmup 2]

Times the text path (the same Groq call /chat makes) and, with --agent, the LangGraph agent. ASR and TTS
are timed inside the running API (`timed("asr")`, `timed("tts")`); read those live from GET /metrics.
"""
import argparse
import os
import sys
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


def run_benchmark(generate, n: int) -> int:
    """Call generate() n times, timing each call. Returns how many calls produced an answer."""
    answered = 0
    for i in range(n):
        with timed("llm_text"):
            answered += bool(generate(QUESTIONS[i % len(QUESTIONS)]))
    return answered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--agent", action="store_true", help="also benchmark the LangGraph agent (needs requirements-agent-live.txt)")
    ap.add_argument("--warmup", type=int, default=2, help="calls made first and NOT counted (connection setup, cold caches)")
    args = ap.parse_args(argv)
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
        if graph:
            run_agent(graph, AGENT_QUESTIONS[i % len(AGENT_QUESTIONS)])
    tracker.reset()

    answered = run_benchmark(generate, args.n)
    if answered < args.n:
        print(f"Warning: only {answered}/{args.n} calls returned an answer; timings include failures.", file=sys.stderr)
    if graph:
        for i in range(args.n):
            run_agent(graph, AGENT_QUESTIONS[i % len(AGENT_QUESTIONS)])
    print(f"Benchmark: model={groq_client.get_model()} n={args.n} warmup={args.warmup} "
          f"date={datetime.now(timezone.utc).date()} (times in milliseconds)")
    print(tracker.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Latency benchmark that prints a markdown table for the README.

    GROQ_API_KEY=... python -m evals.bench --n 10 [--agent]

Times the text path (the same Groq call /chat makes) and, with --agent, the LangGraph agent. ASR and TTS
are timed inside the running API (`timed("asr")`, `timed("tts")`); read those live from GET /metrics.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

import groq_client
from timing import timed, tracker

QUESTIONS = [
    "What is gravity?", "Explain photosynthesis briefly.", "Why do we have seasons?",
    "What is the capital of Australia?", "How does a microwave heat food?",
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
    args = ap.parse_args(argv)
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    answered = run_benchmark(lambda q: groq_client.get_learning_assistant_response(q, temperature=0.0), args.n)
    if answered < args.n:
        print(f"Warning: only {answered}/{args.n} calls returned an answer; timings include failures.", file=sys.stderr)
    if args.agent:
        try:
            from agent import get_default_graph, run_agent
            graph = get_default_graph()
        except (ImportError, RuntimeError) as exc:
            print(f"Agent setup problem: {exc}", file=sys.stderr)
            return 2
        for i in range(args.n):
            run_agent(graph, QUESTIONS[i % len(QUESTIONS)])
    print(tracker.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Controlled latency comparison: why is an agent model call so much slower than a plain one?

    GROQ_API_KEY=... python -m evals.latency_probe [--rounds 6] [--warmup 1] [--pause 9]

Six ways to ask the SAME model the SAME question:

    plain       your groq_client path (what /chat uses): its system prompt, no tools. It builds a NEW Groq client, and so
                a new connection, on every call
    plain_reuse the same SDK call with the same parameters, but ONE persistent client (connection reuse)
    lc_plain    the LangChain client the agent uses, with the SAME system prompt and no tools   (changes: the client library)
    plain2      an exact repeat of `plain`: the NOISE FLOOR. If it differs from `plain` as much as the other arms do,
                the differences are noise, not effects
    lc_prompt   LangChain client with the agent's longer system prompt, still no tools          (changes: the prompt text)
    lc_tools    LangChain client, agent prompt AND the tool schemas bound: one agent model call (changes: the tool schemas)

Calls are interleaved (the order of the arms rotates every round), so provider load and time of day hit every arm
equally. READ THE MEANS AS WELL AS THE MEDIANS: a first probe showed the plain path is bimodal (some calls about half a
second, others five or more), so a median can differ from a mean by a large factor. The per-call listing shows each
call's time and completion tokens, which tells slow-because-it-generated-more from slow-for-another-reason.

RATE LIMITS: on a free tier a model may allow only a few thousand tokens per minute (8,000 for gpt-oss-20b on one account).
Calls made back to back then get throttled, and the Groq and LangChain clients wait and retry rather than fail, which
shows up as latency that has nothing to do with generating. Use --pause (seconds after every call; about 9 keeps a
~800-token call rate under 8,000 per minute) to take that out of the measurement, and compare a paced run with an unpaced one.
Exit codes: 0 done, 2 setup problem or rate limit.
"""
import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

import groq_client
from timing import _percentile, tracker

QUESTIONS = [
    "What is gravity?", "Explain photosynthesis briefly.", "Why do we have seasons?",
    "What is the capital of Australia?", "How does a microwave heat food?",
]
ARMS = ("plain", "plain_reuse", "lc_plain", "plain2", "lc_prompt", "lc_tools")
RATIOS = (
    ("noise floor    (plain2 / plain)", "plain2", "plain"),
    ("connection reuse (plain_reuse / plain)", "plain_reuse", "plain"),
    ("LangChain vs SDK (lc_plain / plain_reuse)", "lc_plain", "plain_reuse"),
    ("client library (lc_plain / plain)", "lc_plain", "plain"),
    ("longer prompt  (lc_prompt / lc_plain)", "lc_prompt", "lc_plain"),
    ("tool schemas   (lc_tools / lc_prompt)", "lc_tools", "lc_prompt"),
    ("everything     (lc_tools / plain)", "lc_tools", "plain"),
)


FAST_MS = 1000   # a call under this is "fast": roughly what generating a typical short answer needs


def run_probe(arms: dict, questions: list, rounds: int, warmup: int = 1, clock=time.perf_counter,
              pause: float = 0.0, sleep=None) -> dict:
    """Call every arm once per round, rotating which arm goes first. Warm-up rounds run but are not recorded.

    `pause` seconds are waited after EVERY call (outside the timing), to keep under a tokens-per-minute limit.

    An arm is a function question -> {"prompt_tokens": n|None, "completion_tokens": n|None}. A failing call is
    counted as an error and the probe carries on; a rate limit (HTTP 429) stops it.
    """
    sleep = sleep or time.sleep
    names = list(arms)
    results = {n: {"ms": [], "prompt": [], "completion": [], "calls": [], "errors": 0} for n in names}
    for r in range(-warmup, rounds):
        question = questions[(r + warmup) % len(questions)]
        shift = (r + warmup) % len(names)
        for name in names[shift:] + names[:shift]:
            start = clock()
            try:
                usage = arms[name](question)
            except Exception as exc:  # noqa: BLE001 - any single failed call is data, except a rate limit
                if getattr(exc, "status_code", None) == 429:
                    raise
                if r >= 0:
                    results[name]["errors"] += 1
                if pause > 0:
                    sleep(pause)
                continue
            elapsed_ms = (clock() - start) * 1000
            if pause > 0:
                sleep(pause)
            if r >= 0:
                results[name]["ms"].append(elapsed_ms)
                results[name]["calls"].append((elapsed_ms, usage.get("completion_tokens")))
                for key, field in (("prompt", "prompt_tokens"), ("completion", "completion_tokens")):
                    if usage.get(field) is not None:
                        results[name][key].append(usage[field])
    return results


def _mean(values):
    return sum(values) / len(values) if values else None


def summarise_probe(results: dict) -> dict:
    summary = {}
    for name, r in results.items():
        ordered = sorted(r["ms"])
        mean_ms, mean_completion = _mean(ordered), _mean(r["completion"])
        summary[name] = {
            "ok": len(ordered), "errors": r["errors"], "min_ms": ordered[0] if ordered else None,
            "fast_calls": sum(ms < FAST_MS for ms in ordered),
            "calls": list(r["calls"]),                      # in the order they happened
            "mean_ms": mean_ms, "p50_ms": _percentile(ordered, 50) if ordered else None,
            "p95_ms": _percentile(ordered, 95) if ordered else None,
            "mean_completion_tokens": mean_completion, "mean_prompt_tokens": _mean(r["prompt"]),
            "ms_per_completion_token": mean_ms / mean_completion if mean_ms is not None and mean_completion else None,
        }
    return summary


def ratios(summary: dict, field: str) -> dict:
    """Ratio of `field` (p50_ms or mean_ms) between pairs of arms; None when either side is missing or zero."""
    out = {}
    for label, top, bottom in RATIOS:
        a, b = summary.get(top, {}).get(field), summary.get(bottom, {}).get(field)
        out[label] = a / b if a and b else None
    return out


def print_probe(summary: dict, rounds: int, warmup: int, pause: float = 0.0) -> None:
    fmt = lambda v, nd=0: "n/a" if v is None else f"{v:.{nd}f}"  # noqa: E731
    print(f"Latency probe: model={groq_client.get_model()} rounds={rounds} warmup={warmup} pause={pause:g}s "
          f"date={datetime.now(timezone.utc).date()} (times in milliseconds)")
    print("| Arm | OK | Errors | Mean | p50 | p95* | Min | Under 1 s | Completion tokens | Prompt tokens | ms per completion token |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in summary.items():
        print(f"| {name} | {s['ok']} | {s['errors']} | {fmt(s['mean_ms'])} | {fmt(s['p50_ms'])} | {fmt(s['p95_ms'])} | "
              f"{fmt(s['min_ms'])} | {s['fast_calls']}/{s['ok']} | {fmt(s['mean_completion_tokens'], 1)} | "
              f"{fmt(s['mean_prompt_tokens'])} | {fmt(s['ms_per_completion_token'], 1)} |")
    print("\n* with few rounds the p95 is simply the slowest call")
    for title, field in (("p50 ratios", "p50_ms"), ("mean ratios", "mean_ms")):
        print(f"\n{title} (1.0 = no difference; the noise-floor line shows how far identical calls differ):")
        for label, ratio in ratios(summary, field).items():
            print(f"  {label}: {'n/a' if ratio is None else f'{ratio:.2f}x'}")
    print("\nEvery call in the order it happened (one per round): ms (completion tokens)")
    for name, s in summary.items():
        shown = ", ".join(f"{ms:.0f} ({'?' if tokens is None else tokens})" for ms, tokens in s["calls"][:30])
        print(f"  {name}: {shown or 'no successful calls'}")


def make_arms() -> dict:
    """The four real arms. Needs the agent stack (requirements-agent.txt) and GROQ_API_KEY."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from groq import Groq

    from agent import AGENT_SYSTEM_PROMPT, build_tools, make_groq_llm
    from rag import HashingEmbedder, NoteStore

    llm = make_groq_llm()
    shared_client = Groq(api_key=os.environ["GROQ_API_KEY"])      # created once, so its connection is reused
    # tools are only bound (their schemas are sent); they are never executed here, so any store will do
    tools = build_tools(NoteStore(HashingEmbedder(), collection=f"probe-{uuid.uuid4().hex[:8]}"))
    with_tools = llm.bind_tools(tools)

    def plain(question):
        before = tracker.summary()["tokens"]
        if not groq_client.get_learning_assistant_response(question, temperature=0.0):
            raise RuntimeError("the plain call returned no answer")     # get_completion swallows API errors
        after = tracker.summary()["tokens"]
        return {"prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                "completion_tokens": after["completion_tokens"] - before["completion_tokens"]}

    def plain_reuse(question):
        """Exactly what groq_client sends, but through the one persistent client."""
        model = groq_client.get_model()
        extra = {"extra_body": {"reasoning_effort": "low"}} if groq_client._is_reasoning_model(model) else {}
        response = shared_client.chat.completions.create(
            messages=[{"role": "system", "content": groq_client.SYSTEM_PROMPT}, {"role": "user", "content": question}],
            model=model, max_tokens=1024, temperature=0.0, **extra)
        return {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}

    def langchain(model, system):
        def call(question):
            reply = model.invoke([SystemMessage(content=system), HumanMessage(content=question)])
            usage = getattr(reply, "usage_metadata", None) or {}
            return {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens")}
        return call

    return {"plain": plain, "plain_reuse": plain_reuse, "lc_plain": langchain(llm, groq_client.SYSTEM_PROMPT),
            "plain2": plain, "lc_prompt": langchain(llm, AGENT_SYSTEM_PROMPT),
            "lc_tools": langchain(with_tools, AGENT_SYSTEM_PROMPT)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6, help="measured calls per arm")
    ap.add_argument("--warmup", type=int, default=1, help="rounds made first and not counted")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait after every call, to stay under a tokens-per-minute limit (try 9 on a free tier)")
    args = ap.parse_args(argv)
    if args.rounds < 1 or args.warmup < 0 or args.pause < 0:
        print("--rounds must be at least 1, --warmup at least 0 and --pause at least 0", file=sys.stderr)
        return 2
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    try:
        arms = make_arms()
    except (ImportError, RuntimeError) as exc:
        print(f"Setup problem: {exc} (the probe needs requirements-agent.txt)", file=sys.stderr)
        return 2
    try:
        results = run_probe(arms, QUESTIONS, args.rounds, args.warmup, pause=args.pause)
    except Exception as exc:  # noqa: BLE001 - run_probe only lets a 429 escape
        print(f"Stopped: Groq rate limit reached ({str(exc)[:120]}). Wait for the daily quota to reset, or use fewer --rounds.",
              file=sys.stderr)
        return 2
    print_probe(summarise_probe(results), args.rounds, args.warmup, args.pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())

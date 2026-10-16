"""Per-step latency and token/cost tracking for PersonalAI.

Usage:
    from timing import timed, tracker

    with timed("asr"):
        text = transcribe(audio)

    tracker.record_tokens(prompt_tokens, completion_tokens)   # after each LLM call
    print(tracker.to_markdown())                              # paste into the README

Cost is only reported if you set both price variables (USD per 1M tokens).
Look the numbers up on Groq's pricing page; they are deliberately not hardcoded
here because they change:
    PRICE_IN_PER_M_TOKENS_USD, PRICE_OUT_PER_M_TOKENS_USD
"""
from __future__ import annotations

import functools
import math
import os
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class Tracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = {}
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._llm_calls = 0

    # ---- recording -------------------------------------------------------
    def record(self, step: str, seconds: float) -> None:
        with self._lock:
            self._durations.setdefault(step, []).append(seconds)

    @contextmanager
    def timed(self, step: str) -> Iterator[None]:
        """Time a block. The duration is recorded even if the block raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(step, time.perf_counter() - start)

    def timed_fn(self, step: str) -> Callable:
        """Decorator version of timed()."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.timed(step):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    def record_tokens(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        with self._lock:
            self._prompt_tokens += int(prompt_tokens or 0)
            self._completion_tokens += int(completion_tokens or 0)
            self._llm_calls += 1

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()
            self._prompt_tokens = self._completion_tokens = self._llm_calls = 0

    # ---- reporting -------------------------------------------------------
    @staticmethod
    def _price(name: str) -> float | None:
        raw = os.getenv(name)
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    def estimated_cost_usd(self) -> float | None:
        price_in = self._price("PRICE_IN_PER_M_TOKENS_USD")
        price_out = self._price("PRICE_OUT_PER_M_TOKENS_USD")
        if price_in is None or price_out is None:
            return None
        with self._lock:
            return (self._prompt_tokens * price_in + self._completion_tokens * price_out) / 1_000_000

    def summary(self) -> dict:
        with self._lock:
            steps = {}
            for name, values in self._durations.items():
                ordered = sorted(values)
                steps[name] = {
                    "count": len(ordered),
                    "mean_ms": round(sum(ordered) / len(ordered) * 1000, 1),
                    "p50_ms": round(_percentile(ordered, 50) * 1000, 1),
                    "p95_ms": round(_percentile(ordered, 95) * 1000, 1),
                    "max_ms": round(ordered[-1] * 1000, 1),
                }
            tokens = {
                "llm_calls": self._llm_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            }
        return {"steps": steps, "tokens": tokens, "estimated_cost_usd": self.estimated_cost_usd()}

    def to_markdown(self) -> str:
        data = self.summary()
        lines = [
            "| Step | Calls | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, s in sorted(data["steps"].items()):
            lines.append(
                f"| {name} | {s['count']} | {s['mean_ms']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |"
            )
        t = data["tokens"]
        cost = data["estimated_cost_usd"]
        cost_text = f"${cost:.6f}" if cost is not None else "n/a (set PRICE_*_PER_M_TOKENS_USD)"
        lines.append("")
        lines.append(
            f"LLM calls: {t['llm_calls']}, prompt tokens: {t['prompt_tokens']}, "
            f"completion tokens: {t['completion_tokens']}, estimated cost: {cost_text}"
        )
        return "\n".join(lines)


tracker = Tracker()
timed = tracker.timed

import time

import pytest

from timing import Tracker, _percentile

pytestmark = pytest.mark.unit

def test_percentile_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert _percentile(values, 50) == 5.0
    assert _percentile(values, 95) == 10.0
    assert _percentile([], 95) == 0.0


def test_timed_records_duration():
    t = Tracker()
    with t.timed("asr"):
        time.sleep(0.01)
    s = t.summary()["steps"]["asr"]
    assert s["count"] == 1
    assert s["mean_ms"] >= 9


def test_timed_records_even_when_block_raises():
    t = Tracker()
    with pytest.raises(RuntimeError):
        with t.timed("llm"):
            raise RuntimeError("boom")
    assert t.summary()["steps"]["llm"]["count"] == 1


def test_decorator_records_and_preserves_return_value():
    t = Tracker()

    @t.timed_fn("tts")
    def speak(x):
        return x * 2

    assert speak(4) == 8
    assert t.summary()["steps"]["tts"]["count"] == 1


def test_summary_stats():
    t = Tracker()
    for seconds in (0.1, 0.2, 0.3, 0.4):
        t.record("llm", seconds)
    s = t.summary()["steps"]["llm"]
    assert s["count"] == 4
    assert s["mean_ms"] == 250.0
    assert s["p50_ms"] == 200.0
    assert s["max_ms"] == 400.0


def test_cost_is_none_without_prices(monkeypatch):
    monkeypatch.delenv("PRICE_IN_PER_M_TOKENS_USD", raising=False)
    monkeypatch.delenv("PRICE_OUT_PER_M_TOKENS_USD", raising=False)
    t = Tracker()
    t.record_tokens(1000, 500)
    assert t.estimated_cost_usd() is None
    assert "n/a" in t.to_markdown()


def test_cost_with_prices(monkeypatch):
    monkeypatch.setenv("PRICE_IN_PER_M_TOKENS_USD", "0.10")
    monkeypatch.setenv("PRICE_OUT_PER_M_TOKENS_USD", "0.50")
    t = Tracker()
    t.record_tokens(1_000_000, 2_000_000)
    assert t.estimated_cost_usd() == pytest.approx(0.10 + 1.00)


def test_bad_price_value_is_ignored(monkeypatch):
    monkeypatch.setenv("PRICE_IN_PER_M_TOKENS_USD", "free")
    monkeypatch.setenv("PRICE_OUT_PER_M_TOKENS_USD", "0.5")
    assert Tracker().estimated_cost_usd() is None


def test_record_tokens_handles_none():
    t = Tracker()
    t.record_tokens(None, None)
    assert t.summary()["tokens"] == {"llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0}


def test_reset_clears_everything():
    t = Tracker()
    t.record("x", 1.0)
    t.record_tokens(5, 5)
    t.reset()
    assert t.summary()["steps"] == {}
    assert t.summary()["tokens"]["llm_calls"] == 0


def test_markdown_table_has_a_row_per_step():
    t = Tracker()
    t.record("asr", 0.5)
    t.record("llm", 1.0)
    md = t.to_markdown()
    assert "| asr |" in md and "| llm |" in md


def test_module_level_tracker_and_timed_alias():
    import timing
    timing.tracker.reset()
    with timing.timed("alias-step"):
        pass
    assert timing.tracker.summary()["steps"]["alias-step"]["count"] == 1
    timing.tracker.reset()

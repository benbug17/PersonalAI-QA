"""RateLimiter and its environment settings. A fake clock makes the sliding-window arithmetic exact."""
import pytest

from agent_router import RateLimiter, limiter_from_env

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_requests_up_to_the_limit_are_allowed_then_refused(clock):
    limiter = RateLimiter(limit=3, window=60, clock=clock)
    assert [limiter.retry_after("u") for _ in range(3)] == [None, None, None]
    assert limiter.retry_after("u") == 60


def test_retry_after_counts_down_as_time_passes(clock):
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.retry_after("u")
    clock.now = 20
    assert limiter.retry_after("u") == 40
    clock.now = 59.5
    assert limiter.retry_after("u") == 1          # always at least one whole second, never 0


def test_the_window_slides_so_old_requests_stop_counting(clock):
    limiter = RateLimiter(limit=2, window=10, clock=clock)
    limiter.retry_after("u")                       # t=0
    clock.now = 4
    limiter.retry_after("u")                       # t=4
    clock.now = 9.9
    assert limiter.retry_after("u") == 1           # the t=0 request has not yet expired
    clock.now = 10
    assert limiter.retry_after("u") is None        # the t=0 request has expired (exactly one window old)
    clock.now = 10.1
    assert limiter.retry_after("u") == 4           # window now holds t=4 and t=10: next opening at t=14


def test_refused_requests_are_not_recorded_so_hammering_does_not_extend_the_block(clock):
    limiter = RateLimiter(limit=2, window=10, clock=clock)
    limiter.retry_after("u")
    clock.now = 1
    limiter.retry_after("u")
    for t in (2, 3, 4, 5, 6, 7, 8, 9):
        clock.now = t
        assert limiter.retry_after("u") is not None
    clock.now = 10
    assert limiter.retry_after("u") is None        # allowed exactly when the first accepted request aged out


def test_each_key_has_its_own_budget(clock):
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    assert limiter.retry_after("alice") is None
    assert limiter.retry_after("alice") == 60
    assert limiter.retry_after("bob") is None


@pytest.mark.parametrize("limit", [0, -5])
def test_a_limit_of_zero_or_less_disables_limiting(clock, limit):
    limiter = RateLimiter(limit=limit, window=60, clock=clock)
    assert all(limiter.retry_after("u") is None for _ in range(500))


def test_reset_forgets_everything(clock):
    limiter = RateLimiter(limit=1, window=60, clock=clock)
    limiter.retry_after("u")
    limiter.reset()
    assert limiter.retry_after("u") is None


def test_memory_is_bounded_idle_keys_are_dropped_once_there_are_too_many(clock):
    limiter = RateLimiter(limit=5, window=60, clock=clock)
    for i in range(RateLimiter.MAX_KEYS + 1):
        limiter.retry_after(f"user-{i}")
    assert len(limiter._hits) == RateLimiter.MAX_KEYS + 1     # all still recent, so none can be dropped yet
    clock.now = 120                                           # everything above is now idle
    limiter.retry_after("newcomer")
    assert list(limiter._hits) == ["newcomer"]


def test_keys_that_are_still_active_survive_the_cleanup(clock):
    limiter = RateLimiter(limit=5, window=60, clock=clock)
    for i in range(RateLimiter.MAX_KEYS):
        limiter.retry_after(f"idle-{i}")
    clock.now = 100
    limiter.retry_after("active")                             # recent
    clock.now = 130
    limiter.retry_after("trigger")                            # 1002 keys > MAX_KEYS: the idle ones go, 'active' stays
    assert set(limiter._hits) == {"active", "trigger"}


# ---- settings from the environment -------------------------------------------------------------------------------
def test_defaults_are_ten_requests_per_minute(monkeypatch):
    monkeypatch.delenv("AGENT_RATE_LIMIT", raising=False)
    monkeypatch.delenv("AGENT_RATE_WINDOW_SECONDS", raising=False)
    limiter = limiter_from_env()
    assert (limiter.limit, limiter.window) == (10, 60)


def test_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT", "3")
    monkeypatch.setenv("AGENT_RATE_WINDOW_SECONDS", "30")
    limiter = limiter_from_env()
    assert (limiter.limit, limiter.window) == (3, 30)


def test_unparseable_settings_fall_back_to_the_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT", "lots")
    monkeypatch.setenv("AGENT_RATE_WINDOW_SECONDS", "a minute")
    limiter = limiter_from_env()
    assert (limiter.limit, limiter.window) == (10, 60)


def test_a_negative_limit_disables_and_a_non_positive_window_falls_back(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT", "-1")
    monkeypatch.setenv("AGENT_RATE_WINDOW_SECONDS", "0")
    limiter = limiter_from_env()
    assert (limiter.limit, limiter.window) == (0, 60)

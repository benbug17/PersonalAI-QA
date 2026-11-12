"""The router on its own (fake auth, no database), to pin down rate-limit behaviour the real app cannot isolate."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

import agent  # noqa: E402
from agent_router import RateLimiter, make_router  # noqa: E402

pytestmark = pytest.mark.api


class World:
    """A tiny app around make_router with a settable current user and a counting graph factory."""
    def __init__(self, limiter=None):
        self.user = "alice"
        self.graphs_built = 0

        def auth():
            if self.user is None:
                raise HTTPException(status_code=401, detail="no")
            return self.user

        def graph_factory():
            self.graphs_built += 1
            return object()
        self.router = make_router(auth, graph_factory, limiter=limiter)
        app = FastAPI()
        app.include_router(self.router)
        self.client = TestClient(app)

    def ask(self):
        return self.client.post("/agent", json={"query": "hi"})


@pytest.fixture(autouse=True)
def instant_agent(monkeypatch):
    monkeypatch.setattr(agent, "run_agent", lambda graph, question, **kw: agent.AgentResult(answer="ok"))


def test_invalid_requests_do_not_use_up_the_budget():
    world = World(RateLimiter(limit=2, window=60))
    for _ in range(5):
        assert world.client.post("/agent", json={"query": ""}).status_code == 422
    assert [world.ask().status_code for _ in range(3)] == [200, 200, 429]


def test_unauthenticated_requests_do_not_use_up_anyones_budget():
    world = World(RateLimiter(limit=1, window=60))
    world.user = None
    for _ in range(5):
        assert world.ask().status_code == 401
    world.user = "alice"
    assert world.ask().status_code == 200


def test_the_limiter_is_keyed_on_the_authenticated_user():
    world = World(RateLimiter(limit=1, window=60))
    assert world.ask().status_code == 200 and world.ask().status_code == 429
    world.user = "bob"
    assert world.ask().status_code == 200
    world.user = 42                                  # the real app's auth returns an integer user id
    assert world.ask().status_code == 200 and world.ask().status_code == 429


def test_a_disabled_limiter_never_refuses():
    world = World(RateLimiter(limit=0, window=60))
    assert all(world.ask().status_code == 200 for _ in range(30))


def test_a_router_without_an_explicit_limiter_reads_its_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_RATE_LIMIT", "1")
    monkeypatch.setenv("AGENT_RATE_WINDOW_SECONDS", "45")
    world = World()
    assert world.router.limiter.limit == 1 and world.router.limiter.window == 45
    assert world.ask().status_code == 200
    r = world.ask()
    assert r.status_code == 429 and 1 <= int(r.headers["Retry-After"]) <= 45


def test_the_refusal_comes_before_any_agent_work():
    world = World(RateLimiter(limit=1, window=60))
    world.ask()
    world.ask()
    world.ask()
    assert world.graphs_built == 1


def test_the_429_is_logged_with_the_user(caplog):
    world = World(RateLimiter(limit=1, window=60))
    world.ask()
    with caplog.at_level("WARNING", logger="personalai"):
        world.ask()
    assert "rate limit hit by user alice" in caplog.text

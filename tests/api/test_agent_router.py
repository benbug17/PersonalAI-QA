import sys

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

import agent  # noqa: E402
from timing import tracker  # noqa: E402

from tests.agent_fakes import ScriptedChatModel, final, tool_call  # noqa: E402

pytestmark = pytest.mark.api


@pytest.fixture(autouse=True)
def clean_tracker():
    tracker.reset()
    yield
    tracker.reset()


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    """The app's limiter is process-wide; start every test from an empty one."""
    import api
    api.agent_routes.limiter.reset()
    yield
    api.agent_routes.limiter.reset()


def login(client, name):
    client.post("/auth/register", json={"username": name, "password": "secret123"})
    token = client.post("/auth/login", json={"username": name, "password": "secret123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def serve_graph(monkeypatch, *responses):
    """Make the API's agent run on a scripted model with a small in-memory notes store."""
    import uuid

    from rag import HashingEmbedder, NoteStore

    store = NoteStore(HashingEmbedder(), collection=f"api-{uuid.uuid4().hex}")
    store.add_text("n.md", "The upload limit is ten megabytes.")
    graph = agent.build_agent(ScriptedChatModel(responses=list(responses)), store)
    monkeypatch.setattr(agent, "get_default_graph", lambda: graph)


def test_agent_answers_and_reports_the_tool_trace(client, auth, monkeypatch):
    serve_graph(monkeypatch, tool_call("calculator", {"expression": "2 + 2"}), final("Four."))
    r = client.post("/agent", json={"query": "2 plus 2?"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Four." and body["tools_used"] == ["calculator"]
    assert body["llm_calls"] == 2 and body["truncated"] is False and body["latency_ms"] >= 0


def test_agent_requires_a_token(client):
    assert client.post("/agent", json={"query": "hi"}).status_code == 401


def test_agent_without_a_key_is_503(client, auth, monkeypatch):
    def no_key():
        raise agent.MissingApiKey("GROQ_API_KEY is not set")
    monkeypatch.setattr(agent, "get_default_graph", no_key)
    r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_agent_that_cannot_start_is_502_and_is_logged(client, auth, monkeypatch, caplog):
    def boom():
        raise RuntimeError("chroma exploded")
    monkeypatch.setattr(agent, "get_default_graph", boom)
    with caplog.at_level("ERROR", logger="personalai"):
        r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 502 and r.json()["detail"] == "Agent could not start"
    assert "chroma exploded" in caplog.text


def test_agent_run_failure_is_502_and_is_logged(client, auth, monkeypatch, caplog):
    class Broken:
        def invoke(self, *a, **k):
            raise RuntimeError("provider down")
    monkeypatch.setattr(agent, "get_default_graph", lambda: Broken())
    with caplog.at_level("ERROR", logger="personalai"):
        r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 502 and r.json()["detail"] == "Agent call failed" and "provider down" in caplog.text


def test_agent_dependencies_not_installed_is_503_and_rest_of_api_unaffected(client, auth, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent", None)   # makes `import agent` raise ImportError
    r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 503 and "not installed" in r.json()["detail"]
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x" * 2001}, {}])
def test_agent_input_validation_is_422(client, auth, payload):
    assert client.post("/agent", json=payload, headers=auth).status_code == 422


def test_agent_accepts_the_maximum_length_query(client, auth, monkeypatch):
    serve_graph(monkeypatch, final("ok"))
    assert client.post("/agent", json={"query": "x" * 2000}, headers=auth).status_code == 200


def test_metrics_reports_pipeline_steps_after_traffic_to_the_token_holder(client, auth, stub_llm, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "s3cret-token")
    client.post("/chat", json={"query": "what is gravity"}, headers=auth)
    body = client.get("/metrics", headers={"X-Metrics-Token": "s3cret-token"}).json()
    assert set(body) == {"steps", "tokens", "estimated_cost_usd"}
    assert body["steps"]["llm"]["count"] == 1


# ---- /metrics is off unless an operator turns it on, and a login does not open it ------------------------------
def test_metrics_is_disabled_by_default_even_for_a_logged_in_user(client, auth, monkeypatch):
    monkeypatch.delenv("METRICS_TOKEN", raising=False)
    assert client.get("/metrics", headers=auth).status_code == 404
    assert client.get("/metrics").status_code == 404


def test_an_empty_metrics_token_does_not_turn_the_endpoint_on(client, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "")
    assert client.get("/metrics", headers={"X-Metrics-Token": ""}).status_code == 404


def test_metrics_rejects_a_missing_or_wrong_token_and_a_login_is_not_enough(client, auth, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "s3cret-token")
    assert client.get("/metrics").status_code == 403
    assert client.get("/metrics", headers={"X-Metrics-Token": "wrong"}).status_code == 403
    assert client.get("/metrics", headers={"X-Metrics-Token": ""}).status_code == 403
    assert client.get("/metrics", headers=auth).status_code == 403          # a valid user login is not the metrics token
    assert client.get("/metrics", headers={"X-Metrics-Token": "s3cret-token"}).status_code == 200


def test_the_metrics_token_is_compared_in_constant_time(client, monkeypatch):
    import agent_router
    seen = []
    real = agent_router.secrets.compare_digest
    monkeypatch.setattr(agent_router.secrets, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    monkeypatch.setenv("METRICS_TOKEN", "s3cret-token")
    assert client.get("/metrics", headers={"X-Metrics-Token": "s3cret-token"}).status_code == 200
    assert seen == [(b"s3cret-token", b"s3cret-token")]


# ---- the real app limits /agent per user ----------------------------------------------------------------------
def test_the_app_limits_each_user_and_tells_them_when_to_retry(client, auth, monkeypatch):
    import api
    monkeypatch.setattr(api.agent_routes.limiter, "limit", 2)
    serve_graph(monkeypatch, final("ok"))
    assert [client.post("/agent", json={"query": "hi"}, headers=auth).status_code for _ in range(2)] == [200, 200]
    r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 429 and r.json()["detail"] == "Too many agent requests, slow down"
    assert 1 <= int(r.headers["Retry-After"]) <= api.agent_routes.limiter.window


def test_one_users_limit_does_not_affect_another_user(client, auth, monkeypatch):
    import api
    monkeypatch.setattr(api.agent_routes.limiter, "limit", 1)
    serve_graph(monkeypatch, final("ok"))
    bob = login(client, "bob")
    assert client.post("/agent", json={"query": "hi"}, headers=auth).status_code == 200
    assert client.post("/agent", json={"query": "hi"}, headers=auth).status_code == 429       # alice is blocked
    assert client.post("/agent", json={"query": "hi"}, headers=bob).status_code == 200        # bob is not


def test_a_refused_request_never_reaches_the_agent_so_it_costs_no_tokens(client, auth, monkeypatch):
    import api
    monkeypatch.setattr(api.agent_routes.limiter, "limit", 1)
    started = []
    serve_graph(monkeypatch, final("ok"))
    real_factory = agent.get_default_graph
    monkeypatch.setattr(agent, "get_default_graph", lambda: started.append(1) or real_factory())
    client.post("/agent", json={"query": "hi"}, headers=auth)
    client.post("/agent", json={"query": "hi"}, headers=auth)
    client.post("/agent", json={"query": "hi"}, headers=auth)
    assert started == [1]


def test_an_agent_result_carrying_an_error_is_502_and_is_logged(client, auth, monkeypatch, caplog):
    serve_graph(monkeypatch, final("unused"))
    monkeypatch.setattr(agent, "run_agent", lambda graph, question, **kw: agent.AgentResult(answer="Sorry", error="malformed tool call"))
    with caplog.at_level("ERROR", logger="personalai"):
        r = client.post("/agent", json={"query": "hi"}, headers=auth)
    assert r.status_code == 502 and r.json()["detail"] == "Agent call failed"
    assert "malformed tool call" in caplog.text

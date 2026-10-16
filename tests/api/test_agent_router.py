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
    assert client.get("/metrics").status_code == 401


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


def test_metrics_reports_pipeline_steps_after_traffic(client, auth, stub_llm):
    client.post("/chat", json={"query": "what is gravity"}, headers=auth)
    body = client.get("/metrics", headers=auth).json()
    assert set(body) == {"steps", "tokens", "estimated_cost_usd"}
    assert body["steps"]["llm"]["count"] == 1

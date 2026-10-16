import pytest

pytestmark = pytest.mark.api


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_register_login_flow(client):
    r = client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    assert r.status_code == 201 and isinstance(r.json()["user_id"], int)
    r = client.post("/auth/login", json={"username": "alice", "password": "secret123"})
    assert r.status_code == 200 and len(r.json()["token"]) >= 32


def test_chat_returns_llm_answer_and_persists_history(client, auth, stub_llm):
    r = client.post("/chat", json={"query": "  what is gravity  "}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == stub_llm.result
    assert body["audio_url"] is None            # speak defaults to False
    assert stub_llm.queries == ["what is gravity"]  # whitespace trimmed

    items = client.get("/history", headers=auth).json()["items"]
    assert len(items) == 1 and items[0]["query"] == "what is gravity"


def test_chat_with_speak_returns_playable_audio(client, auth, stub_llm, fake_gtts):
    body = client.post("/chat", json={"query": "hi", "speak": True}, headers=auth).json()
    assert body["audio_url"].startswith("/audio/")
    audio = client.get(body["audio_url"], headers=auth)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/mpeg"
    assert audio.content.startswith(b"ID3")


def test_voice_pipeline_response_shape(client, auth, stub_stt, stub_llm, fake_gtts):
    r = client.post("/voice", files={"file": ("q.wav", b"RIFFfake", "audio/wav")}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "what is gravity"
    assert body["response"] == stub_llm.result
    assert body["audio_url"].startswith("/audio/")
    assert stub_stt.seen == [b"RIFFfake"]


def test_history_limit_and_order(client, auth, stub_llm):
    for q in ["one", "two", "three"]:
        client.post("/chat", json={"query": q}, headers=auth)
    items = client.get("/history?limit=2", headers=auth).json()["items"]
    assert [i["query"] for i in items] == ["three", "two"]


def test_history_is_isolated_between_users(client, auth, stub_llm):
    client.post("/chat", json={"query": "alice secret question"}, headers=auth)
    client.post("/auth/register", json={"username": "bobby", "password": "secret123"})
    tok = client.post("/auth/login", json={"username": "bobby", "password": "secret123"}).json()["token"]
    items = client.get("/history", headers={"Authorization": f"Bearer {tok}"}).json()["items"]
    assert items == []


# ---- contract -------------------------------------------------------------------
def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_openapi_exposes_expected_routes_and_bearer_scheme(client):
    spec = client.get("/openapi.json").json()
    assert {"/health", "/auth/register", "/auth/login", "/chat", "/voice", "/history", "/audio/{audio_hash}"} <= set(spec["paths"])
    schemes = spec["components"]["securitySchemes"]
    assert any(s.get("type") == "http" and s.get("scheme") == "bearer" for s in schemes.values()), \
        "Swagger needs a bearer scheme so the Authorize button appears"
    assert "/" not in spec["paths"]


def test_unauthorized_responses_carry_www_authenticate(client):
    r = client.get("/history")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_bearer_scheme_is_case_insensitive(client, auth):
    token = auth["Authorization"].split()[1]
    assert client.get("/history", headers={"Authorization": f"bearer {token}"}).status_code == 200


def test_each_login_gets_a_distinct_working_token(client):
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    body = {"username": "alice", "password": "secret123"}
    t1 = client.post("/auth/login", json=body).json()["token"]
    t2 = client.post("/auth/login", json=body).json()["token"]
    assert t1 != t2
    for t in (t1, t2):
        assert client.get("/history", headers={"Authorization": f"Bearer {t}"}).status_code == 200


def test_login_response_shape(client):
    uid = client.post("/auth/register", json={"username": "alice", "password": "secret123"}).json()["user_id"]
    body = client.post("/auth/login", json={"username": "alice", "password": "secret123"}).json()
    assert set(body) == {"token", "user_id"} and body["user_id"] == uid


def test_chat_response_shape_and_history_item_shape(client, auth, stub_llm):
    body = client.post("/chat", json={"query": "hi"}, headers=auth).json()
    assert set(body) == {"query", "response", "audio_url"}
    item = client.get("/history", headers=auth).json()["items"][0]
    assert set(item) == {"id", "query", "response", "timestamp"}


def test_voice_response_shape(client, auth, stub_stt, stub_llm, fake_gtts):
    body = client.post("/voice", files={"file": ("q.wav", b"x", "audio/wav")}, headers=auth).json()
    assert set(body) == {"transcript", "response", "audio_url"}


def test_chat_query_length_boundary(client, auth, stub_llm):
    assert client.post("/chat", json={"query": "a" * 2000}, headers=auth).status_code == 200
    assert client.post("/chat", json={"query": "a" * 2001}, headers=auth).status_code == 422


def test_unicode_and_emoji_queries_roundtrip(client, auth, stub_llm):
    q = "¿Qué es la fotosíntesis? 🌱 光合作用"
    assert client.post("/chat", json={"query": q}, headers=auth).json()["query"] == q
    assert client.get("/history", headers=auth).json()["items"][0]["query"] == q


def test_served_audio_bytes_match_what_was_synthesised(client, auth, stub_llm, fake_gtts):
    body = client.post("/chat", json={"query": "hi", "speak": True}, headers=auth).json()
    audio = client.get(body["audio_url"], headers=auth).content
    assert audio == b"ID3-fake-mp3:" + stub_llm.result.encode()


def test_markdown_answer_is_spoken_without_symbols_but_returned_intact(client, auth, stub_llm, fake_gtts):
    stub_llm.result = "**Gravity** pulls things down.\n- really"
    body = client.post("/chat", json={"query": "hi", "speak": True}, headers=auth).json()
    assert body["response"] == "**Gravity** pulls things down.\n- really"
    assert fake_gtts.calls[0][0] == "Gravity pulls things down.\nreally"


def test_pipeline_order_is_stt_then_llm_then_save_then_tts(client, auth, monkeypatch):
    import asr
    import db
    import groq_client
    import tts
    order = []
    monkeypatch.setattr(asr, "transcribe_audio", lambda b, **k: order.append("stt") or "q")
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: order.append("llm") or "a")
    real_save = db.save_conversation
    monkeypatch.setattr(db, "save_conversation", lambda *a: order.append("db") or real_save(*a))
    monkeypatch.setattr(tts, "text_to_speech", lambda t: order.append("tts") or None)
    client.post("/voice", files={"file": ("q.wav", b"x", "audio/wav")}, headers=auth)
    assert order == ["stt", "llm", "db", "tts"]

import pytest
import db

pytestmark = pytest.mark.negative


# ---- auth / validation -------------------------------------------------
def test_duplicate_registration_conflict(client):
    body = {"username": "alice", "password": "secret123"}
    assert client.post("/auth/register", json=body).status_code == 201
    assert client.post("/auth/register", json=body).status_code == 409


@pytest.mark.parametrize("payload", [
    {"username": "al", "password": "secret123"},          # username too short
    {"username": "alice", "password": "123"},             # password too short
    {"username": "   ", "password": "secret123"},         # blank username
    {"username": "alice", "password": "x" * 73},          # bcrypt 72-byte limit
    {"username": "alice", "password": "é" * 40},          # 80 bytes, 40 chars
    {"username": "alice"},                                # missing field
    {},
])
def test_register_rejects_invalid_payload(client, payload):
    assert client.post("/auth/register", json=payload).status_code == 422


def test_login_wrong_password_and_unknown_user_look_identical(client):
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    a = client.post("/auth/login", json={"username": "alice", "password": "WRONG-pass"})
    b = client.post("/auth/login", json={"username": "ghost", "password": "secret123"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()      # no user-enumeration leak


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "},
])
@pytest.mark.parametrize("method,path", [
    ("post", "/chat"), ("get", "/history"), ("post", "/voice"), ("get", "/audio/" + "a" * 64),
])
def test_protected_routes_require_valid_token(client, method, path, headers):
    kwargs = {"headers": headers}
    if path == "/chat":
        kwargs["json"] = {"query": "hi"}
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_sql_injection_in_login_is_inert(client):
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    r = client.post("/auth/login", json={"username": "alice' OR '1'='1", "password": "x"})
    assert r.status_code == 401


def test_sql_injection_payload_stored_verbatim_not_executed(client, auth, stub_llm):
    evil = "'); DROP TABLE history;--"
    assert client.post("/chat", json={"query": evil}, headers=auth).status_code == 200
    assert client.get("/history", headers=auth).json()["items"][0]["query"] == evil


# ---- /chat input -------------------------------------------------------
@pytest.mark.parametrize("query", ["", "   ", "\n\n", "x" * 2001])
def test_chat_rejects_empty_or_oversized_query(client, auth, stub_llm, query):
    assert client.post("/chat", json={"query": query}, headers=auth).status_code == 422
    assert stub_llm.queries == []     # LLM never called


def test_history_limit_bounds(client, auth):
    assert client.get("/history?limit=0", headers=auth).status_code == 422
    assert client.get("/history?limit=101", headers=auth).status_code == 422


# ---- upstream service failures ----------------------------------------
def test_llm_failure_returns_502_and_saves_nothing(client, auth, stub_llm):
    stub_llm.result = None
    assert client.post("/chat", json={"query": "hi"}, headers=auth).status_code == 502
    assert client.get("/history", headers=auth).json()["items"] == []


def test_missing_api_key_returns_503(client, auth, stub_llm, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    r = client.post("/chat", json={"query": "hi"}, headers=auth)
    assert r.status_code == 503 and stub_llm.queries == []


def test_stt_failure_returns_422_and_skips_llm(client, auth, stub_stt, stub_llm):
    stub_stt.result = None
    r = client.post("/voice", files={"file": ("q.wav", b"noise", "audio/wav")}, headers=auth)
    assert r.status_code == 422 and stub_llm.queries == []


def test_tts_failure_still_returns_text_answer(client, auth, stub_stt, stub_llm, fake_gtts):
    fake_gtts.fail = True
    r = client.post("/voice", files={"file": ("q.wav", b"audio", "audio/wav")}, headers=auth)
    assert r.status_code == 200
    assert r.json()["audio_url"] is None and r.json()["response"] == stub_llm.result


def test_db_write_failure_does_not_lose_the_answer(client, auth, stub_llm, monkeypatch):
    monkeypatch.setattr(db, "save_conversation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    r = client.post("/chat", json={"query": "hi"}, headers=auth)
    assert r.status_code == 200 and r.json()["response"] == stub_llm.result


# ---- /voice upload -----------------------------------------------------
def test_voice_empty_upload_400(client, auth, stub_stt):
    r = client.post("/voice", files={"file": ("q.wav", b"", "audio/wav")}, headers=auth)
    assert r.status_code == 400 and stub_stt.seen == []


def test_voice_missing_file_422(client, auth):
    assert client.post("/voice", headers=auth).status_code == 422


def test_voice_oversized_upload_413(client, auth, stub_stt, monkeypatch):
    import api
    monkeypatch.setattr(api, "MAX_AUDIO_BYTES", 1024)
    r = client.post("/voice", files={"file": ("q.wav", b"x" * 2048, "audio/wav")}, headers=auth)
    assert r.status_code == 413 and stub_stt.seen == []


# ---- /audio ------------------------------------------------------------
@pytest.mark.parametrize("bad", ["..%2F..%2Fetc%2Fpasswd", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65])
def test_audio_rejects_malformed_ids_incl_path_traversal(client, auth, bad):
    assert client.get(f"/audio/{bad}", headers=auth).status_code in (400, 404)


def test_audio_unknown_hash_404(client, auth):
    assert client.get("/audio/" + "a" * 64, headers=auth).status_code == 404


def test_audio_only_serves_hash_named_files(client, auth):
    """A real file with a non-hash name must be refused (guard, not just 'file missing')."""
    import os
    import tts
    os.makedirs(tts.TTS_CACHE_DIR, exist_ok=True)
    with open(os.path.join(tts.TTS_CACHE_DIR, "private-notes.mp3"), "wb") as f:
        f.write(b"should never be served")
    r = client.get("/audio/private-notes", headers=auth)
    assert r.status_code == 400

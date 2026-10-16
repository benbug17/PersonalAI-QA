"""Real api + db + tts + asr-orchestration code; only the 3 network engines are stubbed."""
import os
import pytest
import asr
import db
import groq_client
import tts

pytestmark = pytest.mark.integration


def _voice(client, auth, data=b"RIFFfake-audio"):
    return client.post("/voice", files={"file": ("q.wav", data, "audio/wav")}, headers=auth)


def test_full_voice_roundtrip_touches_every_layer(client, auth, monkeypatch, fake_gtts):
    # Whisper fails -> real asr.transcribe_audio must fall back to speech_recognition
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: None)
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: "explain photosynthesis")
    seen = {}
    def llm(q, temperature=0.7):
        seen["q"] = q
        return "Plants turn sunlight into sugar."
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", llm)

    r = _voice(client, auth)
    assert r.status_code == 200
    body = r.json()

    assert seen["q"] == "explain photosynthesis"                 # STT -> LLM hand-off
    uid = client.post("/auth/login", json={"username": "alice", "password": "secret123"}).json()["user_id"]
    hist = db.get_user_history(uid)
    assert hist[0]["query"] == "explain photosynthesis"           # LLM -> DB
    assert hist[0]["response"] == "Plants turn sunlight into sugar."
    mp3 = os.path.join(tts.TTS_CACHE_DIR, body["audio_url"].split("/")[-1] + ".mp3")
    assert os.path.isfile(mp3)                                    # LLM -> TTS cache on disk
    assert client.get(body["audio_url"], headers=auth).status_code == 200


def test_identical_answers_are_synthesised_once(client, auth, stub_stt, stub_llm, fake_gtts):
    _voice(client, auth)
    _voice(client, auth)
    assert len(fake_gtts.calls) == 1, "TTS cache should serve the repeat"
    assert len(client.get("/history", headers=auth).json()["items"]) == 2


def test_session_survives_across_requests_and_logout_by_restart(client, auth, stub_llm):
    assert client.post("/chat", json={"query": "a"}, headers=auth).status_code == 200
    import api
    api.SESSIONS.clear()  # simulates server restart
    assert client.post("/chat", json={"query": "b"}, headers=auth).status_code == 401


def test_history_persists_across_server_restart(client, auth, stub_llm):
    client.post("/chat", json={"query": "remember me"}, headers=auth)
    import api
    api.SESSIONS.clear()                                   # restart wipes tokens, not the database
    token = client.post("/auth/login", json={"username": "alice", "password": "secret123"}).json()["token"]
    items = client.get("/history", headers={"Authorization": f"Bearer {token}"}).json()["items"]
    assert [i["query"] for i in items] == ["remember me"]


def test_two_users_share_tts_cache_but_not_history(client, auth, stub_llm, fake_gtts):
    client.post("/chat", json={"query": "same q", "speak": True}, headers=auth)
    client.post("/auth/register", json={"username": "bobby", "password": "secret123"})
    tok = client.post("/auth/login", json={"username": "bobby", "password": "secret123"}).json()["token"]
    bob = {"Authorization": f"Bearer {tok}"}
    client.post("/chat", json={"query": "same q", "speak": True}, headers=bob)
    assert len(fake_gtts.calls) == 1                       # identical answer synthesised once
    assert len(client.get("/history", headers=auth).json()["items"]) == 1
    assert len(client.get("/history", headers=bob).json()["items"]) == 1


def test_failed_turn_leaves_no_trace_then_next_turn_works(client, auth, stub_llm):
    stub_llm.result = None
    assert client.post("/chat", json={"query": "first"}, headers=auth).status_code == 502
    stub_llm.result = "recovered"
    assert client.post("/chat", json={"query": "second"}, headers=auth).json()["response"] == "recovered"
    assert [i["query"] for i in client.get("/history", headers=auth).json()["items"]] == ["second"]

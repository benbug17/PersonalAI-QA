"""Shared fixtures: isolated DB + TTS cache per test, stub external services."""
import pytest
from fastapi.testclient import TestClient

import api
import asr
import db
import groq_client
import tts


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch, request):
    """Every test gets its own SQLite file and TTS cache dir (and a fake API key,
    except `live` tests, which must use the real one)."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(tts, "TTS_CACHE_DIR", str(tmp_path / "tts_cache"))
    if "live" not in request.keywords:
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
    db.init_db()
    api.SESSIONS.clear()
    yield


class FakeGTTS:
    """Stand-in for gtts.gTTS: writes deterministic bytes, records calls."""
    calls = []
    fail = False

    def __init__(self, text, lang="en", slow=False):
        if FakeGTTS.fail:
            raise RuntimeError("gTTS unavailable")
        self.text, self.lang, self.slow = text, lang, slow
        FakeGTTS.calls.append((text, lang, slow))

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"ID3-fake-mp3:" + self.text.encode("utf-8"))


@pytest.fixture
def fake_gtts(monkeypatch):
    FakeGTTS.calls, FakeGTTS.fail = [], False
    monkeypatch.setattr(tts, "gTTS", FakeGTTS)
    return FakeGTTS


@pytest.fixture
def stub_stt(monkeypatch):
    """Control STT output. Set stub_stt.result / stub_stt.seen."""
    class S:
        result = "what is gravity"
        seen = []
    def fake(audio_bytes, prefer_whisper=True):
        S.seen.append(audio_bytes)
        return S.result
    S.seen = []
    monkeypatch.setattr(asr, "transcribe_audio", fake)
    return S


@pytest.fixture
def stub_llm(monkeypatch):
    """Control LLM output. Set stub_llm.result / inspect stub_llm.queries."""
    class L:
        result = "Gravity is the force that pulls objects toward each other."
        queries = []
    def fake(query, temperature=0.7):
        L.queries.append(query)
        return L.result
    L.queries = []
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", fake)
    return L


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def auth(client):
    """Registered + logged-in user -> Authorization headers."""
    client.post("/auth/register", json={"username": "alice", "password": "secret123"})
    token = client.post("/auth/login", json={"username": "alice", "password": "secret123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}

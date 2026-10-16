"""The existing modules feed the latency/cost tracker without changing their behaviour."""
import pytest

import groq_client
from timing import tracker

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_tracker():
    tracker.reset()
    yield
    tracker.reset()


def fake_groq(monkeypatch, usage):
    class FakeGroq:
        def __init__(self, api_key):
            class _Completions:
                def create(self, **kwargs):
                    msg = type("M", (), {"content": "reply"})
                    fields = {"choices": [type("C", (), {"message": msg})]}
                    if usage is not None:
                        fields["usage"] = usage
                    return type("R", (), fields)
            self.chat = type("Chat", (), {"completions": _Completions()})

    monkeypatch.setattr(groq_client, "Groq", FakeGroq)


def test_completion_records_token_usage(monkeypatch):
    fake_groq(monkeypatch, type("U", (), {"prompt_tokens": 120, "completion_tokens": 30}))
    assert groq_client.get_completion("hi") == "reply"
    tokens = tracker.summary()["tokens"]
    assert tokens == {"llm_calls": 1, "prompt_tokens": 120, "completion_tokens": 30}


def test_completion_without_usage_info_still_works(monkeypatch):
    fake_groq(monkeypatch, None)
    assert groq_client.get_completion("hi") == "reply"
    assert tracker.summary()["tokens"]["llm_calls"] == 0


def test_completion_with_partial_usage_info_records_zeros(monkeypatch):
    fake_groq(monkeypatch, type("U", (), {}))
    assert groq_client.get_completion("hi") == "reply"
    assert tracker.summary()["tokens"] == {"llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0}


def test_chat_endpoint_times_the_llm_and_tts_steps(client, auth, stub_llm, fake_gtts):
    client.post("/chat", json={"query": "what is gravity", "speak": True}, headers=auth)
    steps = tracker.summary()["steps"]
    assert steps["llm"]["count"] == 1 and steps["tts"]["count"] == 1


def test_voice_endpoint_times_asr_llm_and_tts(client, auth, stub_stt, stub_llm, fake_gtts):
    r = client.post("/voice", files={"file": ("q.wav", b"RIFFfakewav", "audio/wav")}, headers=auth)
    assert r.status_code == 200
    steps = tracker.summary()["steps"]
    assert {"asr", "llm", "tts"} <= set(steps) and all(steps[s]["count"] == 1 for s in ("asr", "llm", "tts"))

import pytest
import groq_client

pytestmark = pytest.mark.unit


class FakeGroq:
    """Records the request the app sends and returns a canned completion."""
    last_kwargs = None
    reply = "canned reply"
    raise_exc = None

    def __init__(self, api_key):
        self.api_key = api_key
        outer = self

        class _Completions:
            def create(self, **kwargs):
                if FakeGroq.raise_exc:
                    raise FakeGroq.raise_exc
                FakeGroq.last_kwargs = kwargs
                msg = type("M", (), {"content": FakeGroq.reply})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        outer.chat = type("Chat", (), {"completions": _Completions()})


@pytest.fixture(autouse=True)
def fake_groq(monkeypatch):
    FakeGroq.last_kwargs, FakeGroq.reply, FakeGroq.raise_exc = None, "canned reply", None
    monkeypatch.setattr(groq_client, "Groq", FakeGroq)


def test_no_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert groq_client.get_groq_client() is None
    assert groq_client.get_completion("hi") is None


def test_completion_builds_messages_with_system_prompt():
    out = groq_client.get_completion("hi", system_prompt="be brief")
    assert out == "canned reply"
    msgs = FakeGroq.last_kwargs["messages"]
    assert msgs == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_completion_without_system_prompt_sends_only_user_message():
    groq_client.get_completion("hi")
    assert FakeGroq.last_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_learning_assistant_request_shape_regression():
    """Locks model, token budget and the voice-oriented system prompt."""
    groq_client.get_learning_assistant_response("explain atoms")
    kw = FakeGroq.last_kwargs
    assert kw["model"] == groq_client.DEFAULT_MODEL == "openai/gpt-oss-20b"
    assert kw["max_tokens"] == 1024
    assert kw["extra_body"] == {"reasoning_effort": "low"}
    assert kw["temperature"] == 0.7
    system = kw["messages"][0]["content"]
    assert "2-3 paragraphs" in system and "voice" in system
    assert kw["messages"][-1] == {"role": "user", "content": "explain atoms"}


def test_temperature_override_for_evals():
    groq_client.get_learning_assistant_response("x", temperature=0.0)
    assert FakeGroq.last_kwargs["temperature"] == 0.0


def test_api_exception_returns_none():
    FakeGroq.raise_exc = RuntimeError("rate limited")
    assert groq_client.get_learning_assistant_response("hi") is None


def test_model_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    groq_client.get_learning_assistant_response("x")
    assert FakeGroq.last_kwargs["model"] == "some/other-model"
    assert "extra_body" not in FakeGroq.last_kwargs   # reasoning knob only for gpt-oss


def test_decommissioned_model_is_not_the_default():
    """Regression: Groq shut down llama-3.1-8b-instant on 2026-08-16."""
    assert "llama-3.1-8b-instant" not in groq_client.get_model()


def test_client_is_built_with_the_env_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc123")
    assert groq_client.get_groq_client().api_key == "abc123"


def test_system_prompt_encodes_the_voice_and_safety_contract():
    p = groq_client.SYSTEM_PROMPT
    for must in ("2-3 paragraphs", "markdown", "Never reveal", "false premise", "distress"):
        assert must in p, must


def test_empty_model_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "")
    assert groq_client.get_model() == groq_client.DEFAULT_MODEL


def test_explicit_model_argument_wins_over_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "from/env")
    groq_client.get_completion("hi", model="explicit/model")
    assert FakeGroq.last_kwargs["model"] == "explicit/model"


def test_system_prompt_forbids_over_refusal():
    """Regression: gpt-oss answered a benign essay request and a SQL-like string with a bare refusal."""
    p = groq_client.SYSTEM_PROMPT
    assert "do not refuse" in p and "bare refusal" in p
    assert "fragment of SQL" in p and "Never refuse it" in p
    assert "Decline only requests that would cause real harm" in p

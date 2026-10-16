"""Drives the real app.py through Streamlit's AppTest engine.

Only the browser microphone widget and the network engines (STT / LLM / TTS) are stubbed.
"""
import os
import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

import db

pytestmark = pytest.mark.ui
APP = os.path.join(os.path.dirname(__file__), "..", "..", "app.py")


@pytest.fixture
def mic(monkeypatch):
    """Fake `audio_recorder_streamlit` module: set mic.audio = bytes to simulate a recording."""
    state = types.SimpleNamespace(audio=None)
    mod = types.ModuleType("audio_recorder_streamlit")
    mod.audio_recorder = lambda **kwargs: state.audio
    monkeypatch.setitem(sys.modules, "audio_recorder_streamlit", mod)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)   # a real .env must not leak into tests
    return state


def new_app():
    return AppTest.from_file(APP, default_timeout=30)


def login(at, username="alice", password="secret123"):
    at.text_input(key="login_username").input(username)
    at.text_input(key="login_password").input(password)
    at.button(key="login_button").click().run()
    return at


@pytest.fixture
def user():
    db.create_user("alice", "secret123")


def texts(elements):
    return [e.value for e in elements]


# ---------------------------------------------------------------- login / registration
def test_unauthenticated_user_sees_login_and_register_tabs(mic):
    at = new_app().run()
    assert not at.exception
    assert "Voice Learning Assistant" in at.title[0].value
    assert len(at.tabs) == 2 and not at.session_state["authenticated"]


def test_login_with_empty_fields_warns(mic):
    at = new_app().run()
    at.button(key="login_button").click().run()
    assert "Please enter both username and password" in texts(at.warning)


def test_login_with_wrong_password_shows_error(mic, user):
    at = login(new_app().run(), password="wrong-pass")
    assert "Invalid username or password" in texts(at.error)
    assert not at.session_state["authenticated"]


def test_successful_login_shows_main_app(mic, user, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    at = login(new_app().run())
    assert at.session_state["authenticated"] and at.session_state["username"] == "alice"
    assert any("alice" in t.value for t in at.sidebar.title)
    assert any("Record Your Question" in s for s in texts(at.subheader))


def test_register_validation_messages(mic):
    at = new_app().run()
    at.text_input(key="reg_username").input("carol")
    at.text_input(key="reg_password").input("secret123")
    at.text_input(key="reg_password_confirm").input("different1")
    at.button(key="register_button").click().run()
    assert "Passwords do not match" in texts(at.error)

    at.text_input(key="reg_password").input("abc")
    at.text_input(key="reg_password_confirm").input("abc")
    at.button(key="register_button").click().run()
    assert "Password must be at least 6 characters" in texts(at.error)


def test_register_with_missing_fields_warns(mic):
    at = new_app().run()
    at.button(key="register_button").click().run()
    assert "Please fill in all fields" in texts(at.warning)


def test_register_creates_account_and_rejects_duplicates(mic):
    at = new_app().run()
    for k, v in (("reg_username", "carol"), ("reg_password", "secret123"), ("reg_password_confirm", "secret123")):
        at.text_input(key=k).input(v)
    at.button(key="register_button").click().run()
    assert any("Account created" in t for t in texts(at.success))
    assert db.authenticate_user("carol", "secret123")
    at.button(key="register_button").click().run()
    assert "Username already exists" in texts(at.error)


# ---------------------------------------------------------------- main app
def test_missing_api_key_shows_error_and_stops(mic, user, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = login(new_app().run())
    assert any("GROQ_API_KEY not found" in t for t in texts(at.error))
    assert not any("Record Your Question" in s for s in texts(at.subheader))


@pytest.fixture
def signed_in(mic, user, monkeypatch, fake_gtts):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    return login(new_app().run()), mic


def test_voice_turn_transcribes_answers_saves_and_speaks(signed_in, stub_stt, stub_llm):
    at, mic = signed_in
    mic.audio = b"RIFFfake"
    at.run()
    assert not at.exception
    assert any("what is gravity" in t for t in texts(at.success))          # transcript echoed
    assert stub_llm.result in " ".join(texts(at.markdown))                  # answer displayed
    assert stub_llm.queries == ["what is gravity"]
    uid = db.authenticate_user("alice", "secret123")
    assert db.get_user_history(uid)[0]["response"] == stub_llm.result       # persisted
    assert at.session_state["audio_path"].endswith(".mp3")                  # TTS produced
    assert not at.error


def test_stt_failure_is_reported(signed_in, stub_stt, stub_llm):
    at, mic = signed_in
    stub_stt.result, mic.audio = None, b"noise"
    at.run()
    assert "Failed to transcribe audio. Please try again." in texts(at.error)
    assert stub_llm.queries == []


def test_llm_failure_is_reported(signed_in, stub_stt, stub_llm):
    at, mic = signed_in
    stub_llm.result, mic.audio = None, b"x"
    at.run()
    assert any("Failed to get response" in t for t in texts(at.error))
    uid = db.authenticate_user("alice", "secret123")
    assert db.get_user_history(uid) == []


def test_tts_failure_is_reported_but_text_answer_still_shown(signed_in, stub_stt, stub_llm, fake_gtts):
    at, mic = signed_in
    fake_gtts.fail, mic.audio = True, b"x"
    at.run()
    assert "Failed to generate audio response" in texts(at.error)
    assert stub_llm.result in " ".join(texts(at.markdown))


def test_db_save_failure_warns_but_answer_still_shown(signed_in, stub_stt, stub_llm, monkeypatch):
    at, mic = signed_in
    monkeypatch.setattr(db, "save_conversation", lambda *a: (_ for _ in ()).throw(RuntimeError("disk")))
    mic.audio = b"x"
    at.run()
    assert "Could not save conversation to history." in texts(at.warning)
    assert stub_llm.result in " ".join(texts(at.markdown))


def test_history_section_lists_past_conversations(mic, user, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    uid = db.authenticate_user("alice", "secret123")
    db.save_conversation(uid, "what is a sedan", "A sedan is a car.")
    at = login(new_app().run())
    assert any("what is a sedan" in e.label for e in at.expander)
    assert any("Previous question" in i.value for i in at.info)             # last chat restored at login


def test_empty_history_message(signed_in):
    at, _ = signed_in
    assert "No conversation history yet. Start by asking a question!" in texts(at.info)


def test_long_question_title_is_truncated_in_history(mic, user, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    uid = db.authenticate_user("alice", "secret123")
    db.save_conversation(uid, "q" * 200, "a")
    at = login(new_app().run())
    assert any(e.label.endswith("...") and len(e.label) < 90 for e in at.expander)


def test_logout_clears_session(signed_in):
    at, _ = signed_in
    next(b for b in at.sidebar.button if b.label == "Logout").click().run()
    assert not at.session_state["authenticated"] and at.session_state["user_id"] is None
    assert "Login" in [t.label for t in at.tabs]


def test_clear_cache_button_empties_tts_cache(signed_in, stub_stt, stub_llm):
    import tts
    at, mic = signed_in
    mic.audio = b"x"
    at.run()
    assert tts.get_cache_stats()["file_count"] == 1
    mic.audio = None
    next(b for b in at.sidebar.button if b.label == "Clear Cache").click().run()
    assert tts.get_cache_stats()["file_count"] == 0


# ---------------------------------------------------------------- error handling & edge branches
def test_db_failure_when_restoring_last_conversation_warns(mic, user, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(db, "get_user_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    at = login(new_app().run())
    assert "Could not load previous conversation from DB." in texts(at.warning)
    assert "No conversation history yet. Start by asking a question!" in texts(at.info)   # page still renders


def test_history_list_failure_degrades_to_empty_state(signed_in, monkeypatch):
    at, _ = signed_in
    real = db.get_user_history
    def flaky(user_id, limit=50):
        if limit == 10:
            raise RuntimeError("db down")
        return real(user_id, limit)
    monkeypatch.setattr(db, "get_user_history", flaky)
    at.run()
    assert not at.exception and "No conversation history yet. Start by asking a question!" in texts(at.info)


def test_missing_audio_file_is_reported(signed_in, stub_stt, stub_llm, monkeypatch):
    import tts
    at, mic = signed_in
    monkeypatch.setattr(tts, "text_to_speech", lambda text: "/no/such/file.mp3")
    mic.audio = b"x"
    at.run()
    assert "Generated audio file not found." in texts(at.warning)


def test_audio_playback_error_is_reported(signed_in, stub_stt, stub_llm, monkeypatch):
    import streamlit as st
    at, mic = signed_in
    real_audio = st.audio
    def picky(data, *a, **k):
        if isinstance(data, str):
            raise RuntimeError("codec error")
        return real_audio(data, *a, **k)
    monkeypatch.setattr(st, "audio", picky)
    mic.audio = b"x"
    at.run()
    assert "Could not play generated audio." in texts(at.warning)


def test_previous_answer_is_replayed_when_no_new_recording(signed_in, stub_stt, stub_llm):
    at, mic = signed_in
    mic.audio = b"x"
    at.run()
    mic.audio = None
    at.run()                                                  # same session, no new recording
    assert any("Previous question" in i.value for i in at.info)
    assert stub_llm.result in " ".join(texts(at.markdown))
    assert len(stub_llm.queries) == 1                         # replay must not call the LLM again


def test_previous_answer_audio_replay_handles_bytes_and_errors(signed_in, stub_stt, stub_llm, monkeypatch):
    import streamlit as st
    at, mic = signed_in
    mic.audio = b"x"
    at.run()
    mic.audio = None
    at.session_state["audio_path"] = b"raw-mp3-bytes"
    at.run()
    assert not at.exception                                   # bytes branch plays directly
    real_audio = st.audio
    def boom(data, *a, **k):
        raise RuntimeError("no codec")
    monkeypatch.setattr(st, "audio", boom)
    at.session_state["audio_path"] = b"raw-mp3-bytes"
    at.run()
    assert not at.exception                                   # playback errors are swallowed silently
    monkeypatch.setattr(st, "audio", real_audio)


def test_tts_returning_raw_bytes_is_played_directly(signed_in, stub_stt, stub_llm, monkeypatch):
    import tts
    at, mic = signed_in
    monkeypatch.setattr(tts, "text_to_speech", lambda text: b"raw-mp3-bytes")
    mic.audio = b"x"
    at.run()
    assert not at.exception and not at.error and at.session_state["audio_path"] == b"raw-mp3-bytes"


def test_restore_helper_ignores_missing_user(mic):
    import app
    assert app._load_user_last_conversation_into_session(None) is None

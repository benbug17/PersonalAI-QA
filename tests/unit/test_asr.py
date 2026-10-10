import os
import sys
import types

import pytest
import asr

pytestmark = pytest.mark.unit


def test_empty_audio_returns_none_without_calling_engines(monkeypatch):
    def boom(_): raise AssertionError("engine should not be called")
    monkeypatch.setattr(asr, "transcribe_audio_whisper", boom)
    monkeypatch.setattr(asr, "transcribe_audio_sr", boom)
    assert asr.transcribe_audio(b"") is None


def test_whisper_first_and_no_fallback_when_it_works(monkeypatch):
    calls = []
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: calls.append("w") or "hello")
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: calls.append("sr") or "x")
    assert asr.transcribe_audio(b"audio") == "hello"
    assert calls == ["w"]


def test_falls_back_to_speech_recognition(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: None)
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: "fallback text")
    assert asr.transcribe_audio(b"audio") == "fallback text"


def test_prefer_sr_order(monkeypatch):
    calls = []
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: calls.append("w") or "w")
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: calls.append("sr") or "sr")
    assert asr.transcribe_audio(b"a", prefer_whisper=False) == "sr"
    assert calls == ["sr"]


def test_all_engines_fail_returns_none(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: None)
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: None)
    assert asr.transcribe_audio(b"audio") is None


def test_transcribe_from_missing_file_returns_none():
    assert asr.transcribe_from_file("/no/such/file.wav") is None


# ---- real engine code paths, with fake third-party modules ---------------------

@pytest.fixture
def fake_whisper(monkeypatch):
    state = types.SimpleNamespace(paths=[], text=" hello world ", boom=None, model_name=None)

    class Model:
        def transcribe(self, path):
            assert os.path.exists(path), "audio must be on disk while transcribing"
            assert open(path, "rb").read() == b"AUDIO"
            state.paths.append(path)
            if state.boom:
                raise state.boom
            return {"text": state.text}

    mod = types.ModuleType("whisper")
    def load_model(name):
        state.model_name = name
        return Model()
    mod.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", mod)
    return state


def test_whisper_transcribes_strips_text_and_cleans_temp_file(fake_whisper):
    assert asr.transcribe_audio_whisper(b"AUDIO") == "hello world"
    assert fake_whisper.model_name == "base"
    assert not os.path.exists(fake_whisper.paths[0])


def test_whisper_blank_transcript_returns_none(fake_whisper):
    fake_whisper.text = "   "
    assert asr.transcribe_audio_whisper(b"AUDIO") is None


def test_whisper_engine_error_returns_none_and_still_cleans_up(fake_whisper):
    fake_whisper.boom = RuntimeError("decode failed")
    assert asr.transcribe_audio_whisper(b"AUDIO") is None
    assert not os.path.exists(fake_whisper.paths[0])


def test_whisper_not_installed_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)   # makes `import whisper` raise ImportError
    assert asr.transcribe_audio_whisper(b"AUDIO") is None


@pytest.fixture
def fake_sr(monkeypatch):
    state = types.SimpleNamespace(paths=[], text=" good morning ", boom=None)
    mod = types.ModuleType("speech_recognition")

    class AudioFile:
        def __init__(self, path):
            assert os.path.exists(path)
            state.paths.append(path)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Recognizer:
        def record(self, source): return "audio-data"
        def recognize_google(self, data):
            assert data == "audio-data"
            if state.boom:
                raise state.boom
            return state.text

    mod.AudioFile, mod.Recognizer = AudioFile, Recognizer
    monkeypatch.setitem(sys.modules, "speech_recognition", mod)
    return state


def test_speech_recognition_transcribes_and_cleans_temp_file(fake_sr):
    assert asr.transcribe_audio_sr(b"AUDIO") == "good morning"
    assert not os.path.exists(fake_sr.paths[0])


def test_speech_recognition_empty_result_returns_none(fake_sr):
    fake_sr.text = ""
    assert asr.transcribe_audio_sr(b"AUDIO") is None


def test_speech_recognition_error_returns_none(fake_sr):
    fake_sr.boom = RuntimeError("could not understand audio")
    assert asr.transcribe_audio_sr(b"AUDIO") is None
    assert not os.path.exists(fake_sr.paths[0])


def test_speech_recognition_not_installed_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "speech_recognition", None)
    assert asr.transcribe_audio_sr(b"AUDIO") is None


def test_end_to_end_fallback_with_real_orchestration(fake_whisper, fake_sr):
    fake_whisper.text = ""            # whisper hears nothing -> real fallback to SR
    fake_sr.text = "from the fallback"
    assert asr.transcribe_audio(b"AUDIO") == "from the fallback"


def test_transcribe_from_file_reads_bytes(tmp_path, monkeypatch):
    f = tmp_path / "q.wav"
    f.write_bytes(b"RIFFdata")
    seen = []
    monkeypatch.setattr(asr, "transcribe_audio", lambda b: seen.append(b) or "ok")
    assert asr.transcribe_from_file(str(f)) == "ok" and seen == [b"RIFFdata"]


def test_prefer_sr_falls_back_to_whisper_when_sr_hears_nothing(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: None)
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: "whisper rescued it")
    assert asr.transcribe_audio(b"a", prefer_whisper=False) == "whisper rescued it"


def test_prefer_sr_returns_none_when_both_fail(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_audio_sr", lambda b: None)
    monkeypatch.setattr(asr, "transcribe_audio_whisper", lambda b: None)
    assert asr.transcribe_audio(b"a", prefer_whisper=False) is None

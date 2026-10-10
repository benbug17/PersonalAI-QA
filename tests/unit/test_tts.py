import os
import pytest
import tts

pytestmark = pytest.mark.unit


def test_hash_is_stable_sha256():
    assert tts.get_text_hash("hi") == tts.get_text_hash("hi")
    assert len(tts.get_text_hash("hi")) == 64
    assert tts.get_text_hash("hi") != tts.get_text_hash("Hi")


def test_generates_mp3_then_serves_from_cache(fake_gtts):
    p1 = tts.text_to_speech("hello world")
    p2 = tts.text_to_speech("hello world")
    assert p1 == p2 and os.path.exists(p1) and p1.endswith(".mp3")
    assert len(fake_gtts.calls) == 1, "second call must hit the cache"


def test_different_text_different_file(fake_gtts):
    assert tts.text_to_speech("one") != tts.text_to_speech("two")
    assert len(fake_gtts.calls) == 2


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_returns_none_without_calling_gtts(fake_gtts, text):
    assert tts.text_to_speech(text) is None
    assert fake_gtts.calls == []


def test_gtts_failure_returns_none(fake_gtts):
    fake_gtts.fail = True
    assert tts.text_to_speech("hello") is None


def test_bytes_variant(fake_gtts):
    assert tts.text_to_speech_bytes("hello").startswith(b"ID3")


def test_cache_stats_and_clear(fake_gtts):
    assert tts.get_cache_stats() == {"file_count": 0, "total_size_mb": 0.0}
    tts.text_to_speech("a"); tts.text_to_speech("b")
    assert tts.get_cache_stats()["file_count"] == 2
    tts.clear_cache(max_files=5)          # under threshold -> untouched
    assert tts.get_cache_stats()["file_count"] == 2
    tts.clear_cache()
    assert tts.get_cache_stats()["file_count"] == 0


# ---- speech sanitising ------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("**bold** and *italic*", "bold and italic"),
    ("snake_case_name stays intact", "snake_case_name stays intact"),
    ("_emphasis_ here", "emphasis here"),
    ("# Title\nbody", "Title\nbody"),
    ("- one\n- two\n* three", "one\ntwo\nthree"),
    ("see [the docs](https://x.y/z) now", "see the docs now"),
    ("![alt text](img.png)", "alt text"),
    ("use `print()` here", "use print() here"),
    ("before\n```python\nprint(1)\n```\nafter", "before\n \nafter"),
    ("> quoted", "quoted"),
    ("Pet | Care", "Pet , Care"),
    ("Intro\n| Pet | Care |\n|---|---|\n| Cat | Low |", "Intro\nPet, Care.\nCat, Low."),
    ("a\n\n\n\nb", "a\n\nb"),
    ("plain sentence, nothing to strip.", "plain sentence, nothing to strip."),
])
def test_clean_for_speech(raw, expected):
    assert tts.clean_for_speech(raw) == expected


def test_markdown_is_stripped_before_synthesis_and_before_hashing(fake_gtts):
    p1 = tts.text_to_speech("**Hello** world")
    p2 = tts.text_to_speech("Hello world")
    assert fake_gtts.calls == [("Hello world", "en", False)]
    assert p1 == p2, "formatting differences must not defeat the cache"


def test_text_that_is_only_formatting_returns_none(fake_gtts):
    assert tts.text_to_speech("```\ncode only\n```") is None
    assert fake_gtts.calls == []


def test_language_and_slow_flags_are_forwarded(fake_gtts):
    tts.text_to_speech("hola", language="es", slow=True)
    assert fake_gtts.calls == [("hola", "es", True)]


def test_bytes_variant_returns_none_when_generation_fails(fake_gtts):
    fake_gtts.fail = True
    assert tts.text_to_speech_bytes("hello") is None


def test_bytes_variant_survives_read_error(fake_gtts, monkeypatch):
    path = tts.text_to_speech("hello")
    real_open = open
    def bad_open(p, *a, **k):
        if str(p) == path:
            raise OSError("denied")
        return real_open(p, *a, **k)
    monkeypatch.setattr("builtins.open", bad_open)
    assert tts.text_to_speech_bytes("hello") is None


def test_cache_helpers_survive_filesystem_errors(fake_gtts, monkeypatch, capsys):
    tts.text_to_speech("hello")
    monkeypatch.setattr(os, "listdir", lambda p: (_ for _ in ()).throw(OSError("boom")))
    assert tts.get_cache_stats() == {"file_count": 0, "total_size_mb": 0.0}
    tts.clear_cache()                      # must not raise
    assert "Error" in capsys.readouterr().out


def test_clear_cache_and_stats_on_missing_dir():
    tts.clear_cache()                      # dir does not exist: no error
    assert tts.get_cache_stats()["file_count"] == 0


def test_clear_cache_only_touches_mp3(fake_gtts):
    tts.text_to_speech("hello")
    keep = os.path.join(tts.TTS_CACHE_DIR, "notes.txt")
    open(keep, "w").write("keep me")
    tts.clear_cache()
    assert os.path.exists(keep) and tts.get_cache_stats()["file_count"] == 0

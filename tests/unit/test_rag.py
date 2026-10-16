import sys
import types
import uuid

import pytest

pytest.importorskip("chromadb")

import rag  # noqa: E402
from rag import DEFAULT_NOTES_DIR, HashingEmbedder, NoteStore, chunk_text, ingest_directory  # noqa: E402

pytestmark = pytest.mark.unit


def make_store():
    return NoteStore(HashingEmbedder(), collection=f"test-{uuid.uuid4().hex}")


def test_chunk_text_overlap_and_coverage():
    chunks = chunk_text(" ".join(f"w{i}" for i in range(50)), size=20, overlap=5)
    assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w49"
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


def test_chunker_stops_without_a_redundant_trailing_window():
    assert len(chunk_text(" ".join(f"w{i}" for i in range(50)), size=20, overlap=5)) == 3


def test_chunk_text_short_and_empty():
    assert chunk_text("") == []
    assert chunk_text("just a few words") == ["just a few words"]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e(["hello world"])[0], e(["hello world"])[0]
    assert a == b and sum(x * x for x in a) == pytest.approx(1.0)


def test_embedder_handles_text_with_only_stopwords():
    assert sum(HashingEmbedder()(["the a of"])[0]) == 0.0


def test_bundled_notes_directory_exists_and_is_indexable():
    store = make_store()
    counts = ingest_directory(store, DEFAULT_NOTES_DIR)
    assert counts.get("personalai_facts.md", 0) >= 1


def test_ingest_and_retrieve_right_chunk():
    store = make_store()
    ingest_directory(store, DEFAULT_NOTES_DIR)
    hits = store.query("Which Groq model was retired?")
    assert hits and "llama-3.1-8b-instant" in hits[0]["text"] and hits[0]["source"] == "personalai_facts.md"


def test_ingest_directory_only_reads_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text("alpha note")
    (tmp_path / "b.txt").write_text("bravo note")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    assert ingest_directory(make_store(), tmp_path) == {"a.md": 1, "b.txt": 1}


def test_unrelated_question_returns_nothing():
    store = make_store()
    ingest_directory(store, DEFAULT_NOTES_DIR)
    assert store.query("airspeed velocity of an unladen swallow") == []


def test_empty_store_and_blank_query():
    store = make_store()
    assert store.query("anything") == []
    store.add_text("a.md", "some content here")
    assert store.query("   ") == []


def test_adding_blank_text_indexes_nothing():
    assert make_store().add_text("e.md", "   ") == 0


def test_re_adding_a_source_replaces_old_chunks():
    store = make_store()
    store.add_text("a.md", "alpha bravo charlie")
    store.add_text("a.md", "delta echo foxtrot")
    assert store.count() == 1 and store.query("alpha bravo") == [] and store.query("delta echo")


def test_replacing_a_long_note_with_a_short_one_drops_the_extra_chunks():
    store = make_store()
    assert store.add_text("a.md", " ".join(f"alpha{i}" for i in range(300)), size=50, overlap=10) > 3
    store.add_text("a.md", "short replacement note", size=50, overlap=10)
    assert store.count() == 1


def test_k_limits_results():
    store = make_store()
    for i in range(5):
        store.add_text(f"n{i}.md", f"shared keyword zebra number{i}")
    assert len(store.query("zebra", k=2)) == 2


def test_persistent_store_survives_reopen(tmp_path):
    name = f"persist-{uuid.uuid4().hex}"
    first = NoteStore(HashingEmbedder(), path=str(tmp_path), collection=name)
    first.add_text("a.md", "kubernetes deployment rollout strategy")
    del first
    second = NoteStore(HashingEmbedder(), path=str(tmp_path), collection=name)
    assert second.count() == 1 and second.query("kubernetes rollout")


def test_sentence_transformer_embedder_wraps_the_model(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            FakeModel.name = name

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return types.SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    embed = rag.sentence_transformer_embedder("some/model")
    assert FakeModel.name == "some/model" and embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="requirements-agent-live.txt"):
        rag.sentence_transformer_embedder()

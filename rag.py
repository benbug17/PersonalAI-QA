"""Retrieval-augmented generation layer: chunking, embeddings, Chroma store.

The embedding function is injected, so the store works with:
  * sentence_transformer_embedder()  - real semantic embeddings (use this in the app)
  * HashingEmbedder()                - deterministic, no downloads (used in offline tests)

Chroma runs with embedding_function=None, so it never silently downloads its own model.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Callable, Sequence

from timing import timed

DEFAULT_NOTES_DIR = Path(__file__).resolve().parent / "data" / "notes"

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been it its this that these those "
    "what which who whom how why when where do does did i my me you your we our with as at by "
    "from about into than then so if not no can could should would will".split()
)


class HashingEmbedder:
    """Bag-of-words hashing embedder. Not semantic, but deterministic and dependency-free."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            if token in _STOPWORDS:
                continue
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]  # crude plural stripping: "passwords" matches "password"
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def sentence_transformer_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbedFn:
    """Real semantic embeddings. Needs requirements-agent-live.txt (downloads the model once)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed: pip install -r requirements-agent-live.txt") from exc

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(list(texts), normalize_embeddings=True).tolist()

    return embed


def onnx_embedder() -> EmbedFn:
    """Real semantic embeddings via Chroma's bundled MiniLM model on ONNX. No PyTorch needed.

    The model file (about 80 MB) is downloaded the first time it is used.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    model = DefaultEmbeddingFunction()

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in model(list(texts))]

    return embed


EMBEDDERS = ("st", "onnx", "hashing")


def get_embedder(name: str) -> EmbedFn:
    """Look an embedder up by name: st (sentence-transformers), onnx (Chroma's bundled model) or hashing (offline stand-in)."""
    if name == "st":
        return sentence_transformer_embedder()
    if name == "onnx":
        return onnx_embedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {name!r}; choose one of {', '.join(EMBEDDERS)}")


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_paragraphs(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Chunk by paragraph so each fact stays in its own chunk; split only paragraphs longer than `size` words.

    A one-line markdown heading is kept with the paragraph that follows it, so a chunk keeps its context.
    """
    chunks: list[str] = []
    heading = ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if paragraph.startswith("#") and "\n" not in paragraph:
            heading = paragraph
            continue
        body = f"{heading}\n\n{paragraph}" if heading else paragraph
        heading = ""
        if len(body.split()) > size:
            chunks.extend(chunk_text(body, size=size, overlap=overlap))
        else:
            chunks.append(body)
    if heading:  # a heading with nothing after it
        chunks.append(heading)
    return chunks


class NoteStore:
    """A small Chroma-backed store of text notes."""

    def __init__(self, embed_fn: EmbedFn, path: str | None = None, collection: str = "personalai_notes") -> None:
        import chromadb

        self._embed = embed_fn
        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        return self._col.count()

    def add_text(self, source: str, text: str, size: int = 120, overlap: int = 20) -> int:
        """Index one document, one chunk per paragraph. Re-adding the same source replaces its old chunks."""
        chunks = chunk_paragraphs(text, size=size, overlap=overlap)
        self._col.delete(where={"source": source})
        if not chunks:
            return 0
        with timed("rag_embed_index"):
            vectors = self._embed(chunks)
        self._col.upsert(
            ids=[f"{source}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def query(self, question: str, k: int = 3, min_score: float = 0.2) -> list[dict]:
        """Return up to k chunks with cosine similarity >= min_score, best first."""
        total = self._col.count()
        if total == 0 or not question.strip():
            return []
        with timed("rag_retrieval"):
            vector = self._embed([question])
            raw = self._col.query(query_embeddings=vector, n_results=min(k, total))
        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            score = 1.0 - float(dist)
            if score >= min_score:
                results.append({"text": doc, "source": meta["source"], "score": round(score, 3)})
        return results


def ingest_directory(store: NoteStore, directory: str | Path) -> dict[str, int]:
    """Index every .md and .txt file in a directory. Returns {filename: chunk_count}."""
    counts: dict[str, int] = {}
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            counts[path.name] = store.add_text(path.name, path.read_text(encoding="utf-8"))
    return counts

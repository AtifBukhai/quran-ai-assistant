"""Multilingual embeddings.

Two backends:

* ``hash``  — deterministic, dependency-free, offline. A hashing-trick bag-of-character-ngrams
  projected to ``embedding_dim`` and L2-normalized. Not semantically strong, but fully
  reproducible and lets the whole pipeline + tests run with zero downloads. Cross-lingual
  retrieval still works for exact/keyword paths and for the sample corpus.
* ``sentence-transformers`` — a real multilingual encoder (e.g. multilingual-e5). Install with
  ``pip install -e ".[ml]"`` and set ``QURAN_EMBEDDING_BACKEND=sentence-transformers``.

Both expose the same interface: ``encode(texts) -> list[list[float]]`` and ``dim``.
Embeddings are computed at ingest time from ``Verse.embedding_source()`` and at query time from
the raw user question, so equivalent AR/EN/UR questions land near the same verse.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    text = f" {text.strip()} "
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


class HashEmbedder:
    """Deterministic hashing-trick embedder. No network, no model download."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for gram in _char_ngrams(text.lower(), 3):
            h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Real multilingual embeddings. Imported lazily so the core stays dependency-light."""

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        # get_sentence_embedding_dimension() was renamed to get_embedding_dimension();
        # fall back for older/newer versions so we don't depend on either name.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim() or dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        # show_progress_bar so large corpus ingests (6236 verses on CPU) are visibly moving,
        # not apparently hung. batch_size keeps memory bounded for long multi-translation inputs.
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True,
        )
        return [v.tolist() for v in vecs]


def build_embedder(backend: str, model_name: str, dim: int) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, dim)
    if backend == "hash":
        return HashEmbedder(dim)
    raise ValueError(f"Unknown embedding backend: {backend!r}")

"""Runtime configuration, loaded from environment (12-factor).

Defaults are chosen so the whole system runs offline with no API key and no server:
``echo`` LLM + ``hash`` embeddings + in-memory Qdrant. Flip the env vars in ``.env`` to move to
cloud generation, real multilingual embeddings, and a Qdrant server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars take precedence."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    llm_model: str
    embedding_backend: str
    embedding_model: str
    embedding_dim: int
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    top_k: int
    top_n: int
    min_score: float
    min_evidence: int
    secondary_score: float


@lru_cache
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        llm_backend=_get("QURAN_LLM_BACKEND", "echo"),
        llm_model=_get("QURAN_LLM_MODEL", "claude-3-5-sonnet-latest"),
        embedding_backend=_get("QURAN_EMBEDDING_BACKEND", "hash"),
        embedding_model=_get("QURAN_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
        embedding_dim=_get_int("QURAN_EMBEDDING_DIM", 384),
        qdrant_url=_get("QURAN_QDRANT_URL", ""),
        qdrant_api_key=_get("QURAN_QDRANT_API_KEY", ""),
        qdrant_collection=_get("QURAN_QDRANT_COLLECTION", "quran_verses"),
        top_k=_get_int("QURAN_TOP_K", 8),
        top_n=_get_int("QURAN_TOP_N", 5),
        min_score=_get_float("QURAN_MIN_SCORE", 0.30),
        min_evidence=_get_int("QURAN_MIN_EVIDENCE", 1),
        secondary_score=_get_float("QURAN_SECONDARY_SCORE", 0.25),
    )

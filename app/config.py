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


def _get_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
    dense_weight: float
    # Conversation memory
    conversation_backend: str
    redis_url: str
    session_ttl_seconds: int
    max_history_turns: int
    history_token_budget: int
    history_in_prompt: bool


@lru_cache
def get_settings() -> Settings:
    _load_dotenv()
    embedding_backend = _get("QURAN_EMBEDDING_BACKEND", "hash")
    # Dense-arm weight in the SEMANTIC blend. Policy (strict, lexical-anchored grounding):
    # vector similarity ASSISTS candidate recall/ranking but must never, on its own, clear the
    # confidence gate — every answer stays anchored to a lexical/concept hit plus mandatory
    # citations. So the dense weight is capped below min_score (0.30): even a perfect cosine of
    # 1.0 contributes at most 0.25 < 0.30, and a verse with no shared/expanded term is refused.
    # This holds for BOTH the offline hash embedder and a real neural embedder — the neural
    # model improves *which* verses surface, not whether pure-vector matches can bypass grounding.
    # To deliberately loosen this (allow strong vector-only matches to answer), set
    # QURAN_DENSE_WEIGHT explicitly (e.g. 0.55); an explicit value always wins.
    default_dense = 0.25
    return Settings(
        llm_backend=_get("QURAN_LLM_BACKEND", "echo"),
        llm_model=_get("QURAN_LLM_MODEL", "claude-3-5-sonnet-latest"),
        embedding_backend=embedding_backend,
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
        dense_weight=_get_float("QURAN_DENSE_WEIGHT", default_dense),
        # Conversation memory: default to in-memory with strict grounding (history_in_prompt off).
        conversation_backend=_get("QURAN_CONVERSATION_BACKEND", "memory"),
        redis_url=_get("QURAN_REDIS_URL", ""),
        session_ttl_seconds=_get_int("QURAN_SESSION_TTL_SECONDS", 3600),
        max_history_turns=_get_int("QURAN_MAX_HISTORY_TURNS", 10),
        history_token_budget=_get_int("QURAN_HISTORY_TOKEN_BUDGET", 2000),
        history_in_prompt=_get_bool("QURAN_HISTORY_IN_PROMPT", False),
    )

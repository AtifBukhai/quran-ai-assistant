"""Shared pytest fixtures.

Builds an Orchestrator backed by an in-memory Qdrant seeded with the sample corpus, using the
offline echo LLM and hash embeddings — so the full grounding suite runs with no network and no
API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.ingest.run import to_payload
from app.models import Verse
from app.orchestrator import Orchestrator

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"


def _offline_settings() -> Settings:
    return Settings(
        llm_backend="echo",
        llm_model="none",
        embedding_backend="hash",
        embedding_model="none",
        embedding_dim=256,
        qdrant_url="",
        qdrant_api_key="",
        qdrant_collection="quran_verses_test",
        top_k=8,
        top_n=5,
        min_score=0.30,
        min_evidence=1,
        secondary_score=0.25,
        dense_weight=0.25,  # strict lexical-first for hash embedder
        conversation_backend="memory",
        redis_url="",
        session_ttl_seconds=3600,
        max_history_turns=10,
        history_token_budget=2000,
        history_in_prompt=False,  # strict: history never enters the generation prompt by default
    )


@pytest.fixture(scope="session")
def sample_verses() -> list[Verse]:
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return [Verse.model_validate(o) for o in raw]


@pytest.fixture()
def orch(sample_verses) -> Orchestrator:
    settings = _offline_settings()
    o = Orchestrator(settings)
    o.store.recreate()
    vectors = o.embedder.encode([v.embedding_source() for v in sample_verses])
    o.store.upsert(
        [(v.verse_id, vec, to_payload(v)) for v, vec in zip(sample_verses, vectors)]
    )
    return o

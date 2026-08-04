"""FastAPI application exposing the grounded Quran assistant.

Endpoints:
* POST /v1/ask                 — grounded Q&A (the full pipeline)
* GET  /v1/verse/{surah}/{ayah}— exact verse lookup (AR/EN/UR + metadata)
* GET  /v1/surah/{surah}       — surah metadata + ayah list
* POST /v1/search              — raw retrieval (no generation), ranked verses
* GET  /v1/health              — liveness + corpus count
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import AskRequest, AskResponse, TOTAL_AYAHS, parse_verse_id
from .orchestrator import Orchestrator
from .retrieval import detect_language, route

app = FastAPI(
    title="Quran AI Assistant",
    version="1.0.0",
    description="Strictly grounded Quran RAG. Answers only from retrieved Quran verses.",
)


@lru_cache
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return get_orchestrator().ask(req)


@app.get("/v1/verse/{surah}/{ayah}")
def get_verse(surah: int, ayah: int):
    ref = parse_verse_id(f"{surah}:{ayah}")
    if not ref:
        raise HTTPException(400, "Invalid verse reference")
    payload = get_orchestrator().store.get_by_verse_id(f"{surah}:{ayah}")
    if not payload:
        raise HTTPException(404, "Verse not found in corpus")
    return payload


@app.get("/v1/surah/{surah}")
def get_surah(surah: int):
    from qdrant_client.http import models as qm  # noqa: PLC0415

    store = get_orchestrator().store
    verses = store.scroll_filter(
        qm.Filter(must=[qm.FieldCondition(key="surah_number", match=qm.MatchValue(value=surah))]),
        limit=300,
    )
    if not verses:
        raise HTTPException(404, "Surah not found in corpus")
    verses.sort(key=lambda p: p["ayah_number"])
    names = verses[0].get("surah_name", {})
    return {
        "surah_number": surah,
        "surah_name": names,
        "revelation_type": verses[0].get("revelation_type"),
        "ayah_count": len(verses),
        "ayat": verses,
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    lang: str | None = None
    limit: int = 5
    filters: dict | None = None


@app.post("/v1/search")
def search(req: SearchRequest):
    """Raw retrieval without generation — returns ranked verses only (debug/explore)."""
    orch = get_orchestrator()
    language = req.lang or detect_language(req.query)
    routed = route(req.query, language, req.filters)
    results = orch._retrieve(routed)  # noqa: SLF001 - intentional internal reuse
    return {
        "language": language,
        "intent": routed.intent.value,
        "results": [
            {"score": round(s, 4), **p} for s, p in results[: req.limit]
        ],
    }


@app.get("/v1/health")
def health():
    orch = get_orchestrator()
    count = orch.store.count()
    return {
        "status": "ok",
        "corpus_verses": count,
        "full_corpus": count == TOTAL_AYAHS,
        "llm_backend": orch.settings.llm_backend,
        "embedding_backend": orch.settings.embedding_backend,
    }

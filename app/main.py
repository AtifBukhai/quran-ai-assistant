"""FastAPI application exposing the grounded Quran assistant.

Endpoints:
* POST /v1/ask                 — grounded Q&A (the full pipeline)
* POST /v1/ask/stream          — same pipeline, streamed token-by-token over SSE (grounding-safe)
* GET  /v1/verse/{surah}/{ayah}— exact verse lookup (AR/EN/UR + metadata)
* GET  /v1/surah/{surah}       — surah metadata + ayah list
* POST /v1/search              — raw retrieval (no generation), ranked verses
* GET  /v1/health              — liveness + corpus count
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .models import AskRequest, AskResponse, TOTAL_AYAHS, parse_verse_id
from .orchestrator import Orchestrator
from .retrieval import detect_language, route
from .ui import INDEX_HTML

app = FastAPI(
    title="Quran AI Assistant",
    version="1.0.0",
    description="Strictly grounded Quran RAG. Answers only from retrieved Quran verses.",
)


@lru_cache
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """Minimal search UI: a text box + button that calls POST /v1/ask."""
    return INDEX_HTML


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return get_orchestrator().ask(req)


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame. UTF-8 JSON payload (Arabic preserved verbatim)."""
    import json  # noqa: PLC0415

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Grounding-safe streaming of a grounded answer over Server-Sent Events.

    IMPORTANT — this endpoint does NOT stream a live model. It runs the *entire* validated
    pipeline first (``orchestrator.ask``: retrieve → confidence gate → generate → deterministic
    citation/provenance validation), and only then replays the ALREADY-VALIDATED answer text
    token-by-token for a typing effect. Nothing unvalidated is ever emitted: the evidence,
    citations, and status delivered in the terminal ``done`` frame are exactly what ``POST
    /v1/ask`` would return. If the turn is a refusal, the refusal text streams and ``done`` carries
    ``is_refusal: true`` with empty evidence. So "history assists continuity, never grounding"
    holds unchanged — streaming is a presentation layer over the validated result.

    Event contract (matches the browser client in ``index.html``):
      event: meta   data: {status, language, mode, trace_id, session_id}
      event: token  data: {"t": "<word plus trailing space>"}      (repeated)
      event: done   data: {<full AskResponse>, "is_refusal": bool}
    """
    import re  # noqa: PLC0415
    import time  # noqa: PLC0415

    # Run the full, validated pipeline up front — before a single token is streamed.
    resp = get_orchestrator().ask(req)

    def gen():
        yield _sse(
            "meta",
            {
                "status": resp.status,
                "language": resp.language,
                "mode": resp.mode,
                "trace_id": resp.trace_id,
                "session_id": resp.session_id,
            },
        )
        # Replay the validated answer, chunked on whitespace so spacing is preserved.
        for token in re.findall(r"\S+\s*", resp.answer):
            yield _sse("token", {"t": token})
            time.sleep(0.012)  # light pacing for the typing effect (runs in a threadpool)
        done = resp.model_dump(mode="json")  # mode="json" -> enums/floats become JSON-safe
        done["is_refusal"] = resp.is_refusal
        yield _sse("done", done)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/session")
def create_session():
    """Mint a new conversation session id. Pass it back in POST /v1/ask as ``session_id``."""
    conv = get_orchestrator().history.create()
    return {"session_id": conv.session_id, "created_at": conv.created_at}


@app.get("/v1/session/{session_id}")
def get_session(session_id: str):
    """Return the conversation history and the running list of previously cited verses."""
    conv = get_orchestrator().history.get(session_id)
    if not conv:
        raise HTTPException(404, "Session not found or expired")
    return {
        "session_id": conv.session_id,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "turn_count": len(conv.turns),
        "previous_citations": conv.previous_citations(),
        "turns": [t.to_dict() for t in conv.turns],
    }


@app.delete("/v1/session/{session_id}")
def delete_session(session_id: str):
    """Clear a conversation session (forget its history)."""
    deleted = get_orchestrator().history.delete(session_id)
    if not deleted:
        raise HTTPException(404, "Session not found or expired")
    return {"session_id": session_id, "deleted": True}


@app.get("/v1/verse/{surah}/{ayah}")
def get_verse(surah: int, ayah: int):
    ref = parse_verse_id(f"{surah}:{ayah}")
    if not ref:
        raise HTTPException(400, "Invalid verse reference")
    payload = get_orchestrator().store.get_by_verse_id(f"{surah}:{ayah}")
    if not payload:
        raise HTTPException(404, "Verse not found in corpus")
    # Return payload with all translation fields (base + extras)
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


@app.get("/v1/translations")
def list_translations():
    """Return available translation metadata (for UI dropdown labels).

    Prefers the translations.json manifest (with real translator names) written by the
    corpus builder; falls back to detecting fields on a sample verse.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    # Try the manifest first (has real translator names like "Maududi", "Yusuf Ali")
    for candidate in (Path("data/translations.json"), Path("data/tanzil/translations.json")):
        if candidate.exists():
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
                if manifest.get("english") or manifest.get("urdu"):
                    return manifest
            except (json.JSONDecodeError, OSError):
                pass

    # Fallback: detect which translation fields exist by sampling a verse
    orch = get_orchestrator()
    points, _ = orch.store.client.scroll(
        collection_name=orch.store.collection,
        limit=1,
        with_payload=True,
    )
    if not points:
        return {"english": [], "urdu": []}

    payload = points[0].payload or {}
    en_translations = [{"key": "1", "label": "Sahih International (primary)"}]
    ur_translations = [{"key": "1", "label": "Jalandhry (primary)"}]

    for key in sorted(payload.keys()):
        if key.startswith("translation_en_") and not key.endswith("_lower"):
            idx = key.replace("translation_en_", "")
            en_translations.append({"key": idx, "label": f"English Translation #{idx}"})

    for key in sorted(payload.keys()):
        if key.startswith("translation_ur_") and not key.endswith("_normalized"):
            idx = key.replace("translation_ur_", "")
            ur_translations.append({"key": idx, "label": f"Urdu Translation #{idx}"})

    return {"english": en_translations, "urdu": ur_translations}


class WordSearchRequest(BaseModel):
    term: str = Field(min_length=1)
    lang: str = "ar"  # ar | en | ur


@app.post("/v1/wordsearch")
def word_search(req: WordSearchRequest):
    """Find all occurrences of a term across the full Quran (concordance).

    Returns total count and every (surah:ayah, snippet) where the term appears.
    Uses diacritic-insensitive matching for Arabic, case-insensitive for English/Urdu.
    """
    from .models import normalize_arabic, normalize_urdu  # noqa: PLC0415

    orch = get_orchestrator()
    field_map = {"ar": "text_ar", "en": "translation_en", "ur": "translation_ur"}
    norm_field_map = {
        "ar": "text_ar_normalized",
        "en": "translation_en_lower",
        "ur": "translation_ur_normalized",
    }

    if req.lang not in field_map:
        raise HTTPException(400, f"Unsupported language: {req.lang}")

    # Normalize the search term the same way retrieval does
    if req.lang == "ar":
        needle = normalize_arabic(req.term)
    elif req.lang == "ur":
        needle = normalize_urdu(req.term)
    else:
        needle = req.term.lower()

    raw_field = field_map[req.lang]
    norm_field = norm_field_map[req.lang]

    # Scroll the entire corpus
    points, _ = orch.store.client.scroll(
        collection_name=orch.store.collection,
        limit=6300,
        with_payload=True,
    )

    occurrences = []
    for p in points:
        pl = p.payload or {}
        # Check base field
        norm_text = str(pl.get(norm_field, ""))
        count = norm_text.count(needle) if needle in norm_text else 0

        # Check extra translation fields
        if req.lang == "en":
            for key in pl:
                if key.startswith("translation_en_") and key.endswith("_lower") and key != "translation_en_lower":
                    extra_norm = str(pl.get(key, ""))
                    if needle in extra_norm:
                        count += extra_norm.count(needle)
        elif req.lang == "ur":
            for key in pl:
                if key.startswith("translation_ur_") and key.endswith("_normalized") and key != "translation_ur_normalized":
                    extra_norm = str(pl.get(key, ""))
                    if needle in extra_norm:
                        count += extra_norm.count(needle)

        if count > 0:
            raw_text = str(pl.get(raw_field, ""))
            occ = {
                "verse_id": pl.get("verse_id"),
                "surah_number": pl.get("surah_number"),
                "ayah_number": pl.get("ayah_number"),
                "surah_name": pl.get("surah_name", {}),
                "text": raw_text,
                "count_in_verse": count,
            }
            # Include extra translation raw text so the UI can display the selected one
            for key, val in pl.items():
                if (key.startswith("translation_en_") and not key.endswith("_lower")) or (
                    key.startswith("translation_ur_") and not key.endswith("_normalized")
                ):
                    occ[key] = val
            occurrences.append(occ)

    return {
        "term": req.term,
        "language": req.lang,
        "total_occurrences": sum(o["count_in_verse"] for o in occurrences),
        "verses_matched": len(occurrences),
        "occurrences": occurrences,
    }

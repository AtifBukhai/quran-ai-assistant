"""Layer 4 — deterministic post-generation validation.

Code, not the model, has the final say. Given the model's raw answer and the exact set of
retrieved verses, this module enforces:

1. Citation presence      — >= 1 [S:A] citation, else refuse (no citations = no answer).
2. Verse existence        — every cited id exists in the retrieved set / corpus.
3. Provenance             — every cited id was actually retrieved (no smuggled verses).
4. Text fidelity          — the final Arabic/EN/UR is re-rendered from the canonical corpus,
                            so the model can never invent or alter verse text.

The validator returns either a validated answer + the evidence list to render, or a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grounding import REFUSAL_INSUFFICIENT, extract_citations
from .models import EvidenceVerse, RevelationType, SurahName


@dataclass
class ValidationResult:
    ok: bool
    answer: str
    citations: list[str]
    evidence: list[EvidenceVerse]
    reason: str = ""


def _to_evidence(payload: dict, score: float) -> EvidenceVerse:
    names = payload.get("surah_name", {})
    return EvidenceVerse(
        verse_id=payload["verse_id"],
        surah_number=payload["surah_number"],
        ayah_number=payload["ayah_number"],
        surah_name=SurahName(
            ar=names.get("ar", ""), en=names.get("en", ""), ur=names.get("ur", "")
        ),
        text_ar=payload["text_ar"],  # canonical corpus text — authoritative
        translation_en=payload.get("translation_en"),
        translation_ur=payload.get("translation_ur"),
        revelation_type=RevelationType(payload["revelation_type"]),
        juz=payload["juz"],
        score=round(score, 4),
    )


def validate(
    answer: str,
    retrieved: list[tuple[float, dict]],
) -> ValidationResult:
    """Validate the model's answer against the retrieved verses."""
    retrieved_by_id = {p["verse_id"]: (s, p) for s, p in retrieved}

    citations = extract_citations(answer)

    # Rule 1: at least one citation.
    if not citations:
        return ValidationResult(
            False, REFUSAL_INSUFFICIENT, [], [], reason="no_citations"
        )

    # Rules 2 & 3: every citation must exist AND have been retrieved (provenance).
    valid_ids = [c for c in citations if c in retrieved_by_id]
    smuggled = [c for c in citations if c not in retrieved_by_id]
    if smuggled:
        # The model cited a verse that was never in context -> reject entirely.
        return ValidationResult(
            False,
            REFUSAL_INSUFFICIENT,
            [],
            [],
            reason=f"smuggled_citations:{','.join(smuggled)}",
        )
    if not valid_ids:
        return ValidationResult(
            False, REFUSAL_INSUFFICIENT, [], [], reason="no_valid_citations"
        )

    # Rule 4: build evidence from canonical corpus payloads, in citation order.
    evidence = [_to_evidence(*retrieved_by_id[c]) for c in valid_ids]

    return ValidationResult(True, answer.strip(), valid_ids, evidence)

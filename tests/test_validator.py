"""Validator (Layer 4) tests — the deterministic grounding guardrails.

These are the most important tests in the project: they prove no ungrounded answer can escape.
"""

from __future__ import annotations

from app.grounding import REFUSAL_INSUFFICIENT, extract_citations
from app.validator import validate


def _retrieved(*verse_ids):
    """Fake retrieved payloads for the given ids (only fields the validator needs)."""
    out = []
    for i, vid in enumerate(verse_ids):
        s, a = vid.split(":")
        out.append(
            (
                1.0 - i * 0.1,
                {
                    "verse_id": vid,
                    "surah_number": int(s),
                    "ayah_number": int(a),
                    "surah_name": {"ar": "x", "en": "X", "ur": "x"},
                    "text_ar": f"arabic-{vid}",
                    "translation_en": f"english-{vid}",
                    "translation_ur": f"urdu-{vid}",
                    "revelation_type": "Makki",
                    "juz": 1,
                },
            )
        )
    return out


def test_extract_citations_dedup_order():
    assert extract_citations("a [2:255] b [112:1] c [2:255]") == ["2:255", "112:1"]


def test_no_citation_is_refused():
    res = validate("Allah is one and merciful.", _retrieved("112:1"))
    assert res.ok is False
    assert res.answer == REFUSAL_INSUFFICIENT
    assert res.reason == "no_citations"


def test_smuggled_citation_is_rejected():
    # Model cites 4:34 which was never retrieved -> reject entirely (provenance).
    res = validate("Ungrounded claim [4:34].", _retrieved("112:1", "2:255"))
    assert res.ok is False
    assert res.reason.startswith("smuggled_citations")


def test_valid_grounded_answer_passes():
    res = validate("He is Allah, One [112:1].", _retrieved("112:1", "55:3"))
    assert res.ok is True
    assert res.citations == ["112:1"]
    assert len(res.evidence) == 1
    ev = res.evidence[0]
    # Text fidelity: evidence text comes from the canonical retrieved payload.
    assert ev.text_ar == "arabic-112:1"
    assert ev.verse_id == "112:1"


def test_multiple_valid_citations_all_rendered():
    res = validate("Creation of man [15:26] and [96:2].", _retrieved("15:26", "96:2", "112:1"))
    assert res.ok is True
    assert res.citations == ["15:26", "96:2"]
    assert [e.verse_id for e in res.evidence] == ["15:26", "96:2"]


def test_partial_smuggle_still_rejected():
    # One valid + one smuggled -> the whole answer is rejected (strict).
    res = validate("Mix [112:1] and [4:34].", _retrieved("112:1"))
    assert res.ok is False

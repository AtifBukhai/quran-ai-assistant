"""Grounding contract: prompts, refusal strings, context builder, confidence gate, citations.

This module encodes the non-negotiable grounding rules as code and constants so they are shared
by the orchestrator, the LLM adapters, and the validator — one source of truth for the guardrails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- Exact refusal / scope strings (spec-mandated wording) --------------------
REFUSAL_INSUFFICIENT = (
    "I cannot answer this question from the Quran alone because the available "
    "Quranic verses do not provide sufficient information."
)
REFUSAL_LOW_CONFIDENCE = (
    "I could not find sufficient Quranic evidence to answer this question."
)
OUT_OF_SCOPE = (
    "This question cannot be answered from the Quran alone. This assistant is "
    "intentionally limited to the Quran as its only knowledge source."
)

# --- Strict system prompt (Layer 2) -------------------------------------------
SYSTEM_PROMPT = (
    "You are a Quran AI Assistant.\n"
    "Use ONLY the retrieved Quran verses provided in the context.\n"
    "Never use prior knowledge.\n"
    "Never use memory.\n"
    "Never use internet knowledge.\n"
    "Never use external religious knowledge.\n"
    "Never use Hadith, Tafsir, or scholarly opinion.\n"
    "Never invent verses.\n"
    "Never invent meanings.\n"
    "Never fabricate interpretations.\n"
    "If the retrieved context does not contain enough information, reply with exactly:\n"
    f'"{REFUSAL_INSUFFICIENT}"\n'
    "Always cite every verse you use with its reference in square brackets, e.g. [2:255].\n"
    "An answer with no citation is not permitted."
)

_CITATION_RE = re.compile(r"\[(\d{1,3}):(\d{1,3})\]")


def extract_citations(text: str) -> list[str]:
    """Return unique, order-preserving ``S:A`` citations found in the answer text."""
    seen: list[str] = []
    for s, a in _CITATION_RE.findall(text):
        vid = f"{int(s)}:{int(a)}"
        if vid not in seen:
            seen.append(vid)
    return seen


def build_context_block(verses: list[dict]) -> str:
    """Render retrieved verses into the context the model is allowed to use.

    Format per verse (stable, parseable):
        [S:A] (Surah En)
        Arabic: ...
        English: ...
        Urdu: ...
    """
    blocks = []
    for p in verses:
        names = p.get("surah_name", {})
        blocks.append(
            f"[{p['verse_id']}] ({names.get('en', '')})\n"
            f"Arabic: {p['text_ar']}\n"
            f"English: {p.get('translation_en', '')}\n"
            f"Urdu: {p.get('translation_ur', '')}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context_block: str) -> str:
    if not context_block.strip():
        context_block = "(no verses retrieved)"
    return (
        "Retrieved Quran verses (the ONLY information you may use):\n"
        "----------------------------------------------------------\n"
        f"{context_block}\n"
        "----------------------------------------------------------\n\n"
        f"Question: {question}\n\n"
        "Answer using only the verses above, and cite every verse you use."
    )


def build_history_preamble(pairs: list[tuple[str, str]]) -> str:
    """Render prior turns as *conversational context only* — explicitly NOT evidence.

    ``pairs`` is a list of ``(user_question, assistant_answer)`` for recent turns. This preamble is
    only ever included when ``history_in_prompt`` is enabled; even then it is fenced off and
    labeled so the model treats it as continuity, not as a source of facts. Grounding is still
    enforced downstream: the answer must cite verses from THIS turn's retrieved context, and the
    validator rejects any citation that was not retrieved this turn.
    """
    if not pairs:
        return ""
    lines = [
        "Conversation so far (CONTEXT ONLY — do NOT treat as evidence, do NOT cite from here;",
        "cite only the retrieved verses provided below):",
        "==========================================================",
    ]
    for i, (q, a) in enumerate(pairs, start=1):
        lines.append(f"Turn {i} — User: {q}")
        lines.append(f"Turn {i} — Assistant: {a}")
    lines.append("==========================================================")
    return "\n".join(lines)


# --- Confidence gate (Layer 3) ------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    reason: str = ""


def confidence_gate(
    retrieved: list[tuple[float, dict]],
    *,
    min_score: float,
    min_evidence: int,
    secondary_score: float,
) -> GateResult:
    """Refuse before generation when retrieval is too weak.

    * best hit must clear ``min_score``
    * at least ``min_evidence`` verses must clear ``secondary_score``
    """
    if not retrieved:
        return GateResult(False, "empty")
    top_score = retrieved[0][0]
    if top_score < min_score:
        return GateResult(False, f"top_score {top_score:.3f} < {min_score}")
    on_topic = [1 for s, _ in retrieved if s >= secondary_score]
    if len(on_topic) < min_evidence:
        return GateResult(False, "insufficient_on_topic_evidence")
    return GateResult(True)

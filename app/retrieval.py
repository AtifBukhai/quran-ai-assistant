"""Retrieval: language detection, query routing, and hybrid (dense + lexical) search.

Intents:
* EXACT_REF  — "2:255"          -> direct payload lookup (no vector search)
* SURAH      — "Surah Maryam"   -> surah-name match, list the surah's ayat
* KEYWORD    — "mercy"/"الرحمن" -> lexical scan over the language-appropriate field
* SEMANTIC   — a question       -> dense vector search, blended with lexical overlap

The SEMANTIC score is a calibrated blend: lexical overlap in the query's own language is the
primary, well-scaled signal, and dense cosine is a capped refinement. A verse supported by dense
similarity alone can never clear the confidence gate, so out-of-scope questions — which share no
vocabulary with any verse — are refused deterministically instead of latching onto a spurious
nearest neighbour. ``reciprocal_rank_fusion`` is retained as a utility for rank-only fusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .embeddings import Embedder
from .models import normalize_arabic, normalize_urdu, parse_verse_id
from .vectorstore import VectorStore, build_filter


class Intent(str, Enum):
    EXACT_REF = "exact_ref"
    SURAH = "surah"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"


_ARABIC_RANGE = re.compile(r"[؀-ۿ]")
# Urdu-specific letters not used in standard Arautf script (heuristic for ar vs ur).
_URDU_MARKERS = re.compile(r"[ٹڈڑںھۃیےگچپژ]")
_EXACT_REF_RE = re.compile(r"^\s*(\d{1,3}):(\d{1,3})\s*$")
_SURAH_RE = re.compile(r"^\s*(surah|sura|سورة|سورۃ|surat)\s+(.+)$", re.IGNORECASE)

# Word tokenizer for lexical overlap. Unicode-aware: keeps Arabic/Urdu/Latin letters and
# digits, splits on everything else. Applied to already-normalized (lowercased / diacritic-
# stripped) text, so tokens compare cleanly across the three scripts.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens, dropping 1-2 char noise tokens."""
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 2]


# High-frequency function words across EN / AR / UR that carry no topical signal. Kept short
# and conservative — only words that would otherwise create spurious overlap matches.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "the", "and", "for", "are", "was", "were", "with", "that", "this", "does",
        "did", "has", "have", "had", "you", "your", "our", "his", "her", "its", "their",
        "who", "what", "when", "where", "why", "how", "which", "about", "from", "into",
        "say", "says", "said", "tell", "told", "quran", "koran", "quranic", "verse",
        "verses", "ayah", "ayat", "surah", "surahs", "chapter", "chapters", "please",
        "there", "here", "they", "them", "then", "than", "some", "any", "all", "can",
        "will", "would", "should", "could", "may", "might", "must", "not",
        # Arabic
        "من", "ما", "عن", "في", "على", "الى", "هل", "كيف", "لماذا", "الذي", "التي",
        "هو", "هي", "قال", "قل", "يقول", "ايه", "ايات", "سوره", "القران", "الكريم",
        # Urdu
        "کیا", "کے", "کا", "کی", "میں", "سے", "پر", "اور", "یہ", "وہ", "ہے", "ہیں",
        "کون", "کیوں", "کیسے", "کہاں", "کب", "جو", "کو", "قران", "قرآن", "آیت", "سورہ",
    }
)


def detect_language(text: str) -> str:
    """Return 'ar', 'en', or 'ur'. Heuristic; optionally override with fastText if installed."""
    try:  # optional dependency
        from ftlangdetect import detect  # type: ignore  # noqa: PLC0415

        code = detect(text.replace("\n", " "))["lang"]
        if code in {"ar", "ur", "en"}:
            return code
    except Exception:  # noqa: BLE001 - fall through to heuristic
        pass

    if _ARABIC_RANGE.search(text):
        return "ur" if _URDU_MARKERS.search(text) else "ar"
    return "en"


@dataclass
class RoutedQuery:
    intent: Intent
    language: str
    raw: str
    exact_ref: Optional[tuple[int, int]] = None
    surah_term: Optional[str] = None
    filters: dict = field(default_factory=dict)


# Keyword queries are short (1-3 tokens), no question words/punctuation.
_QUESTION_MARKERS = re.compile(r"[?؟]|who|what|when|where|why|how|کیا|کون|کیوں|کیسے|من|ما|هل|كيف|لماذا")


def route(query: str, language: str, filters: Optional[dict] = None) -> RoutedQuery:
    filters = filters or {}
    q = query.strip()

    m = _EXACT_REF_RE.match(q)
    if m:
        ref = parse_verse_id(q)
        if ref:
            return RoutedQuery(Intent.EXACT_REF, language, q, exact_ref=ref, filters=filters)

    ms = _SURAH_RE.match(q)
    if ms:
        return RoutedQuery(
            Intent.SURAH, language, q, surah_term=ms.group(2).strip(), filters=filters
        )

    tokens = q.split()
    is_short = len(tokens) <= 3
    has_question = bool(_QUESTION_MARKERS.search(q.lower()))
    if is_short and not has_question:
        return RoutedQuery(Intent.KEYWORD, language, q, filters=filters)

    return RoutedQuery(Intent.SEMANTIC, language, q, filters=filters)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    # --- per-intent retrieval -------------------------------------------------
    def exact(self, ref: tuple[int, int]) -> list[tuple[float, dict[str, Any]]]:
        verse_id = f"{ref[0]}:{ref[1]}"
        payload = self.store.get_by_verse_id(verse_id)
        return [(1.0, payload)] if payload else []

    def surah(self, term: str) -> list[tuple[float, dict[str, Any]]]:
        term_l = term.lower().strip("ۃةہ ")
        # Pull a broad set and match on any of the three surah-name fields.
        from qdrant_client.http import models as qm  # noqa: PLC0415

        results: list[tuple[float, dict[str, Any]]] = []
        # scroll the whole (small) collection and filter in Python for fuzzy name match
        points, _ = self.store.client.scroll(
            collection_name=self.store.collection, limit=6300, with_payload=True
        )
        for p in points:
            pl = p.payload or {}
            names = pl.get("surah_name", {})
            haystack = " ".join(
                [str(names.get("ar", "")), str(names.get("en", "")).lower(), str(names.get("ur", ""))]
            )
            if term_l and term_l in haystack.lower():
                results.append((0.99, pl))
        results.sort(key=lambda r: (r[1].get("surah_number", 0), r[1].get("ayah_number", 0)))
        return results

    def _field_for(self, language: str) -> str:
        if language == "ar":
            return "text_ar_normalized"
        if language == "ur":
            return "translation_ur_normalized"
        return "translation_en_lower"

    def _normalize_query(self, query: str, language: str) -> str:
        if language == "ar":
            return normalize_arabic(query)
        if language == "ur":
            return normalize_urdu(query)
        return query.lower()

    def _all_points(self) -> list[dict[str, Any]]:
        points, _ = self.store.client.scroll(
            collection_name=self.store.collection, limit=6300, with_payload=True
        )
        return [p.payload or {} for p in points]

    def keyword(self, query: str, language: str, limit: int) -> list[tuple[float, dict[str, Any]]]:
        """Exact-phrase / substring keyword search (KEYWORD intent, e.g. 'mercy', 'الرحمن')."""
        needle = self._normalize_query(query, language)
        field_name = self._field_for(language)
        scored: list[tuple[float, dict[str, Any]]] = []
        for pl in self._all_points():
            hay = str(pl.get(field_name, ""))
            if needle and needle in hay:
                tf = hay.count(needle)
                scored.append((min(1.0, 0.5 + 0.1 * tf), pl))
        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:limit]

    def lexical_overlap(
        self, query: str, language: str, limit: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Token-overlap lexical search — the reliable lexical arm of SEMANTIC retrieval.

        Scores verses by the fraction of meaningful query tokens they contain, so a question
        like "what does the quran say about patience?" surfaces verses mentioning 'patience'
        even when dense embeddings are weak (e.g. the offline hash embedder).
        """
        field_name = self._field_for(language)
        tokens = [t for t in _tokenize(self._normalize_query(query, language)) if t not in _STOPWORDS]
        if not tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for pl in self._all_points():
            hay_tokens = set(_tokenize(str(pl.get(field_name, ""))))
            matched = sum(1 for t in tokens if t in hay_tokens)
            if matched:
                scored.append((matched / len(tokens), pl))
        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:limit]

    # Blend weights for the semantic score. Lexical overlap (matched in the query's own
    # language against the verse's own-language translation) is the reliable, calibrated
    # signal and dominates. Dense cosine only refines ranking and adds recall — it is capped
    # so that a verse supported by dense similarity ALONE can never clear the confidence gate
    # (max dense contribution 0.25 < default min_score 0.30). This makes out-of-scope queries,
    # which share no vocabulary with any verse, refuse deterministically rather than latch onto
    # a spurious nearest neighbour. It is also the correct conservative policy for a strictly
    # grounded assistant: an answer must be lexically anchored in the cited verse.
    _W_LEXICAL = 0.75
    _W_DENSE = 0.25

    def semantic(
        self, query: str, language: str, limit: int, filters: Optional[dict]
    ) -> list[tuple[float, dict[str, Any]]]:
        qvec = self.embedder.encode([query])[0]
        qfilter = build_filter(filters)
        dense = self.store.search(qvec, limit=limit, query_filter=qfilter)
        lexical = self.lexical_overlap(query, language, limit=limit)

        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for frac, pl in lexical:
            vid = pl.get("verse_id")
            if not vid:
                continue
            scores[vid] = self._W_LEXICAL * frac
            payloads[vid] = pl

        for cos, pl in dense:
            vid = pl.get("verse_id")
            if not vid:
                continue
            payloads.setdefault(vid, pl)
            # Clamp cosine to [0,1]; its weighted share can never exceed _W_DENSE.
            contrib = self._W_DENSE * max(0.0, min(1.0, cos))
            scores[vid] = min(1.0, scores.get(vid, 0.0) + contrib)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(score, payloads[vid]) for vid, score in ordered[:limit]]


def reciprocal_rank_fusion(
    lists: list[list[tuple[float, dict[str, Any]]]], top: int, k: int = 60
) -> list[tuple[float, dict[str, Any]]]:
    """Merge ranked candidate lists via RRF. Returns fused (score, payload), best first."""
    fused: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for lst in lists:
        for rank, (_, payload) in enumerate(lst):
            vid = payload.get("verse_id")
            if not vid:
                continue
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (k + rank + 1)
            payloads[vid] = payload
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return []
    # Normalize fused scores to [0,1] for the confidence gate.
    max_score = ordered[0][1] or 1.0
    return [(score / max_score, payloads[vid]) for vid, score in ordered[:top]]

"""Retrieval: language detection, query routing, and hybrid (dense + lexical) search.

Intents:
* EXACT_REF  — "2:255"          -> direct payload lookup (no vector search)
* SURAH      — "Surah Maryam"   -> surah-name match, list the surah's ayat
* KEYWORD    — "mercy"/"الرحمن" -> lexical scan over the language-appropriate field
* SEMANTIC   — a question       -> dense vector search, blended with lexical overlap

The SEMANTIC score is a calibrated hybrid blend: Okapi BM25 (IDF + TF-saturation + length
normalization) over the query's own language is the primary, well-scaled lexical signal, and
multilingual-e5 dense cosine is a capped refinement that widens/reorders candidates. Dense
similarity alone can never clear the confidence gate (its weighted share stays below min_score),
so semantic hits are *candidates*, not automatic evidence, and out-of-scope questions — which
share no vocabulary with any verse — are refused deterministically instead of latching onto a
spurious nearest neighbour. Concept/synonym expansion adds a small, capped bonus for related
terms across AR/EN/UR. ``reciprocal_rank_fusion`` is retained as a utility for rank-only fusion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .concepts import expand_terms
from .embeddings import Embedder
from .models import normalize_arabic, normalize_urdu, parse_verse_id
from .vectorstore import VectorStore, build_filter


class Intent(str, Enum):
    EXACT_REF = "exact_ref"
    SURAH = "surah"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"


# The three concrete corpus languages, and the sentinel that means "search all of them".
_LANGS: tuple[str, ...] = ("ar", "en", "ur")
ALL_LANGUAGES = "all"


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


# Per-hit credit for a concept-expanded (synonym) match. Small, so a verse matched only
# through expansion needs multiple related terms to approach the weight of one literal hit,
# and expansion never lets an off-topic verse clear the confidence gate on its own.
_EXPANSION_WEIGHT = 0.15

# --- BM25 parameters (Okapi BM25) --------------------------------------------
# k1 controls term-frequency saturation: additional occurrences of a term help less and less.
# b controls document-length normalization: 1.0 = full normalization, 0 = none. These are the
# standard Okapi defaults and work well for short documents like single verses.
_BM25_K1 = 1.5
_BM25_B = 0.75


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
# Question markers are matched WHOLE-WORD, never as substrings: a single content word like
# 'الرحمن' ends in the letters 'من' ("who/from") and 'سليمان' contains 'ما' ("what"), and the
# English 'shower' contains 'how' — a substring test would wrongly flag these as questions and
# force them down the SEMANTIC path (where a lone term can score under the confidence gate and
# be refused) instead of the literal KEYWORD path. So we tokenize and test membership.
_QUESTION_PUNCT = re.compile(r"[?؟]")
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "who", "what", "when", "where", "why", "how",
        "کیا", "کون", "کیوں", "کیسے",
        "من", "ما", "هل", "كيف", "لماذا",
    }
)


def _has_question_marker(query: str) -> bool:
    """True if the query carries interrogative intent — '?'/'؟' punctuation or a whole-word
    question word. Substring matches are deliberately avoided (see note above)."""
    if _QUESTION_PUNCT.search(query):
        return True
    return any(t in _QUESTION_WORDS for t in _TOKEN_RE.findall(query.lower()))


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

    has_question = _has_question_marker(q)
    # KEYWORD is for a single meaningful term ('mercy', 'الرحمن') where exact substring match is
    # the right tool. Multi-word phrases ('women inheritance', 'punishment of pharaoh') go to
    # SEMANTIC so per-token lexical overlap + concept expansion apply, rather than trying to
    # match the whole phrase as one substring.
    meaningful = [t for t in _tokenize(_normalize_for_route(q, language)) if t not in _STOPWORDS]
    if len(meaningful) <= 1 and not has_question:
        return RoutedQuery(Intent.KEYWORD, language, q, filters=filters)

    return RoutedQuery(Intent.SEMANTIC, language, q, filters=filters)


def _normalize_for_route(query: str, language: str) -> str:
    """Language-appropriate normalization used only for token counting during routing."""
    if language == "ar":
        return normalize_arabic(query)
    if language == "ur":
        return normalize_urdu(query)
    return query.lower()


class BM25Index:
    """Okapi BM25 over the corpus, one index per language field.

    Built lazily from the vector store's payloads and cached. Provides IDF-weighted, TF-
    saturated, length-normalized scoring — the real lexical signal that replaces flat token-
    overlap. Scores are returned in an absolute [0,1] band (see ``score``) so a verse sharing no
    query vocabulary scores 0 and is refused by the confidence gate, exactly as before.
    """

    def __init__(self, docs: list[tuple[str, list[str]]]) -> None:
        # docs: list of (verse_id, token_list) for one language field.
        self.n_docs = len(docs)
        self.doc_len: dict[str, int] = {}
        self.doc_tf: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}
        total_len = 0
        for vid, tokens in docs:
            self.doc_len[vid] = len(tokens)
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf[vid] = tf
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.avg_len = (total_len / self.n_docs) if self.n_docs else 0.0
        # Okapi IDF with +1 smoothing so it is always positive (no negative term weights).
        self.idf: dict[str, float] = {
            t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }
        self._max_idf = max(self.idf.values()) if self.idf else 1.0

    def _term_score(self, term: str, vid: str) -> float:
        tf = self.doc_tf.get(vid, {}).get(term, 0)
        if tf == 0:
            return 0.0
        idf = self.idf.get(term, 0.0)
        dl = self.doc_len.get(vid, 0)
        denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * (dl / self.avg_len if self.avg_len else 1))
        return idf * (tf * (_BM25_K1 + 1)) / denom

    def score(self, vid: str, query_terms: list[str]) -> float:
        """Absolute BM25 score in [0,1].

        Normalized against the ideal where every query term is present at TF saturation, so the
        score reflects *how much of the query this verse covers*, not just its rank. A verse with
        no query terms scores 0 (→ refused by the gate); full coverage approaches 1.0.
        """
        if not query_terms:
            return 0.0
        got = sum(self._term_score(t, vid) for t in query_terms)
        # Ideal per-term contribution at saturation (dl == avg_len): idf * (k1+1).
        ideal = sum((self.idf.get(t, self._max_idf)) * (_BM25_K1 + 1) for t in query_terms)
        return min(1.0, got / ideal) if ideal else 0.0


class Retriever:
    def __init__(
        self, store: VectorStore, embedder: Embedder, *, dense_weight: float | None = None
    ) -> None:
        self.store = store
        self.embedder = embedder
        # Blend weights for the SEMANTIC score (see note below). ``dense_weight`` lets the
        # caller raise the dense arm when a real neural embedder is in use; it defaults to the
        # strict lexical-first policy suited to the offline hash embedder.
        if dense_weight is not None:
            dw = max(0.0, min(1.0, dense_weight))
            self._W_DENSE = dw
            self._W_LEXICAL = 1.0 - dw
        # Lazily-built BM25 indexes, one per language ('ar' | 'en' | 'ur'), cached after first use.
        self._bm25: dict[str, BM25Index] = {}

    # --- lexical (BM25) index -------------------------------------------------
    def _verse_tokens(self, pl: dict[str, Any], language: str) -> list[str]:
        """All searchable tokens for a verse in one language: base field + extra translations.

        Mirrors the field-gathering used elsewhere so BM25 and the concept arm see the same text.
        """
        toks = _tokenize(str(pl.get(self._field_for(language), "")))
        if language == "en":
            for key, val in pl.items():
                if key.startswith("translation_en_") and key.endswith("_lower") and key != "translation_en_lower":
                    toks.extend(_tokenize(str(val)))
        elif language == "ur":
            for key, val in pl.items():
                if key.startswith("translation_ur_") and key.endswith("_normalized") and key != "translation_ur_normalized":
                    toks.extend(_tokenize(str(val)))
        return toks

    def _bm25_index(self, language: str) -> BM25Index:
        idx = self._bm25.get(language)
        if idx is None:
            docs = [
                (pl["verse_id"], self._verse_tokens(pl, language))
                for pl in self._all_points()
                if pl.get("verse_id")
            ]
            idx = BM25Index(docs)
            self._bm25[language] = idx
        return idx

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

    @staticmethod
    def _resolve_languages(language: str) -> tuple[str, ...]:
        """Concrete languages to scan for a requested scope.

        An explicit 'ar' | 'en' | 'ur' searches only that language's field(s); anything else —
        the ``ALL_LANGUAGES`` sentinel or an unrecognized code — searches all three. Scoring
        each language with its OWN calibrated index and merging by max (see ``keyword`` /
        ``lexical_overlap``) only widens recall: a verse's score can never drop below its
        single-language value, so the confidence gate thresholds stay valid and an out-of-scope
        query — sharing no vocabulary with any verse in ANY language — still scores 0 and is
        refused. Grounding is unchanged; this only broadens which field can supply the match.
        """
        return (language,) if language in _LANGS else _LANGS

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

    def _keyword_score_for_verse(
        self, pl: dict[str, Any], language: str, needle: str, expanded: set[str]
    ) -> float:
        """Keyword score for ONE verse in ONE language: literal substring hits, else expansion.

        Literal substring matches (base field + that language's extra translations) score
        ``0.5 + 0.1*matches``; with no literal hit, concept-expanded terms checked against the
        verse's own-language tokens and the Arabic text add a small capped bonus. Identical to
        the original single-language logic — ``keyword`` just calls it once per scanned language.
        """
        field_name = self._field_for(language)
        hay = str(pl.get(field_name, ""))
        matches = hay.count(needle) if needle and needle in hay else 0

        # Extra translation fields (translation_en_2_lower, translation_ur_2_normalized, ...).
        if language == "en":
            for key in pl:
                if key.startswith("translation_en_") and key.endswith("_lower") and key != "translation_en_lower":
                    extra_hay = str(pl.get(key, ""))
                    if needle and needle in extra_hay:
                        matches += extra_hay.count(needle)
        elif language == "ur":
            for key in pl:
                if key.startswith("translation_ur_") and key.endswith("_normalized") and key != "translation_ur_normalized":
                    extra_hay = str(pl.get(key, ""))
                    if needle and needle in extra_hay:
                        matches += extra_hay.count(needle)

        if matches > 0:
            return min(1.0, 0.5 + 0.1 * matches)

        # No literal hit — try concept expansion (token-level, incl. Arabic text).
        if expanded:
            hay_tokens = set(_tokenize(str(pl.get(field_name, ""))))
            ar_tokens = set(_tokenize(str(pl.get("text_ar_normalized", ""))))
            exp_hits = sum(1 for t in expanded if t in hay_tokens or t in ar_tokens)
            if exp_hits:
                return min(0.5, _EXPANSION_WEIGHT * exp_hits)
        return 0.0

    def keyword(
        self, query: str, language: str, limit: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """Exact-phrase / substring keyword search (KEYWORD intent, e.g. 'mercy', 'الرحمن').

        When ``language`` is 'ar' | 'en' | 'ur' this searches only that language's field(s).
        When it is ``ALL_LANGUAGES`` (the default for an auto-detected query), it scans the
        Arabic text AND every English/Urdu translation, scoring each language independently and
        keeping the best (max) per verse — so a term is found whichever language it appears in.

        Also applies offline concept expansion so a single topical word — 'justice',
        'marriage', 'parents' — surfaces verses using related terms and their AR/UR
        equivalents, not only the literal string. Literal substring matches always rank
        first; expansion matches fill in below them.
        """
        # Precompute the per-language needle + expansion terms once, then scan the corpus once.
        per_lang: list[tuple[str, str, set[str]]] = []
        for lang in self._resolve_languages(language):
            needle = self._normalize_query(query, lang)
            base_tokens = [t for t in _tokenize(needle) if t not in _STOPWORDS]
            expanded = expand_terms(base_tokens) - set(base_tokens) if base_tokens else set()
            per_lang.append((lang, needle, expanded))

        scored: list[tuple[float, dict[str, Any]]] = []
        for pl in self._all_points():
            best = 0.0
            for lang, needle, expanded in per_lang:
                s = self._keyword_score_for_verse(pl, lang, needle, expanded)
                if s > best:
                    best = s
            if best > 0:
                scored.append((best, pl))
        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:limit]

    def lexical_overlap(
        self, query: str, language: str, limit: int
    ) -> list[tuple[float, dict[str, Any]]]:
        """BM25 lexical search — the reliable, dominant lexical arm of SEMANTIC retrieval.

        Uses Okapi BM25 (IDF weighting + TF saturation + length normalization) so a rare,
        topical query term contributes far more than a common one, and a verse is not rewarded
        for merely being long. This is the real lexical signal; it replaces the earlier flat
        token-overlap fraction. Scores are absolute in [0,1] (coverage of the query), so a verse
        sharing no query vocabulary scores 0 and is refused by the confidence gate — out-of-scope
        questions still fail deterministically rather than latching onto a nearest neighbour.

        When ``language`` is 'ar' | 'en' | 'ur' only that language's BM25 index is used. When it
        is ``ALL_LANGUAGES`` (the default for an auto-detected query), the query is scored against
        each language's OWN calibrated index and the best (max) score per verse is kept. Because
        each index keeps its own [0,1] calibration and we take a max, this only widens recall — a
        verse's score never drops below its single-language value, so the confidence gate stays
        valid and off-topic queries are still refused.

        Concept expansion (offline, no model) adds *related* terms — e.g. 'anger' also matches
        verses about 'wrath'/'rage' and their Arabic/Urdu equivalents. The BM25 score over the
        literal query terms is the primary signal; expansion-only hits add a small, capped bonus
        so a verse matched purely through a synonym ranks below one containing the literal term.
        The method name is retained for call-site compatibility.
        """
        # Precompute, per scanned language, the literal query tokens, expansion terms, and the
        # (cached) BM25 index. Languages whose query has no meaningful tokens are skipped.
        per_lang: list[tuple[str, list[str], set[str], BM25Index]] = []
        for lang in self._resolve_languages(language):
            base_tokens = [
                t for t in _tokenize(self._normalize_query(query, lang)) if t not in _STOPWORDS
            ]
            if not base_tokens:
                continue
            expanded = expand_terms(base_tokens) - set(base_tokens)
            per_lang.append((lang, base_tokens, expanded, self._bm25_index(lang)))
        if not per_lang:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for pl in self._all_points():
            vid = pl.get("verse_id")
            if not vid:
                continue
            ar_tokens: set[str] | None = None  # lazily computed; shared across languages
            best = 0.0
            for lang, base_tokens, expanded, bm25 in per_lang:
                # Primary lexical signal: BM25 over the literal query terms.
                base_score = bm25.score(vid, base_tokens)

                # Additive, capped concept-expansion bonus. Expansion terms are checked against
                # the verse's own-language tokens and (always) the Arabic text, so cross-script
                # synonyms match. Kept small so expansion lifts recall/ranking but never
                # dominates a literal hit.
                exp_frac = 0.0
                if expanded:
                    if ar_tokens is None:
                        ar_tokens = set(_tokenize(str(pl.get("text_ar_normalized", ""))))
                    hay_tokens = set(self._verse_tokens(pl, lang))
                    exp_hits = sum(1 for t in expanded if t in hay_tokens or t in ar_tokens)
                    exp_frac = min(0.5, _EXPANSION_WEIGHT * exp_hits)

                score = min(1.0, base_score + exp_frac)
                if score > best:
                    best = score
            if best > 0:
                scored.append((best, pl))
        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:limit]

    # Default blend weights for the semantic score (used when the caller does not override
    # them via ``dense_weight``). Lexical overlap — matched in the query's own language against
    # the verse's own-language translation — is the reliable, calibrated signal and dominates.
    # With the offline hash embedder, dense cosine only refines ranking and adds recall, and is
    # capped so a verse supported by dense similarity ALONE cannot clear the confidence gate
    # (max dense contribution 0.25 < default min_score 0.30). This makes out-of-scope queries,
    # which share no vocabulary with any verse, refuse deterministically rather than latch onto
    # a spurious nearest neighbour.
    #
    # A real multilingual neural embedder improves *which* verses surface (recall + ranking),
    # but the dense weight stays capped below the gate for it too: semantic similarity assists
    # retrieval, it does not replace grounding. A verse matched by vector similarity alone —
    # 'anger' -> a verse about 'wrath' with no shared or concept-expanded token — is still
    # refused, so every answer remains anchored to a lexical/concept hit plus mandatory
    # citations. (Set QURAN_DENSE_WEIGHT explicitly to deliberately loosen this.) Grounding is
    # further enforced downstream: the validator re-renders cited verse text from the canonical
    # corpus, so the model can never invent or alter it.
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

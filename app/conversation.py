"""Conversation memory — sessions, history, trimming, and follow-up query rewriting.

Grounding posture (identical in spirit to the retrieval guardrails):

    Conversation history assists CONTINUITY, never GROUNDING.

Concretely:
* Every turn still retrieves fresh verses and is validated against them. History is *not* an
  evidence source — a follow-up whose answer is not supported by this-turn's retrieved verses is
  still refused, and the validator still rejects any citation that was not retrieved this turn.
* History is used for exactly two things:
    1. Follow-up query rewriting — expand pronouns / elliptical questions ("what about the next
       one?", "explain it") using prior *user questions and topics* so retrieval can find the
       right verses. We never copy a prior *answer's* text into the new query as if it were fact.
    2. (Optional, off by default) A compact history preamble in the generation prompt, behind
       ``history_in_prompt``. Even then, the system prompt + validator keep answers grounded in
       this-turn's retrieved verses only.

Backends mirror the vector-store swap pattern: an in-memory store for dev/tests (default) and an
optional Redis store for persistence across processes/restarts. Redis is imported lazily so the
dependency is only needed when ``QURAN_CONVERSATION_BACKEND=redis``.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol


# --- data models --------------------------------------------------------------
@dataclass
class ConversationTurn:
    """One user question and the grounded outcome we produced for it.

    We store the answer text and citations for continuity and for the "previous citations"
    feature, but they are treated as a *record of what happened*, never as evidence for future
    turns.
    """

    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    status: str = "answered"
    language: str = "en"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        return cls(
            query=d.get("query", ""),
            answer=d.get("answer", ""),
            citations=list(d.get("citations", [])),
            status=d.get("status", "answered"),
            language=d.get("language", "en"),
            created_at=float(d.get("created_at", time.time())),
        )


@dataclass
class Conversation:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            session_id=d["session_id"],
            turns=[ConversationTurn.from_dict(t) for t in d.get("turns", [])],
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def previous_citations(self) -> list[str]:
        """Unique, order-preserving list of every verse cited so far in this session."""
        seen: list[str] = []
        for t in self.turns:
            for c in t.citations:
                if c not in seen:
                    seen.append(c)
        return seen


def new_session_id() -> str:
    return uuid.uuid4().hex


# --- token estimation & trimming ----------------------------------------------
def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Good enough for budgeting history. Counts word-ish chunks and CJK/Arabic runs; errs slightly
    high so we stay *under* real model limits rather than over.
    """
    if not text:
        return 0
    # Split on whitespace and punctuation; count non-empty pieces. Add a fudge for subword
    # tokenization (~1.3 tokens per whitespace word is typical for multilingual text).
    words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, int(len(words) * 1.3))


def trim_turns(
    turns: list[ConversationTurn],
    *,
    max_turns: int,
    token_budget: int,
) -> list[ConversationTurn]:
    """Keep the most recent turns within both a turn cap and a token budget.

    Returns turns in chronological order (oldest kept -> newest). The newest turns are the most
    relevant for follow-up continuity, so we fill the budget from the end backwards.
    """
    if not turns:
        return []
    # Turn cap first (cheap), newest last.
    capped = turns[-max_turns:] if max_turns > 0 else list(turns)

    # Then token budget, walking from newest backwards until we'd overflow.
    kept: list[ConversationTurn] = []
    used = 0
    for turn in reversed(capped):
        cost = estimate_tokens(turn.query) + estimate_tokens(turn.answer)
        if kept and used + cost > token_budget:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept


# --- follow-up query rewriting ------------------------------------------------
# Signals that a query is elliptical / anaphoric and needs prior context to be retrievable on its
# own. Kept deliberately small and precise so we don't rewrite self-contained questions.
_FOLLOWUP_PATTERNS = [
    r"\b(it|its|this|that|those|these|them|they|he|she|his|her)\b",
    r"\b(next|previous|another|more|other|same)\b",
    r"\b(explain|elaborate|expand|continue|clarify|why|how come)\b",
    r"^\s*(what about|and|but|so|then|also)\b",
    r"\b(the (verse|ayah|surah))\b",
]
_FOLLOWUP_RE = re.compile("|".join(_FOLLOWUP_PATTERNS), re.IGNORECASE)

# Stopwords stripped when harvesting topical keywords from prior *questions* (English-centric; AR
# and UR topical words are content words already and pass through).
_STOPWORDS = {
    "what", "does", "the", "say", "about", "of", "in", "on", "a", "an", "is", "are", "to",
    "and", "or", "how", "should", "we", "our", "do", "quran", "allah", "verse", "verses",
    "ayah", "surah", "tell", "me", "give", "for", "with", "that", "this", "it", "his", "her",
    "explain", "why", "when", "who", "can", "you",
}

# Purely referential words: a query made ONLY of these (plus stopwords/triggers) cannot stand on
# its own and needs prior context to be retrievable. A query that also carries its own content
# terms is self-contained and must NOT borrow prior topics (that would answer the wrong question).
_ANAPHORA = {
    "it", "its", "this", "that", "those", "these", "them", "they", "he", "she", "his", "her",
    "one", "ones", "more", "next", "previous", "another", "other", "same", "again", "former",
    "latter", "above", "below", "former", "elaborate", "expand", "continue", "clarify",
}


def looks_like_followup(query: str) -> bool:
    """Heuristic: does this query depend on earlier context to be retrievable?"""
    q = query.strip()
    # Very short queries are usually follow-ups ("more?", "the next one").
    if len(q.split()) <= 3:
        return True
    return bool(_FOLLOWUP_RE.search(q))


def _topic_terms(text: str) -> list[str]:
    """Content words from a prior user question, used to disambiguate a follow-up."""
    terms: list[str] = []
    for tok in re.findall(r"\w+", text, flags=re.UNICODE):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if len(tok) <= 2 and tok.isascii():
            continue
        if low not in (t.lower() for t in terms):
            terms.append(tok)
    return terms


def _own_content_terms(query: str) -> list[str]:
    """Content terms the query carries by itself — excluding stopwords AND referential words.

    If this is non-empty, the query names its own topic (e.g. "what about *inheritance*?") and is
    self-contained: we must not graft prior topics onto it. If it is empty, the query is pure
    anaphora ("explain it", "why?", "tell me more") and genuinely needs prior context.
    """
    terms: list[str] = []
    for tok in re.findall(r"\w+", query, flags=re.UNICODE):
        low = tok.lower()
        if low in _STOPWORDS or low in _ANAPHORA:
            continue
        if low.isdigit():  # bare years/numbers are their own content ("Apple in 2024")
            terms.append(tok)
            continue
        if len(tok) <= 2 and tok.isascii():
            continue
        terms.append(tok)
    return terms


def rewrite_followup(query: str, history: list[ConversationTurn], *, max_terms: int = 6) -> str:
    """Expand a follow-up query with topical terms from recent prior *questions*.

    IMPORTANT grounding note: we only borrow words from earlier user *questions* (topics the user
    themselves raised), never from prior answer text or cited verse content. The result is used
    solely to steer retrieval; the retrieved verses are re-scored and re-validated from scratch.

    We borrow ONLY when the query cannot stand on its own — i.e. it is pure anaphora with no
    content terms of its own. A follow-up that introduces a new nameable topic (even an
    out-of-scope one like "what about Apple's stock price?") keeps its own words untouched, so it
    retrieves on its own terms and is correctly refused when the Quran has no grounding for it.
    """
    if not history or not looks_like_followup(query):
        return query
    # Self-contained query (brings its own topic) — never graft prior context onto it.
    if _own_content_terms(query):
        return query

    # Gather topic terms from the most recent questions (newest first), de-duplicated.
    borrowed: list[str] = []
    for turn in reversed(history):
        for term in _topic_terms(turn.query):
            if term.lower() not in (b.lower() for b in borrowed) and term.lower() not in query.lower():
                borrowed.append(term)
        if len(borrowed) >= max_terms:
            break
    if not borrowed:
        return query

    context = " ".join(borrowed[:max_terms])
    # Append as extra retrieval context; keep the user's own words first so their intent leads.
    return f"{query} {context}".strip()


# --- history store backends ---------------------------------------------------
class HistoryStore(Protocol):
    def create(self) -> Conversation: ...
    def get(self, session_id: str) -> Optional[Conversation]: ...
    def append(self, session_id: str, turn: ConversationTurn) -> Conversation: ...
    def delete(self, session_id: str) -> bool: ...


class InMemoryHistoryStore:
    """Process-local session store with TTL and a turn cap. Default backend (dev/tests).

    Not shared across processes and cleared on restart — use the Redis backend for persistence.
    """

    def __init__(self, *, ttl_seconds: int, max_turns: int) -> None:
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._data: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def _expired(self, conv: Conversation) -> bool:
        return self._ttl > 0 and (time.time() - conv.updated_at) > self._ttl

    def create(self) -> Conversation:
        conv = Conversation(session_id=new_session_id())
        with self._lock:
            self._data[conv.session_id] = conv
        return conv

    def get(self, session_id: str) -> Optional[Conversation]:
        with self._lock:
            conv = self._data.get(session_id)
            if conv is None:
                return None
            if self._expired(conv):
                self._data.pop(session_id, None)
                return None
            return conv

    def append(self, session_id: str, turn: ConversationTurn) -> Conversation:
        with self._lock:
            conv = self._data.get(session_id)
            if conv is None or self._expired(conv):
                conv = Conversation(session_id=session_id)
            conv.turns.append(turn)
            if self._max_turns > 0:
                conv.turns = conv.turns[-self._max_turns :]
            conv.updated_at = time.time()
            self._data[session_id] = conv
            return conv

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None


class RedisHistoryStore:
    """Redis-backed session store for persistent, cross-process conversation history.

    Each conversation is one JSON string at key ``quran:session:{id}`` with a sliding TTL refreshed
    on every append (via EXPIRE). Redis is imported lazily so it is only required when this backend
    is actually selected.
    """

    def __init__(self, *, url: str, ttl_seconds: int, max_turns: int, key_prefix: str = "quran:session:") -> None:
        try:
            import redis  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - only hit without redis installed
            raise RuntimeError(
                "conversation_backend='redis' requires the 'redis' package. "
                "Install it with: pip install redis"
            ) from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _save(self, conv: Conversation) -> None:
        key = self._key(conv.session_id)
        self._redis.set(key, json.dumps(conv.to_dict()))
        if self._ttl > 0:
            self._redis.expire(key, self._ttl)

    def create(self) -> Conversation:
        conv = Conversation(session_id=new_session_id())
        self._save(conv)
        return conv

    def get(self, session_id: str) -> Optional[Conversation]:
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return None
        try:
            return Conversation.from_dict(json.loads(raw))
        except (json.JSONDecodeError, KeyError):
            return None

    def append(self, session_id: str, turn: ConversationTurn) -> Conversation:
        conv = self.get(session_id) or Conversation(session_id=session_id)
        conv.turns.append(turn)
        if self._max_turns > 0:
            conv.turns = conv.turns[-self._max_turns :]
        conv.updated_at = time.time()
        self._save(conv)
        return conv

    def delete(self, session_id: str) -> bool:
        return bool(self._redis.delete(self._key(session_id)))


def build_history_store(
    backend: str,
    *,
    redis_url: str,
    ttl_seconds: int,
    max_turns: int,
) -> HistoryStore:
    """Factory mirroring the embedder/vectorstore swap pattern."""
    if backend == "redis":
        return RedisHistoryStore(url=redis_url, ttl_seconds=ttl_seconds, max_turns=max_turns)
    return InMemoryHistoryStore(ttl_seconds=ttl_seconds, max_turns=max_turns)

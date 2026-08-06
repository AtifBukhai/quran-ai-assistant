"""Conversation memory tests — with the grounding invariant as the headline assertion.

Two classes of tests:
1. Unit tests for the store / trimming / follow-up rewrite (offline, no orchestrator).
2. End-to-end tests through the Orchestrator proving that conversation memory NEVER weakens
   grounding: follow-ups still retrieve fresh, still cite only this-turn's verses, and an
   out-of-scope follow-up is still refused even mid-conversation.
"""

from __future__ import annotations

import time

from app.conversation import (
    Conversation,
    ConversationTurn,
    InMemoryHistoryStore,
    RedisHistoryStore,  # noqa: F401  (imported to ensure module surface stays importable)
    build_history_store,
    estimate_tokens,
    looks_like_followup,
    rewrite_followup,
    trim_turns,
)
from app.models import AskRequest


# --- store: create / append / get / delete -----------------------------------
def test_inmemory_create_and_append():
    store = InMemoryHistoryStore(ttl_seconds=3600, max_turns=10)
    conv = store.create()
    assert conv.session_id
    store.append(conv.session_id, ConversationTurn(query="q1", answer="a1 [2:153]", citations=["2:153"]))
    got = store.get(conv.session_id)
    assert got is not None
    assert len(got.turns) == 1
    assert got.turns[0].citations == ["2:153"]


def test_inmemory_turn_cap_enforced():
    store = InMemoryHistoryStore(ttl_seconds=3600, max_turns=3)
    conv = store.create()
    for i in range(5):
        store.append(conv.session_id, ConversationTurn(query=f"q{i}", answer=f"a{i}"))
    got = store.get(conv.session_id)
    assert len(got.turns) == 3
    # Only the most recent three survive.
    assert [t.query for t in got.turns] == ["q2", "q3", "q4"]


def test_inmemory_ttl_expiry():
    store = InMemoryHistoryStore(ttl_seconds=1, max_turns=10)
    conv = store.create()
    store.append(conv.session_id, ConversationTurn(query="q", answer="a"))
    # Force expiry by backdating updated_at.
    stored = store.get(conv.session_id)
    stored.updated_at = time.time() - 5
    assert store.get(conv.session_id) is None


def test_inmemory_delete():
    store = InMemoryHistoryStore(ttl_seconds=3600, max_turns=10)
    conv = store.create()
    assert store.delete(conv.session_id) is True
    assert store.get(conv.session_id) is None
    assert store.delete(conv.session_id) is False


def test_build_history_store_defaults_to_memory():
    store = build_history_store("memory", redis_url="", ttl_seconds=60, max_turns=5)
    assert isinstance(store, InMemoryHistoryStore)


# --- previous citations -------------------------------------------------------
def test_previous_citations_dedup_order():
    conv = Conversation(session_id="s")
    conv.turns.append(ConversationTurn(query="q1", answer="a", citations=["2:153", "103:3"]))
    conv.turns.append(ConversationTurn(query="q2", answer="a", citations=["103:3", "2:155"]))
    assert conv.previous_citations() == ["2:153", "103:3", "2:155"]


# --- token estimation & trimming ----------------------------------------------
def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    short = estimate_tokens("patience")
    long = estimate_tokens("patience and perseverance in the face of hardship")
    assert 0 < short < long


def test_trim_turns_respects_token_budget():
    turns = [ConversationTurn(query=f"question number {i}", answer="a fairly long answer " * 10) for i in range(10)]
    kept = trim_turns(turns, max_turns=10, token_budget=60)
    # Budget is tight, so not all 10 survive, and the ones that do are the most recent.
    assert 0 < len(kept) < 10
    assert kept[-1].query == "question number 9"


def test_trim_turns_respects_turn_cap():
    turns = [ConversationTurn(query=f"q{i}", answer="a") for i in range(10)]
    kept = trim_turns(turns, max_turns=4, token_budget=100000)
    assert len(kept) == 4
    assert [t.query for t in kept] == ["q6", "q7", "q8", "q9"]


def test_trim_turns_keeps_at_least_newest_even_if_over_budget():
    turns = [ConversationTurn(query="q", answer="word " * 500)]
    kept = trim_turns(turns, max_turns=10, token_budget=5)
    # A single over-budget turn is still kept (we never return empty when turns exist).
    assert len(kept) == 1


# --- follow-up detection & rewrite --------------------------------------------
def test_looks_like_followup_detects_anaphora():
    assert looks_like_followup("explain it")
    assert looks_like_followup("what about the next one?")
    assert looks_like_followup("why?")
    # Self-contained question is not a follow-up.
    assert not looks_like_followup("What does the Quran say about charity and almsgiving?")


def test_rewrite_followup_borrows_prior_topic_terms():
    history = [ConversationTurn(query="What does the Quran say about patience?", answer="... [2:153]")]
    rewritten = rewrite_followup("explain it more", history)
    # The topical word "patience" from the prior question steers retrieval.
    assert "patience" in rewritten.lower()
    # The user's own words are preserved.
    assert "explain" in rewritten.lower()


def test_rewrite_followup_leaves_selfcontained_query_untouched():
    history = [ConversationTurn(query="Tell me about patience", answer="... [2:153]")]
    q = "What does the Quran say about inheritance shares for daughters?"
    assert rewrite_followup(q, history) == q


def test_rewrite_followup_no_history_is_identity():
    assert rewrite_followup("explain it", []) == "explain it"


# --- END-TO-END grounding invariant through the orchestrator ------------------
def test_session_roundtrip_records_turns(orch):
    conv = orch.history.create()
    res = orch.ask(AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id))
    assert res.session_id == conv.session_id
    stored = orch.history.get(conv.session_id)
    assert stored is not None
    assert len(stored.turns) == 1
    assert stored.turns[0].query.startswith("What does the Quran say about patience")


def test_followup_stays_grounded_and_cites_only_this_turn(orch):
    """A follow-up must cite only verses retrieved THIS turn — never smuggle prior citations."""
    conv = orch.history.create()
    first = orch.ask(AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id))
    assert first.citations  # patience verses present in the sample corpus

    follow = orch.ask(AskRequest(query="tell me more about it", session_id=conv.session_id))
    if follow.citations:
        # Every citation corresponds to returned evidence from this turn (validator invariant).
        assert set(follow.citations) == {e.verse_id for e in follow.evidence}


def test_out_of_scope_followup_still_refused(orch):
    """Conversation context must NOT let an unanswerable follow-up slip past the guardrails."""
    conv = orch.history.create()
    orch.ask(AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id))
    # A follow-up that cannot be answered from the Quran must still be refused, even though the
    # session has prior grounded turns. History assists continuity, not grounding.
    res = orch.ask(AskRequest(query="what about the stock price of Apple in 2024?", session_id=conv.session_id))
    assert res.is_refusal
    assert res.citations == []


def test_history_default_off_does_not_leak_into_prompt(orch):
    """With history_in_prompt off (default), answers are identical with or without a session:
    grounding depends only on this-turn retrieval, not on accumulated conversation."""
    assert orch.settings.history_in_prompt is False
    stateless = orch.ask(AskRequest(query="What does the Quran say about patience?"))
    conv = orch.history.create()
    orch.ask(AskRequest(query="Tell me about charity", session_id=conv.session_id))  # seed unrelated history
    with_session = orch.ask(
        AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id)
    )
    # Same grounded citation set regardless of prior (unrelated) conversation.
    assert set(stateless.citations) == set(with_session.citations)


def test_use_history_false_disables_rewrite(orch):
    conv = orch.history.create()
    orch.ask(AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id))
    # With use_history=False the follow-up is not expanded; a bare pronoun query should not
    # inherit "patience" and so should refuse or return unrelated — but crucially still grounded.
    res = orch.ask(AskRequest(query="explain it", session_id=conv.session_id, use_history=False))
    if res.citations:
        assert set(res.citations) == {e.verse_id for e in res.evidence}


def test_delete_session_forgets_history(orch):
    conv = orch.history.create()
    orch.ask(AskRequest(query="What does the Quran say about patience?", session_id=conv.session_id))
    assert orch.history.delete(conv.session_id) is True
    assert orch.history.get(conv.session_id) is None

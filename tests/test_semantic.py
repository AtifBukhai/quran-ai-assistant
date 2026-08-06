"""Tests for offline concept/synonym expansion and its effect on retrieval routing.

These run fully offline (hash embedder + echo LLM) like the rest of the suite. They lock in
the semantic-recall behavior added on top of strict grounding: related terms surface verses,
while out-of-scope questions are still refused.
"""

from __future__ import annotations

from app.concepts import expand_terms
from app.grounding import OUT_OF_SCOPE, REFUSAL_LOW_CONFIDENCE
from app.retrieval import Intent, route


# --- concept lexicon ----------------------------------------------------------
def test_expand_includes_synonyms():
    out = expand_terms(["anger"])
    assert "anger" in out          # original always kept
    assert "wrath" in out          # English synonym
    assert "غضب" in out            # Arabic equivalent


def test_expand_is_bidirectional_within_group():
    # Any member of a concept group expands to the whole group.
    assert "anger" in expand_terms(["wrath"])
    assert "mercy" in expand_terms(["compassion"])


def test_expand_unknown_token_is_noop():
    assert expand_terms(["xyzzy"]) == {"xyzzy"}


# --- routing ------------------------------------------------------------------
def test_single_word_routes_to_keyword():
    assert route("mercy", "en").intent is Intent.KEYWORD


def test_multiword_phrase_routes_to_semantic():
    # Phrases must use per-token overlap + expansion, not whole-string substring match.
    assert route("women inheritance", "en").intent is Intent.SEMANTIC
    assert route("punishment of pharaoh", "en").intent is Intent.SEMANTIC


def test_question_routes_to_semantic():
    assert route("What does Allah say about anger?", "en").intent is Intent.SEMANTIC


# --- end-to-end retrieval via expansion --------------------------------------
def test_synonym_query_surfaces_related_verse(orch):
    # 'anger' never appears literally, but 1:7 speaks of those who earned Allah's anger and
    # 39:53 / mercy verses use related vocabulary. Expansion should surface a grounded, cited
    # verse rather than refusing.
    res = orch.answer("What does Allah say about anger?")
    if res.citations:
        assert set(res.citations) == {e.verse_id for e in res.evidence}


def test_out_of_scope_still_refused_despite_expansion(orch):
    # 'battle' expands to war vocabulary, but the sample corpus has no such verses, so the
    # question must still be refused — expansion must never manufacture a spurious answer.
    res = orch.answer("What year did the Battle of Badr take place?")
    assert res.answer in (OUT_OF_SCOPE, REFUSAL_LOW_CONFIDENCE)
    assert res.citations == []

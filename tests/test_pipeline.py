"""End-to-end grounding tests through the Orchestrator (offline: echo LLM + hash embeddings)."""

from __future__ import annotations

from app.grounding import OUT_OF_SCOPE, REFUSAL_LOW_CONFIDENCE


def test_exact_verse_lookup(orch):
    res = orch.answer("112:1")
    assert res.mode == "exact"
    assert res.citations == ["112:1"]
    ev = res.evidence[0]
    assert ev.verse_id == "112:1"
    assert ev.text_ar.startswith("قُلْ")
    assert ev.translation_en
    assert ev.translation_ur


def test_exact_verse_not_in_corpus_declines(orch):
    # Well-formed reference but not in the sample corpus -> out_of_scope refusal, no fabrication.
    res = orch.answer("2:100")
    assert res.citations == []
    assert res.evidence == []
    assert res.is_refusal
    # Exact ref miss returns OUT_OF_SCOPE (never routed through confidence gate).
    assert res.answer == OUT_OF_SCOPE


def test_semantic_answer_is_grounded_and_cited(orch):
    res = orch.answer("What does the Quran say about patience?")
    assert res.citations, "must return at least one citation"
    # Every citation must correspond to a piece of returned evidence.
    ev_ids = {e.verse_id for e in res.evidence}
    assert set(res.citations) == ev_ids
    # Patience verses (2:153 / 103:3) should surface.
    assert ev_ids & {"2:153", "103:3"}


def test_every_answer_has_all_three_languages(orch):
    res = orch.answer("Who created mankind?")
    assert res.citations
    for ev in res.evidence:
        assert ev.text_ar.strip()
        assert ev.translation_en.strip()
        assert ev.translation_ur.strip()


def test_arabic_query_retrieves_same_domain(orch):
    res = orch.answer("من خلق الإنسان؟")  # "who created man?"
    assert res.mode in ("semantic", "keyword")
    # Should surface a creation verse present in the sample corpus.
    ev_ids = {e.verse_id for e in res.evidence}
    assert ev_ids & {"15:26", "96:2", "55:3", "96:1"}


def test_urdu_query_returns_urdu(orch):
    res = orch.answer("اللہ ایک ہے")  # "Allah is one"
    if res.citations:
        assert any(e.translation_ur.strip() for e in res.evidence)


def test_out_of_scope_question_declined(orch):
    # Depends on Hadith/history, not answerable from Quran text alone.
    res = orch.answer("What year did the Battle of Badr take place?")
    assert res.answer in (OUT_OF_SCOPE, REFUSAL_LOW_CONFIDENCE)
    assert res.citations == []


def test_no_answer_without_citation_invariant(orch):
    # Sweep a range of questions; any non-refusal answer MUST carry citations.
    questions = [
        "What does the Quran say about mercy?",
        "How should believers seek help?",
        "Tell me about forgiveness.",
        "asdfghjkl qwerty nonsense",
    ]
    for q in questions:
        res = orch.answer(q)
        if res.citations:
            assert set(res.citations) == {e.verse_id for e in res.evidence}
        else:
            # No citations -> must be one of the exact refusal strings.
            assert res.is_refusal

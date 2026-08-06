"""The RAG orchestrator — the full grounded pipeline.

User Question
  -> detect language
  -> route intent
  -> retrieve (exact | surah | keyword | semantic)
  -> confidence gate           (weak -> refuse, LLM never called)
  -> build context (ONLY retrieved verses)
  -> grounded generation       (constrained rewriter)
  -> validate                  (citations exist? retrieved? >=1? text fidelity?)
  -> assemble response         (Answer + AR/EN/UR evidence + citations)
"""

from __future__ import annotations

import uuid

from .config import Settings, get_settings
from .conversation import (
    ConversationTurn,
    HistoryStore,
    build_history_store,
    rewrite_followup,
    trim_turns,
)
from .embeddings import build_embedder
from .grounding import (
    OUT_OF_SCOPE,
    REFUSAL_LOW_CONFIDENCE,
    build_context_block,
    build_history_preamble,
    build_user_prompt,
    confidence_gate,
    SYSTEM_PROMPT,
)
from .llm.base import GroundedLLM
from .llm.echo import EchoGroundedLLM
from .models import AskRequest, AskResponse, EvidenceVerse, LanguageMode
from .retrieval import Intent, Retriever, detect_language, route
from .validator import validate
from .vectorstore import VectorStore


def _build_llm(settings: Settings) -> GroundedLLM:
    if settings.llm_backend == "anthropic":
        from .llm.anthropic_llm import AnthropicGroundedLLM  # noqa: PLC0415

        return AnthropicGroundedLLM(settings.llm_model)
    return EchoGroundedLLM()


def _apply_mode(ev: EvidenceVerse, mode: LanguageMode) -> EvidenceVerse:
    """Arabic always stays; drop EN/UR translations not requested by the display mode."""
    show_en = mode in ("ar_en", "ar_en_ur")
    show_ur = mode in ("ar_ur", "ar_en_ur")
    return ev.model_copy(
        update={
            "translation_en": ev.translation_en if show_en else None,
            "translation_ur": ev.translation_ur if show_ur else None,
        }
    )


class Orchestrator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = VectorStore(
            collection=self.settings.qdrant_collection,
            dim=self.settings.embedding_dim,
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        self.store.ensure()
        self.embedder = build_embedder(
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.embedding_dim,
        )
        self.retriever = Retriever(self.store, self.embedder, dense_weight=self.settings.dense_weight)
        self.llm = _build_llm(self.settings)
        self.history: HistoryStore = build_history_store(
            self.settings.conversation_backend,
            redis_url=self.settings.redis_url,
            ttl_seconds=self.settings.session_ttl_seconds,
            max_turns=self.settings.max_history_turns,
        )

    # --- retrieval dispatch ---------------------------------------------------
    def _retrieve(self, routed) -> list[tuple[float, dict]]:
        s = self.settings
        if routed.intent is Intent.EXACT_REF:
            return self.retriever.exact(routed.exact_ref)
        if routed.intent is Intent.SURAH:
            return self.retriever.surah(routed.surah_term)[: s.top_n]
        if routed.intent is Intent.KEYWORD:
            return self.retriever.keyword(routed.raw, routed.language, s.top_k)[: s.top_n]
        return self.retriever.semantic(
            routed.raw, routed.language, s.top_k, routed.filters
        )[: s.top_n]

    # --- main entry -----------------------------------------------------------
    def answer(self, query: str, *, mode: str = "ar_en_ur", lang: str | None = None) -> AskResponse:
        """Convenience wrapper around :meth:`ask` for a plain string query."""
        from .models import AskRequest  # noqa: PLC0415

        return self.ask(AskRequest(query=query, mode=mode, lang=lang))  # type: ignore[arg-type]

    def ask(self, req: AskRequest) -> AskResponse:
        # --- load conversation history (continuity only — never evidence) -----------------
        history: list[ConversationTurn] = []
        session_id = req.session_id
        if session_id and req.use_history:
            conv = self.history.get(session_id)
            if conv:
                history = trim_turns(
                    conv.turns,
                    max_turns=self.settings.max_history_turns,
                    token_budget=self.settings.history_token_budget,
                )

        # --- follow-up rewrite steers RETRIEVAL ONLY --------------------------------------
        # The model still answers the user's actual question, grounded in THIS turn's freshly
        # retrieved verses. We only rewrite topical (keyword/semantic) queries — structural
        # lookups (exact ref, whole surah) are self-contained and must not be perturbed.
        retrieval_query = req.query
        if history:
            probe = route(req.query, req.lang or detect_language(req.query), req.filters)
            if probe.intent in (Intent.KEYWORD, Intent.SEMANTIC):
                retrieval_query = rewrite_followup(req.query, history)

        resp = self._ask_core(req, retrieval_query=retrieval_query, history=history)

        # --- record the turn (every outcome, including refusals) --------------------------
        if session_id:
            self.history.append(
                session_id,
                ConversationTurn(
                    query=req.query,
                    answer=resp.answer,
                    citations=list(resp.citations),
                    status=resp.status,
                    language=resp.language,
                ),
            )
            resp.session_id = session_id
        return resp

    def _ask_core(
        self,
        req: AskRequest,
        *,
        retrieval_query: str,
        history: list[ConversationTurn],
    ) -> AskResponse:
        """The stateless grounded pipeline. ``retrieval_query`` may be a follow-up-expanded form
        of ``req.query``; ``history`` is used only for the optional (off-by-default) prompt
        preamble. Grounding is unchanged: answers cite only verses retrieved this turn."""
        trace_id = uuid.uuid4().hex
        language = req.lang or detect_language(req.query)
        routed = route(retrieval_query, language, req.filters)
        intent = routed.intent.value

        retrieved = self._retrieve(routed)

        # For an exact reference the user asked for a specific verse: return it directly,
        # rendered from canonical corpus, without invoking the generator.
        if routed.intent is Intent.EXACT_REF:
            if not retrieved:
                return self._refuse(
                    "out_of_scope", OUT_OF_SCOPE, language, trace_id, mode=intent
                )
            return self._direct_verse(retrieved, req.mode, language, trace_id)

        # Confidence gate (Layer 3) — refuse before generation if retrieval is weak.
        gate = confidence_gate(
            retrieved,
            min_score=self.settings.min_score,
            min_evidence=self.settings.min_evidence,
            secondary_score=self.settings.secondary_score,
        )
        if not gate.passed:
            return self._refuse(
                "refused_low_confidence", REFUSAL_LOW_CONFIDENCE, language, trace_id, mode=intent
            )

        # Build context from ONLY the retrieved verses and generate.
        context_verses = [p for _, p in retrieved]
        context_block = build_context_block(context_verses)
        user_prompt = build_user_prompt(req.query, context_block)

        # Optional continuity: prepend a fenced, CONTEXT-ONLY history preamble. Off by default
        # (history_in_prompt). Even when on, the preamble is labeled "do not cite from here" and
        # the validator still rejects any citation not retrieved this turn.
        if self.settings.history_in_prompt and history:
            preamble = build_history_preamble([(t.query, t.answer) for t in history])
            if preamble:
                user_prompt = f"{preamble}\n\n{user_prompt}"

        raw_answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        # If the model itself refused, surface the insufficient-info refusal.
        if raw_answer.strip() == self._insufficient():
            return self._refuse(
                "refused_insufficient", self._insufficient(), language, trace_id, mode=intent
            )

        # Validate (Layer 4).
        result = validate(raw_answer, retrieved)
        if not result.ok:
            return self._refuse(
                "refused_insufficient", self._insufficient(), language, trace_id, mode=intent
            )

        evidence = [_apply_mode(ev, req.mode) for ev in result.evidence]
        return AskResponse(
            status="answered",
            language=language,
            answer=result.answer,
            mode=intent,
            evidence=evidence,
            citations=result.citations,
            trace_id=trace_id,
        )

    # --- helpers --------------------------------------------------------------
    def _direct_verse(self, retrieved, mode, language, trace_id) -> AskResponse:
        from .validator import _to_evidence  # noqa: PLC0415

        score, payload = retrieved[0]
        ev = _apply_mode(_to_evidence(payload, score), mode)
        return AskResponse(
            status="answered",
            language=language,
            answer=f"[{payload['verse_id']}]",
            mode="exact",
            evidence=[ev],
            citations=[payload["verse_id"]],
            trace_id=trace_id,
        )

    @staticmethod
    def _insufficient() -> str:
        from .grounding import REFUSAL_INSUFFICIENT  # noqa: PLC0415

        return REFUSAL_INSUFFICIENT

    @staticmethod
    def _refuse(status, message, language, trace_id, *, mode="") -> AskResponse:
        return AskResponse(
            status=status,
            language=language,
            answer=message,
            mode=mode,
            evidence=[],
            citations=[],
            trace_id=trace_id,
        )

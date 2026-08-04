# Quran AI Assistant — System Architecture & Design

**Version:** 1.0
**Date:** 2026-08-04
**Status:** Design specification (pre-implementation)
**Governing principle:** The Quran is the single source of truth. Every answer must be traceable to one or more retrieved Quranic verses, or the assistant declines.

---

## 1. Executive Summary

The Quran AI Assistant is a production-grade, strictly grounded Retrieval-Augmented Generation (RAG) system. It answers user questions **only** from a controlled corpus of Quran verses (Arabic Uthmani + English and Urdu translations). It performs verse-level semantic and lexical retrieval, feeds **only** the retrieved verses to a generation model, and returns a concise answer together with the supporting verses and citations. If retrieval does not surface sufficient supporting evidence, the assistant refuses rather than speculating.

The single hardest engineering requirement is not retrieval quality — it is **groundedness enforcement**. A conventional LLM will happily answer Quran questions from its pretraining. This design therefore treats the generation model purely as a *constrained rewriter over supplied text*, never as a knowledge source, and layers deterministic guardrails (citation validation, verse-existence checks, confidence gating) around it so that ungrounded output cannot reach the user.

### Design decisions locked for this build

| Decision | Choice | Rationale |
|---|---|---|
| Generation model | Cloud LLM (e.g. Anthropic Claude), **strict grounding** | Fed only retrieved verses; used as a constrained text generator, not a knowledge source. The "no external APIs" rule targets *religious-knowledge* sources — API-as-compute is permitted, API-as-oracle is forbidden and structurally prevented. |
| Embeddings | Local multilingual model (self-hosted) | No verse content leaves the environment for indexing; deterministic, reproducible vectors; cross-lingual retrieval. |
| Vector database | **Qdrant** | Rich payload filtering, hybrid (dense + sparse) search, quantization, horizontal scale, runs as a service. |
| Dataset | **Open source** — Tanzil (CC BY 3.0): Arabic Uthmani + Sahih International (EN) + Fateh Muhammad Jalandhry (UR) | Authoritative, redistributable, well-structured, verse-aligned. |

---

## 2. Non-Negotiable Requirements (Traceability Anchor)

Every downstream component is justified against these. Nothing in this system may contradict them.

1. **Sole source of truth:** the Quran corpus. No internet, web search, Wikipedia, external religious APIs, Hadith, Tafsir, scholarly opinion, or LLM pretraining knowledge may inform an answer.
2. **Verse-level grounding:** each retrievable unit is exactly one ayah. Chunks are never merged across verses.
3. **Cited or silent:** no citations ⇒ no answer. Every claim maps to a retrieved verse.
4. **Refuse over guess:** insufficient evidence ⇒ fixed refusal text, never fabrication.
5. **Trilingual output:** Arabic (primary, always available), English, Urdu for every cited verse, plus Surah name/number and ayah number.
6. **Deterministic pipeline:** the retrieval→generation flow is reproducible and auditable.

---

## 3. High-Level Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                 CLIENTS                      │
                         │   Web UI · Mobile · API consumers            │
                         └───────────────────────┬─────────────────────┘
                                                 │ HTTPS / JSON
                         ┌───────────────────────▼─────────────────────┐
                         │                API GATEWAY (FastAPI)         │
                         │  auth · rate limit · request validation      │
                         └───────────────────────┬─────────────────────┘
                                                 │
         ┌───────────────────────────────────────────────────────────────────────┐
         │                          RAG ORCHESTRATOR                               │
         │                                                                         │
         │  1. Language Detection ──► 2. Query Router ──► 3. Retrieval             │
         │                                    │                │                   │
         │              ┌─────────────────────┼────────────────┼───────────┐      │
         │              │ exact-ref parser     │ semantic       │ keyword   │      │
         │              │ (e.g. 2:255)         │ (dense vec)    │ (sparse)  │      │
         │              └─────────────────────┴────────────────┴───────────┘      │
         │                                    │                                    │
         │  4. Confidence Gate ──► 5. Context Builder ──► 6. Grounded Generation   │
         │                                    │                        │           │
         │  7. Citation & Verse-Existence Validator ──► 8. Response Assembler      │
         └───────────────────────────┬────────────────────────────────┬───────────┘
                                     │                                │
              ┌──────────────────────▼──────┐        ┌────────────────▼──────────┐
              │   QDRANT (vector DB)         │        │   CLOUD LLM (generation)  │
              │   dense + sparse + payload   │        │   strict-grounding prompt │
              └──────────────────────────────┘        └────────────────────────────┘
                                     ▲
              ┌──────────────────────┴──────┐        ┌───────────────────────────┐
              │  LOCAL EMBEDDING SERVICE     │        │  RELATIONAL DB (Postgres) │
              │  multilingual encoder        │        │  users · bookmarks · notes│
              └──────────────────────────────┘        └───────────────────────────┘
```

The generation model sits **behind** the validator, not in front of the user. Its output is untrusted until every citation it produced has been confirmed to (a) exist in the corpus and (b) have actually been part of the retrieved context.

---

## 4. Data Model

### 4.1 Canonical verse schema

Each verse is one document. Field list matches the required dataset exactly, plus derived fields for retrieval and audit.

| Field | Type | Example | Source |
|---|---|---|---|
| `verse_id` | string `S:A` | `112:1` | derived |
| `surah_number` | int (1–114) | `112` | Tanzil |
| `surah_name_ar` | string | `الإخلاص` | Tanzil metadata |
| `surah_name_en` | string | `Al-Ikhlas` | Tanzil metadata |
| `surah_name_ur` | string | `اخلاص` | Tanzil metadata |
| `ayah_number` | int | `1` | Tanzil |
| `text_ar` | string (Uthmani) | `قُلْ هُوَ ٱللَّهُ أَحَدٌ` | Tanzil Uthmani |
| `translation_en` | string | `Say, "He is Allah, [who is] One,"` | Sahih International |
| `translation_ur` | string | `کہو کہ وہ (ذات پاک جس کا نام) اللہ (ہے) ایک ہے` | Jalandhry |
| `revelation_type` | enum `Makki`/`Madani` | `Makki` | Tanzil metadata |
| `juz` | int (1–30) | `30` | Tanzil metadata |
| `hizb` | int (1–60) | `60` | Tanzil metadata |
| `ruku` | int | `558`-scheme | Tanzil metadata |
| `page` | int (1–604) | `604` | Madani Mushaf mapping |
| `word_count` | int | derived from `text_ar` | derived |
| `char_count` | int | derived from `text_ar` | derived |
| `text_ar_normalized` | string | diacritic/tatweel-stripped form | derived (for lexical search) |
| `translation_en_lower` | string | lowercased EN | derived (for lexical search) |
| `translation_ur_normalized` | string | normalized UR | derived (for lexical search) |
| `embedding_source` | string | the concatenated multilingual string that was embedded | derived (audit) |
| `checksum` | string | SHA-256 of Arabic text | derived (integrity) |

**Corpus invariants (validated at ingestion):** 114 surahs · 6,236 ayahs · juz ∈ [1,30] · hizb ∈ [1,60] · page ∈ [1,604] · every `verse_id` unique · every Arabic text non-empty and checksum-verified.

### 4.2 Qdrant collection layout

```
Collection: quran_verses
  Vectors:
    dense:  <D>-dim, cosine        # multilingual semantic vector
    sparse: sparse                  # BM25-style lexical vector (multilingual keyword)
  Payload (indexed):
    verse_id (keyword)  surah_number (int)  ayah_number (int)
    juz (int)  hizb (int)  ruku (int)  page (int)
    revelation_type (keyword)
    surah_name_ar/en/ur (keyword)
    text_ar, translation_en, translation_ur (stored, not filtered)
  Point id: deterministic UUID5(verse_id)   # re-ingest is idempotent
```

Payload indexing on `surah_number`, `juz`, etc. enables filtered retrieval ("verses about mercy **in Makki surahs**") and exact-range lookups without a separate SQL round-trip.

> **Implementation note (v1).** The shipped code uses Qdrant **dense** vectors plus a **lexical-overlap** arm computed in Python over the normalized payload fields, blended into a single calibrated score (see §6.2). This keeps the offline/in-memory path dependency-free while preserving the "lexical must anchor the answer" guarantee. A native sparse vector can be swapped in for high-QPS deployments without changing the grounding contract.

### 4.3 Relational store (future-facing, not on the grounding path)

Postgres for `users`, `bookmarks`, `notes`, `saved_searches`, `translation_registry`. Deliberately isolated from retrieval so that user data can never leak into answer generation.

---

## 5. Ingestion & Indexing Pipeline

Deterministic, idempotent, offline batch job. Run once at build time and on any corpus/version change.

```
Tanzil files (Arabic Uthmani, EN Sahih Intl, UR Jalandhry, metadata)
    │
    ▼  (1) Loader — parse per-source files, key by (surah, ayah)
    ▼  (2) Aligner — join the three languages + structural metadata on verse_id
    ▼  (3) Validator — assert corpus invariants (114/6236/juz/hizb/page/uniqueness)
    ▼  (4) Enricher — normalize Arabic, compute word/char counts, checksums
    ▼  (5) Embedder — build embedding_source string, call LOCAL encoder (batched)
    ▼  (6) Lexical indexer — normalized fields for token-overlap search (per-language)
    ▼  (7) Upserter — idempotent upsert into Qdrant (UUID5 point ids)
    ▼  (8) Post-check — count == 6236, random-sample re-query, checksum audit
```

**Embedding source string.** To make Arabic/English/Urdu versions of the *same* question retrieve the *same* verse, each verse is embedded from a single concatenated multilingual string:

```
[AR] {text_ar}  [EN] {translation_en}  [UR] {translation_ur}
```

This co-locates all three language surfaces of a verse in one vector, so a query in any language lands near it. (Alternative considered: three separate vectors per verse with query-language routing. Rejected for v1 as it triples index size and complicates dedup; the concatenated approach is simpler and, with a strong multilingual encoder, sufficient. Documented as a tunable if evaluation shows cross-lingual recall gaps.)

**No network at ingest** other than the one-time dataset download from Tanzil (CC BY 3.0), which is vendored into the repo so builds are reproducible offline.

---

## 6. Query Pipeline (Request → Answer)

The mandated flow, made concrete:

```
User Question
   │
   ▼ (1) Language Detection      → ar | en | ur  (fastText/CLD3; script heuristics as fallback)
   │
   ▼ (2) Query Router            → classify intent:
   │        • EXACT_REF     e.g. "2:255", "112:1"        → direct payload lookup
   │        • SURAH         e.g. "Surah Maryam"          → surah-name match + list
   │        • KEYWORD       e.g. "mercy", "الرحمن"       → lexical search
   │        • SEMANTIC      e.g. "who created mankind?"  → dense + lexical blend
   │
   ▼ (3) Retrieval               → Qdrant top-k (k≈8, then narrow to top-n≈5)
   │        dense ⊕ lexical-overlap, calibrated blend; optional payload filters
   │
   ▼ (4) Confidence Gate         → if best score < τ  OR  fused evidence too thin → REFUSE
   │
   ▼ (5) Context Builder         → assemble ONLY retrieved verses into a fenced context block
   │        each verse tagged with its verse_id; nothing else is added
   │
   ▼ (6) Grounded Generation     → Cloud LLM with strict-grounding system prompt
   │        temperature 0; output must cite verse_ids present in context
   │
   ▼ (7) Validator               → for every citation the model emitted:
   │        • verse_id exists in corpus?           (verse-existence check)
   │        • verse_id was in the retrieved set?   (no smuggled-in verses)
   │        • answer contains ≥1 valid citation?   (else → REFUSE)
   │        • Arabic text rendered == corpus text? (no invented/altered text)
   │
   ▼ (8) Response Assembler      → Answer + per-verse (AR/EN/UR + Surah/Ayah) + citations
```

### 6.1 Retrieval detail per intent

- **Exact reference** (`2:255`): parsed by regex `^(\d{1,3}):(\d{1,3})$`, validated against corpus bounds, resolved by direct payload lookup — no vector search, no LLM required for the lookup itself (the LLM only phrases the answer from that single verse if the user asked a question).
- **Surah search** ("Surah Maryam" / "سورة مريم" / "سورۃ مریم"): fuzzy match against `surah_name_{ar,en,ur}` (+ transliteration table), returns the surah's ayah list; question-answering within a surah restricts retrieval by `surah_number` filter.
- **Keyword search** (`mercy`, `الرحمن`, `رحمت`): lexical search over the language-appropriate normalized field; returns verses whose translations/text contain the term, ranked by lexical score.
- **Semantic search** ("who created mankind?", "من خلق الإنسان؟", "انسان کو کس نے پیدا کیا؟"): dense retrieval blended with lexical overlap so the three phrasings converge on the same verses (e.g. 15:26, 96:1–2, 55:3–4 as retrieved — not asserted).

### 6.2 Blended semantic ranking

The SEMANTIC score is a **calibrated blend**, not rank-only fusion. Lexical overlap — the fraction of meaningful query tokens (stopwords removed) found in the verse's own-language field — is the primary, well-scaled signal and contributes up to `0.75`. Dense cosine similarity refines ranking and adds recall but is **capped** so that a verse supported by dense similarity *alone* contributes at most `0.25` — below the default confidence-gate threshold `τ = 0.30`.

The consequence is exactly the strict-grounding behavior required: an out-of-scope question, which shares no vocabulary with any verse, cannot clear the gate on a spurious nearest-neighbour and is refused deterministically. An answer must be **lexically anchored** in the verse it cites. (A `reciprocal_rank_fusion` utility is retained in the code for rank-only fusion when a strong native sparse index is available.)

---

## 7. Grounding & Guardrails (the core of the system)

Groundedness is enforced in **four independent layers**, so a failure in any one does not produce an ungrounded answer.

### Layer 1 — Corpus isolation
The generation model receives *only* the fenced context block of retrieved verses. It is given no tools, no web access, no memory, no retrieval of its own. Structurally it cannot reach outside the supplied text.

### Layer 2 — Strict-grounding system prompt
The exact instruction contract sent with every generation call:

> You are a Quran AI Assistant. Use ONLY the Quran verses provided in the CONTEXT block below. Never use prior knowledge. Never use memory. Never use internet knowledge. Never use external religious knowledge (no Hadith, no Tafsir, no scholarly opinion). Never invent verses. Never invent meanings. Never fabricate interpretations. Every sentence of your answer must be supported by a verse in CONTEXT, and you must cite the verse id(s) you used in square brackets, e.g. [2:255]. If CONTEXT does not contain enough information to answer, reply EXACTLY: "I cannot answer this question from the Quran alone because the available Quranic verses do not provide sufficient information." Do not add anything before or after that sentence when refusing.

Temperature is fixed at 0 for determinism.

### Layer 3 — Confidence gate (pre-generation)
If the fused top retrieval score is below threshold τ, or fewer than the minimum number of on-topic verses clear a secondary threshold, the pipeline **short-circuits before calling the LLM** and returns the fixed refusal. This prevents the model from being handed weak context it might "rescue" with pretraining.

### Layer 4 — Post-generation validation
Deterministic code, not the model, has the final say:

1. **Citation presence:** answer must contain ≥1 `[S:A]` citation, else refuse (rule: *no citations = no answer*).
2. **Verse existence:** every cited `verse_id` must exist in the corpus.
3. **Provenance:** every cited `verse_id` must have been in the retrieved context (blocks the model smuggling a memorized verse).
4. **Text fidelity:** any Arabic/translation text the model reproduces is replaced at assembly time by the *canonical* corpus text keyed on `verse_id` — the model never gets to author verse text; it only references ids. This makes invented or altered verse text impossible in the final output.

Only output passing all four is assembled and returned.

### Refusal & scope templates (verbatim)

| Trigger | Response (exact) |
|---|---|
| Retrieved verses insufficient | `I cannot answer this question from the Quran alone because the available Quranic verses do not provide sufficient information.` |
| Low retrieval confidence | `I could not find sufficient Quranic evidence to answer this question.` |
| Question requires Hadith/Tafsir/history/science | `This question cannot be answered from the Quran alone. This assistant is intentionally limited to the Quran as its only knowledge source.` |

---

## 8. Response Format

Every successful answer returns:

```
## Answer
<concise answer, generated ONLY from retrieved verses, with inline [S:A] citations>

## Quran Evidence
  For each supporting verse:
    Surah:  <name_en> (<name_ar> / <name_ur>) — Surah <n>, Ayah <a>
    Arabic (RTL):        <canonical text_ar>
    English (LTR):       <translation_en>
    Urdu (RTL):          <translation_ur>

## Citations
  [S:A] [S:A] ...
```

**JSON contract (API):**

```json
{
  "status": "answered | refused_insufficient | refused_low_confidence | out_of_scope",
  "language": "en",
  "answer": "…text with [15:26] citations…",
  "evidence": [
    {
      "verse_id": "15:26",
      "surah_number": 15,
      "ayah_number": 26,
      "surah_name": {"ar": "الحجر", "en": "Al-Hijr", "ur": "الحجر"},
      "text_ar": "…", "translation_en": "…", "translation_ur": "…",
      "revelation_type": "Makki",
      "score": 0.83
    }
  ],
  "citations": ["15:26"],
  "trace_id": "…"
}
```

Arabic `text_ar` in `evidence` is always populated from the canonical corpus regardless of language mode.

---

## 9. API Surface

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/ask` | Grounded Q&A. Body: `{query, lang?, mode?, filters?}`. Returns the JSON contract above. |
| `GET` | `/v1/verse/{surah}/{ayah}` | Exact verse lookup (AR/EN/UR + metadata). |
| `GET` | `/v1/surah/{id}` | Surah metadata + ayah list. |
| `POST` | `/v1/search` | Raw retrieval (semantic/keyword/hybrid) without generation — returns ranked verses only. |
| `GET` | `/v1/health` | Liveness/readiness, corpus count assertion (== 6236). |

`mode` ∈ `{ar, ar_en, ar_ur, ar_en_ur}` controls which languages render in the response; Arabic is always included. `filters` may constrain by `surah_number`, `juz`, `revelation_type`, etc.

---

## 10. UI Requirements

- **Directionality:** Arabic and Urdu rendered RTL; English LTR. Mixed-direction blocks use explicit `dir` attributes to prevent bleed.
- **Typography:** an Uthmani-capable Arabic font (e.g. a KFGQPC/Uthmanic face) and a Nastaʿlīq-style Urdu font for authenticity.
- **Language toggle:** Arabic-only · Arabic+English · Arabic+Urdu · all three. Arabic never removable.
- **Each verse card:** Surah name + number, ayah number, Arabic (primary, largest), then selected translations, with a copyable `[S:A]` citation.
- **Refusals** are shown plainly, without fabricated filler.

---

## 11. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| API | FastAPI (Python) + Uvicorn/Gunicorn | async, typed, OpenAPI out of the box |
| Orchestration | thin custom pipeline | custom preferred for determinism & auditability |
| Vector DB | Qdrant | dense + payload filters, quantization |
| Embeddings | local multilingual encoder (self-hosted) | reproducible, no content egress at index time |
| Reranker (optional) | cross-encoder, local | precision boost on top-k |
| Generation | Cloud LLM, temperature 0, strict prompt | constrained rewriter only |
| Relational | PostgreSQL | users/bookmarks/notes (off the grounding path) |
| Cache | Redis | query→result cache, rate limiting |
| Container/orchestration | Docker + Compose (→ K8s) | Qdrant, API, embedder, Postgres, Redis |
| Observability | structured logs + OpenTelemetry traces | every answer carries a `trace_id` |

---

## 12. Evaluation & Testing

**Grounding is testable and must be tested.** The shipped harness (`tests/`) covers:

- **Refusal set:** questions with no Quranic answer (modern science specifics, Hadith-dependent rulings, historical minutiae). Expected: exact refusal templates. Metric: refusal precision (must be ~100%).
- **Cross-lingual recall:** the same question in AR/EN/UR must retrieve the same verse set. Metric: Jaccard overlap of top-k across languages.
- **Citation integrity:** fuzz the generator; assert the validator rejects any answer citing a non-retrieved or non-existent verse. Metric: 0 escapes.
- **Text-fidelity:** assert rendered Arabic always equals canonical corpus text (byte-for-byte on the normalized form).
- **Exact-ref correctness:** all 6,236 verse lookups return the right ayah.
- **Retrieval quality:** a labeled topic→verse set (e.g. patience, forgiveness, creation of mankind) scored with recall@k / MRR.

CI gates: corpus invariants, validator escape tests, and refusal-precision must pass before deploy.

---

## 13. Security, Integrity & Operations

- **Corpus integrity:** SHA-256 per verse; a collection-level manifest checksum verified at startup; health check asserts 6,236 verses.
- **Immutability:** the verse corpus is read-only in production; changes go through the ingestion pipeline with a new version tag.
- **No egress on grounding path** beyond the generation call, which carries only retrieved verse text + the user question — never external lookups.
- **Rate limiting & authn** at the gateway; user/bookmark data isolated from retrieval.
- **Auditability:** every `/v1/ask` logs query, detected language, retrieved verse_ids, scores, gate decision, citations, and final status under a `trace_id`.

---

## 14. Scalability & Roadmap

Designed so these can be added **without touching the grounding core**:

- Authenticated users, bookmarks, notes, saved searches (Postgres, already scoped).
- Additional translations (register in `translation_registry`; embeddings extended per language).
- **Optional Tafsir / Hadith datasets** as *separate, explicitly-labeled collections* with their own toggles — never silently merged into Quran answers; any such answer must visibly distinguish Quran evidence from supplementary sources.
- Read-replica Qdrant + quantized vectors for high-QPS scale; sharded by nothing needed (corpus is tiny — 6,236 points — so the bottleneck is generation, not retrieval).

---

## 15. Compliance with the Governing Constraints (Checklist)

| Requirement | How this design satisfies it |
|---|---|
| Quran is the only source of truth | Corpus isolation (L1) + provenance validation (L4) |
| No internet/web/Wikipedia/external religious APIs on answers | Generator has no tools/web; only retrieved verses in context |
| No Hadith/Tafsir/opinion mixed in | Separate future collections, explicitly toggled & labeled; default Quran-only |
| Verse-level chunking, never merged | One ayah = one document = one point; enforced at ingest |
| No citations ⇒ no answer | Validator L4 rule #1 refuses citation-less output |
| Refuse over guess | Confidence gate (L3) + fixed refusal templates |
| Trilingual, Arabic primary | Canonical AR always in evidence; EN/UR per mode |
| Surah/Ayah always referenced | Enforced in response assembler |
| Deterministic RAG | temp 0, calibrated blend, idempotent index, reproducible embeddings |
| No invented verse text/translations | Text authored only from canonical corpus by id (L4 rule #4) |
| Scalable for users/bookmarks/Tafsir later | Isolated Postgres + separate optional collections |

---

## 16. Open Questions for Next Phase

1. Choice of specific local multilingual embedding model (recall vs. size trade-off) — to be selected empirically against the labeled topic set.
2. Ruku numbering scheme (558 Indo-Pak vs. 540) — pick one and store consistently; Jalandhry/South-Asian context suggests 558.
3. Page mapping to a specific Mushaf edition (Madani 604-page standard assumed).
4. Whether to expose raw `/v1/search` publicly or keep it internal for debugging.

---

*Sources for dataset & structural facts: Tanzil Project (CC BY 3.0) for Arabic Uthmani text and the Sahih International / Jalandhry translations; standard Hafs/Madani structural metadata (114 surahs, 6,236 ayahs, 30 juz, 60 hizb, 558 ruku, 604 pages).*

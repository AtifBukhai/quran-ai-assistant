# Quran AI Assistant — Strictly Grounded RAG

A production-grade Quran question-answering system that is **100% grounded in the Quran**.
Every answer is generated **only** from retrieved Quran verses (Arabic Uthmani + English & Urdu
translations). If the retrieved verses do not contain sufficient information, the assistant
declines rather than speculating.

> **Governing principle:** The Quran is the single source of truth. Every answer must be
> traceable to one or more retrieved Quranic verses — or the assistant refuses.

See [`docs/quran-ai-architecture.md`](docs/quran-ai-architecture.md) for the full design.

---

## What "grounded" means here

The generation LLM is used **only as a constrained rewriter** over supplied verses. It has no
web access, no tools, no memory. Around it we run **four independent guardrail layers**:

1. **Corpus isolation** — the model only ever sees retrieved verses.
2. **Strict-grounding prompt** — explicit "use ONLY the context / cite every verse / else refuse".
3. **Confidence gate** — weak retrieval short-circuits to a refusal *before* the LLM is called.
4. **Deterministic validator** — code (not the model) has the final say: every citation must
   exist in the corpus **and** have been retrieved; verse text is re-rendered from the canonical
   corpus so the model can never invent or alter an ayah; no citation ⇒ no answer.

---

## Architecture at a glance

```
User Question
  -> Language Detection (ar|en|ur)
  -> Query Router (EXACT_REF | SURAH | KEYWORD | SEMANTIC)
  -> Retrieval (Qdrant dense + lexical overlap, calibrated blend)
  -> Confidence Gate            (weak -> refuse)
  -> Context Builder            (ONLY retrieved verses)
  -> Grounded Generation        (Cloud LLM, temperature 0, strict prompt)
  -> Validator                  (citations exist? in-context? >=1? text fidelity?)
  -> Response Assembler         (Answer + AR/EN/UR evidence + citations)
```

- **Vector DB:** Qdrant (dense vectors + payload filters)
- **Embeddings:** local multilingual encoder (no verse content egress at index time)
- **Generation:** pluggable cloud LLM behind an interface (`app/llm/base.py`); a deterministic
  offline `EchoGroundedLLM` is included so the whole pipeline runs and is testable with **no
  API key and no network**.
- **Dataset:** Tanzil (CC BY 3.0) — Arabic Uthmani + Sahih International (EN) + Jalandhry (UR).
  A small **sample corpus** ships in `data/sample_corpus.json` so you can run immediately.

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. (optional) Start Qdrant as a server
docker compose up -d qdrant

# 3. Configure
cp .env.example .env        # defaults work offline with the sample corpus

# 4. Ingest the sample corpus
python -m app.ingest.run --source data/sample_corpus.json

# 5. Serve
uvicorn app.main:app --reload
# -> http://localhost:8000/       web UI (text box + Ask button)
# -> http://localhost:8000/docs   interactive API docs
```

### Run fully offline (no Qdrant server, no API key)

The default settings use an **in-memory Qdrant client** and the **EchoGroundedLLM**, so:

```bash
pip install -e ".[dev]"
python -m app.ingest.run --source data/sample_corpus.json   # in-memory store
pytest -q                                                    # grounding tests pass offline
```

### Ask from the command line (no server)

Once a corpus is ingested, query the full pipeline directly:

```bash
python -m app.ask "What does the Quran say about patience?"
python -m app.ask "Recite verse 112:1" --mode ar_en
python -m app.ask --json "who created the heavens"
```

`--mode` chooses which translations to show (`ar`, `ar_en`, `ar_ur`, `ar_en_ur`; Arabic is
always shown), `--lang` forces the query language, and `--json` prints the raw `AskResponse`.
The command exits `0` on a grounded answer and `1` on any refusal, so it is scriptable.

> Note: with the in-memory Qdrant client, each process starts with an empty store — the CLI is
> most useful against a **persisted** Qdrant server (`docker compose up -d qdrant`) so the
> ingested corpus is still there when you ask.

---

## Example

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"query":"Who created mankind?","mode":"ar_en_ur"}' | jq
``````jsonc
{
  "status": "answered",
  "language": "en",
  "answer": "The Quran states that Allah created man from clay ... [15:26]",
  "evidence": [
    {"verse_id":"15:26","surah_number":15,"ayah_number":26,
     "surah_name":{"ar":"الحجر","en":"Al-Hijr","ur":"الحجر"},
     "text_ar":"...","translation_en":"...","translation_ur":"...",
     "revelation_type":"Makki","score":0.83}
  ],
  "citations": ["15:26"],
  "trace_id": "..."
}
```

When evidence is insufficient the `status` becomes `refused_insufficient`,
`refused_low_confidence`, or `out_of_scope`, and the `answer` is the exact mandated template.

---

## Loading the full Quran (6,236 verses)

The sample corpus is a handful of verses for demonstration. To build the full corpus from the
open Tanzil files, see [`data/README.md`](data/README.md) — it documents the exact Tanzil URLs
(Uthmani, Sahih International, Jalandhry, metadata) and how `app/ingest/tanzil.py` merges them
into the canonical schema. The download step is the only time the system touches the network,
and it happens **offline at build time**, never on the answer path.

---

## Project layout

```
app/
  config.py            # settings (env-driven dataclass, minimal .env loader)
  models.py            # canonical Verse + response contract + invariants
  grounding.py         # refusal templates, confidence gate, prompt contract
  embeddings.py        # local multilingual encoder (deterministic fallback)
  vectorstore.py       # Qdrant wrapper (dense + payload filters, in-memory or server)
  retrieval.py         # language detect, router, lexical/dense blended retrieval
  llm/
    base.py            # GroundedLLM interface
    echo.py            # offline deterministic LLM (no network)
    anthropic_llm.py   # cloud adapter (strict grounding), optional
  validator.py         # deterministic post-generation guardrails
  orchestrator.py      # the full pipeline
  main.py              # FastAPI app + endpoints
  ask.py               # command-line query interface (no server)
  ingest/
    run.py             # CLI entry
    tanzil.py          # full-corpus builder from Tanzil files
data/
  sample_corpus.json   # runnable sample verses
  README.md            # how to fetch the full Tanzil corpus
tests/                 # grounding, provenance, refusal, exact-ref, invariants
docs/quran-ai-architecture.md
docker-compose.yml
pyproject.toml
```

## License

Code: MIT. Quran text & translations:Tanzil Project, CC BY 3.0 (attribution required).

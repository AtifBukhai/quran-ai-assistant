# Retrieval Evaluation & Threshold Policy

This directory holds the **evidence** used to make retrieval-tuning decisions. The core rule:

> **No confidence threshold or hybrid weight is relaxed without benchmark evidence that the
> change improves precision/recall.** Grounding and mandatory citations are non-negotiable; they
> are never traded away for recall.

## Files

- `benchmark.json` — labeled question set. Each item is a natural-language query (EN/AR/UR) with
  a set of **gold** `verse_id`s: canonical, widely-cited primary references for that topic. Gold
  sets are deliberately non-exhaustive — they represent the verses a knowledgeable reader would
  expect to see surfaced first.
- `evaluate.py` — runs each query through the retriever and reports Precision@k, Recall@k, and
  MRR. With `--sweep`, it varies the dense (semantic) weight and **reports** the best-performing
  blend. It never edits config.

## How to run

```bash
# Evaluate at the current default weight
python -m eval.evaluate

# Sweep the dense/lexical blend and report the best config
python -m eval.evaluate --sweep
```

Run with the same embedding backend you serve with (the neural backend needs its env vars set):

```bash
# example: evaluate the neural hybrid
QURAN_EMBEDDING_BACKEND=sentence-transformers QURAN_EMBEDDING_DIM=768 python -m eval.evaluate --sweep
```

## Metrics, briefly

- **Precision@k** — of the top-k returned verses, what fraction are gold. High precision = few
  irrelevant verses shown.
- **Recall@k** — of the gold verses, what fraction appear in the top-k. High recall = the right
  verses are being found.
- **MRR** — mean of `1/(rank of first gold verse)`. Rewards putting a correct verse near the top.

## The decision procedure (measure before relaxing)

1. **Baseline.** Run `python -m eval.evaluate` and record the metrics at the current settings.
2. **Propose.** Want to raise `QURAN_DENSE_WEIGHT`, lower `QURAN_MIN_SCORE`, or change BM25 `k1`/
   `b`? Treat it as a hypothesis, not a fact.
3. **Measure.** Run `--sweep` (for weights) or set the candidate value and re-evaluate. Compare
   against the baseline on the **same** benchmark.
4. **Adopt only on evidence.** Change the default **only if** precision/recall/MRR improve (or
   recall improves at no meaningful precision cost). If metrics regress, keep the current value.
5. **Guardrail stays.** Even a better-scoring blend must keep the dense arm capped so a pure
   vector match cannot clear the confidence gate alone (see `app/config.py`). Semantic results are
   *candidates*; the lexical/concept anchor + citation validator decide what becomes evidence.

## Why the default dense weight is capped

With `min_score = 0.30` and dense weight `0.25`, a verse's semantic-only contribution is at most
`0.25 × 1.0 = 0.25 < 0.30` — below the gate. So neural similarity widens and re-ranks the
candidate pool, but a verse with no shared or concept-expanded query term is still refused. This
is the mechanism that keeps "hybrid retrieval" from quietly becoming "semantic answers." Raising
the weight past `~0.30` breaks that property and must be justified by the benchmark **and** a
conscious decision to change the grounding posture — not done casually.

## Extending the benchmark

More labeled items = a more trustworthy signal. Add questions with gold verses you are confident
about (primary, well-known references). Aim for topical and linguistic spread (EN/AR/UR, legal /
narrative / ethical / theological). Keep gold sets tight and canonical rather than exhaustive.

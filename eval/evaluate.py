"""Retrieval evaluation and hybrid-weight sweep.

Loads the labeled benchmark (eval/benchmark.json), runs each query through the retriever,
computes Precision@k, Recall@k, and MRR (mean reciprocal rank), and sweeps the lexical/dense
weight blend to REPORT the best-performing config. This is evidence for threshold decisions —
the script never auto-changes min_score or dense_weight; those stay gated by manual review.

Usage:
    python -m eval.evaluate --sweep     # sweep dense_weight, print best config
    python -m eval.evaluate             # evaluate at current default only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.embeddings import build_embedder
from app.retrieval import Retriever, detect_language, route
from app.vectorstore import VectorStore


def load_benchmark(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"]


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of top-k results that are gold (relevant)."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for vid in top_k if vid in gold)
    return hits / k


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold verses found in top-k."""
    if not gold:
        return 1.0  # vacuous: no gold to miss
    top_k = retrieved[:k]
    hits = sum(1 for vid in top_k if vid in gold)
    return hits / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    """Mean reciprocal rank: 1 / (rank of first gold verse), or 0 if none found."""
    for rank, vid in enumerate(retrieved, start=1):
        if vid in gold:
            return 1.0 / rank
    return 0.0


def evaluate(
    benchmark: list[dict],
    retriever: Retriever,
    top_k: int = 10,
) -> dict[str, float]:
    """Run all benchmark queries and compute aggregate metrics."""
    p_at_1_all, p_at_5_all, p_at_10_all = [], [], []
    r_at_1_all, r_at_5_all, r_at_10_all = [], [], []
    mrr_all = []

    for item in benchmark:
        query = item["query"]
        lang = item.get("lang") or detect_language(query)
        gold = set(item["gold"])

        routed = route(query, lang)
        results = retriever.semantic(routed.raw, routed.language, limit=top_k, filters=None)
        retrieved_ids = [pl["verse_id"] for _, pl in results if pl.get("verse_id")]

        p_at_1_all.append(precision_at_k(retrieved_ids, gold, k=1))
        p_at_5_all.append(precision_at_k(retrieved_ids, gold, k=5))
        p_at_10_all.append(precision_at_k(retrieved_ids, gold, k=10))
        r_at_1_all.append(recall_at_k(retrieved_ids, gold, k=1))
        r_at_5_all.append(recall_at_k(retrieved_ids, gold, k=5))
        r_at_10_all.append(recall_at_k(retrieved_ids, gold, k=10))
        mrr_all.append(mrr(retrieved_ids, gold))

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "P@1": avg(p_at_1_all),
        "P@5": avg(p_at_5_all),
        "P@10": avg(p_at_10_all),
        "R@1": avg(r_at_1_all),
        "R@5": avg(r_at_5_all),
        "R@10": avg(r_at_10_all),
        "MRR": avg(mrr_all),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and sweep weights.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep dense_weight and report the best config (vs. evaluate at default only).",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("eval/benchmark.json"),
        help="Path to benchmark JSON (default: eval/benchmark.json).",
    )
    args = parser.parse_args(argv)

    if not args.benchmark.exists():
        print(f"error: benchmark not found: {args.benchmark}", file=sys.stderr)
        return 2

    benchmark = load_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark)} benchmark items from {args.benchmark}")

    settings = get_settings()
    store = VectorStore(
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    store.ensure()
    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )

    if not args.sweep:
        # Single-point evaluation at the current default.
        retriever = Retriever(store, embedder, dense_weight=settings.dense_weight)
        metrics = evaluate(benchmark, retriever, top_k=settings.top_k)
        print(f"\nEvaluation at dense_weight={settings.dense_weight:.2f}:")
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")
        return 0

    # Sweep dense_weight from 0.1 to 0.5 in 0.05 steps.
    print("\nSweeping dense_weight (lexical + dense = 1.0)...")
    print("=" * 70)
    candidates = []
    for dw in [i * 0.05 for i in range(2, 11)]:  # 0.10, 0.15, ..., 0.50
        retriever = Retriever(store, embedder, dense_weight=dw)
        metrics = evaluate(benchmark, retriever, top_k=settings.top_k)
        # Composite score: weight P@5 and MRR equally as a proxy for "useful top results."
        # Adjust weights here if you care more about precision or recall.
        composite = 0.5 * metrics["P@5"] + 0.5 * metrics["MRR"]
        candidates.append((dw, composite, metrics))
        print(
            f"dense={dw:.2f} | P@5={metrics['P@5']:.4f} R@5={metrics['R@5']:.4f} MRR={metrics['MRR']:.4f} | composite={composite:.4f}"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_dw, best_score, best_metrics = candidates[0]

    print("=" * 70)
    print(f"\nBest config: dense_weight={best_dw:.2f} (composite score {best_score:.4f})")
    print("Full metrics at best config:")
    for name, val in best_metrics.items():
        print(f"  {name}: {val:.4f}")
    print(
        f"\nTo adopt: set QURAN_DENSE_WEIGHT={best_dw:.2f} in .env and re-run evaluation to confirm."
    )
    print("(Remember: any threshold change requires benchmark evidence — see eval/README.md.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

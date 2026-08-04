"""Ingestion CLI.

Reads a JSON corpus (list of verse objects matching the canonical schema), validates it with the
``Verse`` model, embeds each verse from its multilingual ``embedding_source()``, and upserts into
Qdrant. Idempotent: re-running upserts the same point ids.

Usage:
    python -m app.ingest.run --source data/sample_corpus.json
    python -m app.ingest.run --source data/full_corpus.json --recreate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import get_settings
from ..embeddings import build_embedder
from ..models import TOTAL_AYAHS, Verse
from ..vectorstore import VectorStore


def load_corpus(path: Path) -> list[Verse]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Corpus file must be a JSON array of verse objects.")
    verses = [Verse.model_validate(obj) for obj in raw]
    return verses


def validate_invariants(verses: list[Verse], *, full: bool) -> None:
    ids = [v.verse_id for v in verses]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise ValueError(f"Duplicate verse ids: {sorted(dupes)[:10]}")
    for v in verses:
        if v.checksum == "":
            raise ValueError(f"Empty checksum for {v.verse_id}")
    if full and len(verses) != TOTAL_AYAHS:
        raise ValueError(
            f"Full corpus must contain {TOTAL_AYAHS} verses, got {len(verses)}."
        )


def to_payload(v: Verse) -> dict:
    return {
        "verse_id": v.verse_id,
        "surah_number": v.surah_number,
        "ayah_number": v.ayah_number,
        "surah_name": v.surah_name.model_dump(),
        "text_ar": v.text_ar,
        "translation_en": v.translation_en,
        "translation_ur": v.translation_ur,
        "revelation_type": v.revelation_type.value,
        "juz": v.juz,
        "hizb": v.hizb,
        "ruku": v.ruku,
        "page": v.page,
        "word_count": v.word_count,
        "char_count": v.char_count,
        "text_ar_normalized": v.text_ar_normalized,
        "translation_en_lower": v.translation_en_lower,
        "translation_ur_normalized": v.translation_ur_normalized,
        "checksum": v.checksum,
    }


def ingest(source: Path, *, recreate: bool, full: bool) -> int:
    settings = get_settings()
    verses = load_corpus(source)
    validate_invariants(verses, full=full)

    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    store = VectorStore(
        collection=settings.qdrant_collection,
        dim=embedder.dim,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    if recreate:
        store.recreate()
    else:
        store.ensure()

    vectors = embedder.encode([v.embedding_source() for v in verses])
    records = [
        (v.verse_id, vec, to_payload(v)) for v, vec in zip(verses, vectors)
    ]
    n = store.upsert(records)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a Quran corpus into Qdrant.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and recreate the collection first."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Assert the corpus contains all 6236 verses.",
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        return 2

    n = ingest(args.source, recreate=args.recreate, full=args.full)
    print(f"Ingested {n} verses into collection '{get_settings().qdrant_collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

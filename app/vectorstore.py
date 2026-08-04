"""Qdrant vector store wrapper.

Supports both an in-memory client (``qdrant_url`` empty — no server, great for dev/tests) and a
real Qdrant server/cloud. Stores a dense vector per verse plus the full verse payload so
retrieval, exact lookup, and filtering all hit one store.

Point ids are ``UUID5(verse_id)`` so re-ingesting is idempotent (upsert, never duplicate).
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Fixed namespace so UUID5(verse_id) is stable across runs (idempotent upserts).
_NAMESPACE = uuid.UUID("6ba7b814-9dad-11d1-80b4-00c04fd430c8")


def point_id(verse_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, verse_id))


class VectorStore:
    def __init__(
        self,
        collection: str,
        dim: int,
        url: str = "",
        api_key: str = "",
    ) -> None:
        self.collection = collection
        self.dim = dim
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        else:
            # In-memory: no server required.
            self.client = QdrantClient(location=":memory:")

    # --- lifecycle ------------------------------------------------------------
    def recreate(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        # Payload indexes for fast exact/filtered lookups.
        for field, schema in [
            ("verse_id", qm.PayloadSchemaType.KEYWORD),
            ("surah_number", qm.PayloadSchemaType.INTEGER),
            ("ayah_number", qm.PayloadSchemaType.INTEGER),
            ("juz", qm.PayloadSchemaType.INTEGER),
            ("hizb", qm.PayloadSchemaType.INTEGER),
            ("page", qm.PayloadSchemaType.INTEGER),
            ("revelation_type", qm.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(self.collection, field, schema)
            except Exception:  # noqa: BLE001 - index may already exist
                pass

    def ensure(self) -> None:
        try:
            self.client.get_collection(self.collection)
        except Exception:  # noqa: BLE001
            self.recreate()

    # --- writes ---------------------------------------------------------------
    def upsert(self, records: Iterable[tuple[str, list[float], dict[str, Any]]]) -> int:
        points = [
            qm.PointStruct(id=point_id(vid), vector=vec, payload=payload)
            for vid, vec, payload in records
        ]
        if not points:
            return 0
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # --- reads ----------------------------------------------------------------
    def search(
        self,
        vector: list[float],
        limit: int,
        query_filter: Optional[qm.Filter] = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [(h.score, h.payload or {}) for h in hits]

    def get_by_verse_id(self, verse_id: str) -> Optional[dict[str, Any]]:
        res = self.client.retrieve(
            collection_name=self.collection,
            ids=[point_id(verse_id)],
            with_payload=True,
        )
        if not res:
            return None
        return res[0].payload or None

    def scroll_filter(
        self, query_filter: qm.Filter, limit: int = 1000
    ) -> list[dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [p.payload or {} for p in points]


def build_filter(filters: Optional[dict]) -> Optional[qm.Filter]:
    """Translate a simple ``{field: value}`` dict into a Qdrant filter."""
    if not filters:
        return None
    must = []
    for key, value in filters.items():
        must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    return qm.Filter(must=must) if must else None

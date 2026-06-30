"""Motor model for the rag_index_state collection.

Tracks which content chunks have been upserted into Qdrant and stores their
content_hash for change detection — unchanged chunks are skipped on re-runs.

Schema (rag_index_state):
  source_type  str   — server|channel|thread|reel|user|message|notification
  source_id    str   — MongoDB _id as string
  chunk_index  int   — 0-based position within a multi-chunk document
  content_hash str   — SHA-1 of the chunk text
  vector_id    str   — Qdrant point UUID (deterministic)
  collection   str   — content_public | content_private
  status       str   — ok | error
  embedded_at  date  — timestamp of the last successful upsert

Unique index: (source_type, source_id, chunk_index).
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.mongo import get_db

_COLLECTION = "rag_index_state"


def _coll():
    return get_db()[_COLLECTION]


async def ensure_indexes() -> None:
    """Create required indexes (idempotent — safe to call before every ingest run)."""
    coll = _coll()
    await coll.create_index(
        [("source_type", 1), ("source_id", 1), ("chunk_index", 1)],
        unique=True,
        name="source_chunk_unique",
    )
    await coll.create_index([("embedded_at", -1)], name="embedded_at_desc")


async def get_existing_hashes(
    source_type: str, source_ids: list[str]
) -> dict[tuple[str, int], str]:
    """Return {(source_id, chunk_index): content_hash} for the given source ids.

    Used to skip chunks whose text has not changed since the last ingest run.
    """
    cursor = _coll().find(
        {"source_type": source_type, "source_id": {"$in": source_ids}},
        {"source_id": 1, "chunk_index": 1, "content_hash": 1, "_id": 0},
    )
    result: dict[tuple[str, int], str] = {}
    async for doc in cursor:
        result[(doc["source_id"], doc["chunk_index"])] = doc["content_hash"]
    return result


async def upsert_index_entry(
    *,
    source_type: str,
    source_id: str,
    chunk_index: int,
    content_hash: str,
    vector_id: str,
    collection: str,
) -> None:
    """Record a successful Qdrant upsert (or update the hash if content changed)."""
    await _coll().update_one(
        {
            "source_type": source_type,
            "source_id": source_id,
            "chunk_index": chunk_index,
        },
        {
            "$set": {
                "content_hash": content_hash,
                "vector_id": vector_id,
                "collection": collection,
                "status": "ok",
                "embedded_at": datetime.now(tz=timezone.utc),
            }
        },
        upsert=True,
    )

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.rag import chunker
from app.rag.chunker import RagChunk
from app.rag.embedder import embed_batch
from app.rag.qdrant_client import bootstrap_collections, upsert_points
from app.rag.rag_document_index import (
    ensure_indexes,
    get_existing_hashes,
    upsert_index_entry,
)

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.ingest_sync")

_EMBED_BATCH = settings.embedding_batch_size


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    source_type: str
    mongo_total: int = 0  
    docs: int = 0         
    chunks: int = 0        
    embeddings: int = 0  
    points: int = 0        
    upserted: int = 0      
    skipped: int = 0       
    errors: int = 0

    def log_summary(self) -> None:
        logger.info(
            "SOURCE=%s MONGO_TOTAL=%d DOCS=%d CHUNKS=%d EMBEDDINGS=%d "
            "POINTS=%d UPSERTED=%d SKIPPED=%d ERRORS=%d",
            self.source_type, self.mongo_total, self.docs, self.chunks,
            self.embeddings, self.points, self.upserted, self.skipped, self.errors,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _point_id(source_id: str, chunk_index: int) -> str:
    """Deterministic UUID-v5 so the same chunk always maps to the same Qdrant point."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{source_id}:{chunk_index}"))


def _qdrant_collection(visibility: str) -> str:
    return (
        settings.qdrant_collection_public
        if visibility == "public"
        else settings.qdrant_collection_private
    )


async def _bootstrap() -> None:
    """Idempotent: create Qdrant collections + rag_index_state indexes if absent."""
    await bootstrap_collections()
    await ensure_indexes()


async def _embed_and_upsert(chunks: list[RagChunk], result: IngestResult) -> None:
    """Embed `chunks` in _EMBED_BATCH-sized batches and upsert each to Qdrant."""
    for start in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[start : start + _EMBED_BATCH]
        texts = [c.text for c in batch]
        try:
            vectors = await embed_batch(texts)
        except Exception:
            logger.exception("embed_batch failed for %s (%d chunk(s))",
                             result.source_type, len(batch))
            result.errors += len(batch)
            continue
        valid = [(c, v) for c, v in zip(batch, vectors) if v]
        dropped = len(batch) - len(valid)
        if dropped:
            logger.warning("dropped %d %s chunk(s) with empty embedding vectors",
                           dropped, result.source_type)
            result.errors += dropped
        result.embeddings += len(valid)

        by_coll: dict[str, list[tuple[RagChunk, list[float]]]] = {}
        for chunk, vector in valid:
            coll = _qdrant_collection(chunk.payload.visibility)
            by_coll.setdefault(coll, []).append((chunk, vector))

        for coll_name, items in by_coll.items():
            points = [
                {
                    "id": _point_id(c.payload.source_id, c.chunk_index),
                    "vector": v,
                    "payload": {**c.payload.to_dict(), "excerpt": c.text},
                }
                for c, v in items
            ]
            result.points += len(points)
            try:
                await upsert_points(coll_name, points)
            except Exception:
                logger.exception("upsert_points failed for %s (%d point(s))",
                                 coll_name, len(points))
                result.errors += len(points)
                continue

            for (chunk, _), point in zip(items, points):
                try:
                    await upsert_index_entry(
                        source_type=chunk.payload.source_type,
                        source_id=chunk.payload.source_id,
                        chunk_index=chunk.chunk_index,
                        content_hash=chunk.content_hash,
                        vector_id=point["id"],
                        collection=coll_name,
                    )
                    result.upserted += 1
                except Exception:
                    logger.exception("rag_index_state upsert failed for %s/%s",
                                     chunk.payload.source_type, chunk.payload.source_id)
                    result.errors += 1


def _since_filter(since: datetime | None, field: str = "updatedAt") -> dict[str, Any]:
    return {field: {"$gt": since}} if since else {}


def _dedup(chunks: list[RagChunk], existing: dict[tuple[str, int], str], result: IngestResult) -> list[RagChunk]:
    """Return only chunks whose content_hash differs from the stored value."""
    to_embed: list[RagChunk] = []
    for chunk in chunks:
        key = (chunk.payload.source_id, chunk.chunk_index)
        if existing.get(key) == chunk.content_hash:
            result.skipped += 1
        else:
            to_embed.append(chunk)
    return to_embed


# ── Server metadata helper ─────────────────────────────────────────────────────

async def _fetch_server_meta(server_oids: list[Any]) -> dict[str, dict]:
    """Batch-fetch {str(server_id): {visibility, member_ids}} for the given OIDs."""
    if not server_oids:
        return {}
    db = get_db()
    docs = await db["discordservers"].find(
        {"_id": {"$in": server_oids}},
        {"visibility": 1, "members": 1},
    ).to_list(length=len(server_oids))
    meta: dict[str, dict] = {}
    for s in docs:
        member_ids = [
            str(m["user"]) for m in s.get("members", []) if m.get("user")
        ]
        meta[str(s["_id"])] = {
            "visibility": s.get("visibility", "private"),
            "member_ids": member_ids,
        }
    return meta


# ── Ingest functions ───────────────────────────────────────────────────────────

async def ingest_servers(
    since: datetime | None = None, limit: int = 100
) -> IngestResult:
    result = IngestResult(source_type="server")
    await _bootstrap()
    db = get_db()
    docs = await db["discordservers"].find(
        _since_filter(since)
    ).limit(limit).to_list(length=limit)
    if not docs:
        return result

    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("server", source_ids)

    raw_chunks = [c for d in docs for c in chunker.normalize_server(d)]
    result.docs = len(docs)
    result.chunks = len(raw_chunks)
    to_embed = _dedup(raw_chunks, existing, result)
    if to_embed:
        await _embed_and_upsert(to_embed, result)
    return result


async def ingest_channels(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    result = IngestResult(source_type="channel")
    await _bootstrap()
    db = get_db()
    docs = await db["channels"].find(
        _since_filter(since)
    ).limit(limit).to_list(length=limit)
    if not docs:
        return result

    server_meta = await _fetch_server_meta(
        list({d["server"] for d in docs if d.get("server")})
    )
    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("channel", source_ids)

    raw_chunks: list[RagChunk] = []
    for doc in docs:
        meta = server_meta.get(str(doc.get("server", "")), {"visibility": "private", "member_ids": []})
        raw_chunks.extend(chunker.normalize_channel(doc, meta["visibility"], meta["member_ids"]))

    result.docs = len(docs)
    result.chunks = len(raw_chunks)
    to_embed = _dedup(raw_chunks, existing, result)
    if to_embed:
        await _embed_and_upsert(to_embed, result)
    return result


async def ingest_threads(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    result = IngestResult(source_type="thread")
    await _bootstrap()
    db = get_db()
    docs = await db["threads"].find(
        _since_filter(since)
    ).limit(limit).to_list(length=limit)
    if not docs:
        return result

    server_meta = await _fetch_server_meta(
        list({d["server"] for d in docs if d.get("server")})
    )
    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("thread", source_ids)

    raw_chunks: list[RagChunk] = []
    for doc in docs:
        meta = server_meta.get(str(doc.get("server", "")), {"visibility": "private", "member_ids": []})
        raw_chunks.extend(chunker.normalize_thread(doc, meta["visibility"], meta["member_ids"]))

    result.docs = len(docs)
    result.chunks = len(raw_chunks)
    to_embed = _dedup(raw_chunks, existing, result)
    if to_embed:
        await _embed_and_upsert(to_embed, result)
    return result


async def ingest_reels(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    result = IngestResult(source_type="reel")
    await _bootstrap()
    db = get_db()
    src = "reel"

    result.mongo_total = await db["reels"].count_documents({})
    logger.info("SOURCE=%s MONGO_TOTAL=%d", src, result.mongo_total)

    skip_deleted = await db["reels"].count_documents({"isDeleted": True})
    skip_not_public = await db["reels"].count_documents(
        {"isDeleted": {"$ne": True}, "isPublic": {"$ne": True}}
    )
    logger.info(
        "SOURCE=%s SKIP_REASONS isDeleted=%d not_public=%d",
        src, skip_deleted, skip_not_public,
    )

    query: dict[str, Any] = {"isDeleted": {"$ne": True}, "isPublic": True}
    if since:
        query["updated_at"] = {"$gt": since}

    docs = await db["reels"].find(query).limit(limit).to_list(length=limit)
    result.docs = len(docs)
    logger.info("SOURCE=%s FILTERED=%d", src, result.docs)

    if not docs:
        logger.info("SOURCE=%s SKIPPED=all reason=no_documents_passed_filter", src)
        return result

    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("reel", source_ids)

    raw_chunks = [c for d in docs for c in chunker.normalize_reel(d)]
    result.chunks = len(raw_chunks)
    logger.info("SOURCE=%s CHUNKS=%d", src, result.chunks)

    to_embed = _dedup(raw_chunks, existing, result)
    logger.info("SOURCE=%s DEDUP=%d new_or_changed skipped=%d unchanged",
                src, len(to_embed), result.skipped)

    if to_embed:
        await _embed_and_upsert(to_embed, result)

    logger.info("SOURCE=%s EMBED=%d SKIPPED=%d", src, result.embeddings, result.skipped)
    return result


async def ingest_users(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    result = IngestResult(source_type="user")
    await _bootstrap()
    db = get_db()
    src = "user"

    result.mongo_total = await db["users"].count_documents({})
    logger.info("SOURCE=%s MONGO_TOTAL=%d", src, result.mongo_total)

    skip_private = await db["users"].count_documents(
        {"settings.privacy.profileVisibility": {"$in": ["private", "friends"]}}
    )
    no_setting = await db["users"].count_documents(
        {"settings.privacy.profileVisibility": {"$exists": False}}
    )
    explicit_public = await db["users"].count_documents(
        {"settings.privacy.profileVisibility": "public"}
    )
    logger.info(
        "SOURCE=%s SKIP_REASONS explicit_private_or_friends=%d "
        "no_visibility_setting=%d explicit_public=%d",
        src, skip_private, no_setting, explicit_public,
    )
    query: dict[str, Any] = {
        "settings.privacy.profileVisibility": {"$nin": ["private", "friends"]}
    }
    if since:
        query["updatedAt"] = {"$gt": since}

    docs = await db["users"].find(query).limit(limit).to_list(length=limit)
    result.docs = len(docs)
    logger.info("SOURCE=%s FILTERED=%d", src, result.docs)

    if not docs:
        logger.info("SOURCE=%s SKIPPED=all reason=no_documents_passed_filter", src)
        return result

    n_explicit = sum(
        1 for d in docs
        if (d.get("settings") or {}).get("privacy", {}).get("profileVisibility") == "public"
    )
    n_fallback = result.docs - n_explicit
    if n_fallback:
        logger.debug(
            "SOURCE=%s FILTERED_DETAIL explicit_public=%d fallback_no_setting=%d",
            src, n_explicit, n_fallback,
        )

    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("user", source_ids)

    raw_chunks = [c for d in docs for c in chunker.normalize_user(d)]
    result.chunks = len(raw_chunks)
    logger.info("SOURCE=%s CHUNKS=%d", src, result.chunks)

    to_embed = _dedup(raw_chunks, existing, result)
    logger.info("SOURCE=%s DEDUP=%d new_or_changed skipped=%d unchanged",
                src, len(to_embed), result.skipped)

    if to_embed:
        await _embed_and_upsert(to_embed, result)

    logger.info("SOURCE=%s EMBED=%d SKIPPED=%d", src, result.embeddings, result.skipped)
    return result


async def ingest_messages(
    since: datetime | None = None, limit: int = 1000
) -> IngestResult:
    result = IngestResult(source_type="message")
    await _bootstrap()
    db = get_db()

    base_query: dict[str, Any] = {"deletedForEveryone": {"$ne": True}}
    if since:
        base_query["updatedAt"] = {"$gt": since}
    docs = await db["messages"].find(base_query).limit(limit).to_list(length=limit)
    if not docs:
        return result

    channel_docs = [d for d in docs if d.get("server") and d.get("channel")]
    dm_docs = [d for d in docs if d.get("conversationId") and not d.get("server")]
    group_docs = [d for d in docs if d.get("groupId") and not d.get("server")]

    server_meta = await _fetch_server_meta(
        list({d["server"] for d in channel_docs if d.get("server")})
    )

    conv_ids = list({d["conversationId"] for d in dm_docs if d.get("conversationId")})
    conv_meta: dict[str, list[str]] = {}
    if conv_ids:
        conv_docs = await db["conversations"].find(
            {"_id": {"$in": conv_ids}}, {"participants": 1}
        ).to_list(length=len(conv_ids))
        for c in conv_docs:
            conv_meta[str(c["_id"])] = [str(p) for p in c.get("participants", [])]

    group_ids = list({d["groupId"] for d in group_docs if d.get("groupId")})
    group_meta: dict[str, list[str]] = {}
    if group_ids:
        grp_docs = await db["groups"].find(
            {"_id": {"$in": group_ids}}, {"participants": 1}
        ).to_list(length=len(group_ids))
        for g in grp_docs:
            group_meta[str(g["_id"])] = [str(p) for p in g.get("participants", [])]

    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("message", source_ids)

    raw_chunks: list[RagChunk] = []
    for doc in channel_docs:
        meta = server_meta.get(str(doc.get("server", "")), {"visibility": "private", "member_ids": []})
        raw_chunks.extend(chunker.normalize_channel_message(doc, meta["visibility"], meta["member_ids"]))
    for doc in dm_docs:
        participants = conv_meta.get(str(doc.get("conversationId", "")), [])
        raw_chunks.extend(chunker.normalize_dm_message(doc, participants))
    for doc in group_docs:
        participants = group_meta.get(str(doc.get("groupId", "")), [])
        raw_chunks.extend(chunker.normalize_group_message(doc, participants))

    result.docs = len(docs)
    result.chunks = len(raw_chunks)
    to_embed = _dedup(raw_chunks, existing, result)
    if to_embed:
        await _embed_and_upsert(to_embed, result)
    return result


async def ingest_notifications(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    result = IngestResult(source_type="notification")
    await _bootstrap()
    db = get_db()
    query: dict[str, Any] = {"read": False}
    if since:
        query["updatedAt"] = {"$gt": since}
    docs = await db["notifications"].find(query).limit(limit).to_list(length=limit)
    if not docs:
        return result

    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("notification", source_ids)

    raw_chunks = [c for d in docs for c in chunker.normalize_notification(d)]
    result.docs = len(docs)
    result.chunks = len(raw_chunks)
    to_embed = _dedup(raw_chunks, existing, result)
    if to_embed:
        await _embed_and_upsert(to_embed, result)
    return result


async def ingest_all(since: datetime | None = None) -> dict[str, Any]:
    logger.info("ingest_all started (since=%s)", since)

    results: list[dict[str, Any]] = []

    for fn in (
        ingest_servers,
        ingest_channels,
        ingest_threads,
        ingest_reels,
        ingest_users,
        ingest_messages,
        ingest_notifications,
    ):
        try:
            result = await fn(since=since)
            result.log_summary()
            results.append({
                "source": result.source_type,
                "processed": result.upserted,
                "docs": result.docs,
                "chunks": result.chunks,
                "embeddings": result.embeddings,
                "points": result.points,
                "skipped": result.skipped,
                "errors": result.errors,
            })
        except Exception:
            logger.exception("ingest source %s failed", fn.__name__)
            results.append({
                "source": fn.__name__.removeprefix("ingest_"),
                "processed": 0,
                "error": "see logs",
            })

    total = sum(r.get("processed", 0) for r in results)
    logger.info("ingest_all complete: %d point(s) upserted across %d source(s)",
                total, len(results))
    return {"status": "success", "results": results}

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


async def _fetch_channel_names(channel_oids: list[Any]) -> dict[str, str]:
    """Batch-fetch {str(channel_id): name} for the given OIDs (display-title only)."""
    if not channel_oids:
        return {}
    db = get_db()
    docs = await db["channels"].find(
        {"_id": {"$in": channel_oids}}, {"name": 1},
    ).to_list(length=len(channel_oids))
    return {str(d["_id"]): d.get("name", "") for d in docs}


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
    # Must match chunker.normalize_user's rule EXACTLY (profileVisibility ==
    # "public"). This used to be a $nin, which in MongoDB also matches
    # documents where the field is absent — so every user without an explicit
    # setting was fetched, counted in FILTERED as though ingested, then
    # silently discarded by the chunker, producing zero chunks. Matching
    # strictly is also the privacy-safe direction: never index a profile that
    # did not explicitly opt in to being public.
    query: dict[str, Any] = {"settings.privacy.profileVisibility": "public"}
    if since:
        query["updatedAt"] = {"$gt": since}

    docs = await db["users"].find(query).limit(limit).to_list(length=limit)
    result.docs = len(docs)
    logger.info("SOURCE=%s FILTERED=%d", src, result.docs)

    if not docs:
        logger.info("SOURCE=%s SKIPPED=all reason=no_documents_passed_filter", src)
        return result

    # (The old "fallback_no_setting" diagnostic lived here. It counted docs the
    # query let through but the chunker then dropped — impossible now that both
    # match on profileVisibility == "public", so the branch could never fire.)

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


_HELP_DIR = Path(__file__).resolve().parents[2] / "content" / "help"


async def ingest_docs() -> IngestResult:
    """Index the product-help corpus (content/help/*.md).

    Unlike every other source these are files, not Mongo documents, so this is
    also the only ingest that can be run without a database. It is called on
    startup (app.main) rather than only from the batch pass, because product
    help is small, static, and must never be missing — a user asking "how do I
    create a community?" should not depend on someone having remembered to
    trigger an ingest.

    Dedup is skipped for the same reason as ingest_one: the hash table records
    what was embedded once, not what currently exists in Qdrant. Embeddings are
    content-hash cached in Redis, so re-running this is nearly free.
    """
    result = IngestResult(source_type="docs")

    if not _HELP_DIR.is_dir():
        logger.warning("help corpus missing at %s — product help disabled", _HELP_DIR)
        return result

    await _bootstrap()

    chunks: list[RagChunk] = []
    for path in sorted(_HELP_DIR.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            logger.exception("could not read help topic %s", path.name)
            continue
        if not raw:
            continue

        # First "# Heading" is the topic title; the rest is the body.
        lines = raw.splitlines()
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title = path.stem.replace("-", " ").title()
            body = raw

        result.docs += 1
        chunks.extend(chunker.normalize_doc(path.stem, title, body))

    result.chunks = len(chunks)
    if chunks:
        await _embed_and_upsert(chunks, result)
    logger.info(
        "SOURCE=docs topics=%d chunks=%d embedded=%d",
        result.docs, result.chunks, result.embeddings,
    )
    return result


# Targeted single-document ingest: source_type → (collection, needs server ACL).
# Deliberately excludes `message`: chat is high-volume and indexing every send
# would embed on the hot path. Messages stay on the batch pass until that is
# designed with debouncing/batching.
_ONE_DOC_SOURCES: dict[str, tuple[str, bool]] = {
    "server": ("discordservers", False),
    "channel": ("channels", True),
    "thread": ("threads", True),
    "reel": ("reels", False),
    "user": ("users", False),
}


async def ingest_one(source_type: str, source_id: str) -> IngestResult:
    """Index ONE document immediately, by id.

    Exists because nothing ever triggered a full ingest: `/internal/v1/ingest`
    had no caller and no scheduler, so newly created communities, channels,
    threads, reels and profiles were never searchable — RAG returned nothing
    and the domain-locked prompt answered with a refusal. The gateway now
    calls this fire-and-forget after a write, mirroring how it already calls
    delete-sync after a delete.

    Deliberately does NOT consult the content-hash dedup table, for the same
    reason ingest_voice_session doesn't: that table records "this text was
    embedded once", not "a vector currently exists in Qdrant". The two drift
    apart — bulk delete paths (delete_by_server/channel/conversation/group)
    intentionally skip hash cleanup, and any Qdrant rebuild loses vectors while
    leaving every hash row behind. A write-triggered re-index that consulted it
    would then report success and index nothing, forever, for exactly the
    documents most likely to need it. Verified live: a public server with a
    surviving hash row and no vector was skipped on every attempt.

    Re-embedding one document is cheap — embeddings are content-hash cached in
    Redis, so unchanged text is a cache hit — and upserts are idempotent
    (point ids are deterministic UUID-v5 of source_id:chunk_index).
    """
    result = IngestResult(source_type=source_type)

    entry = _ONE_DOC_SOURCES.get(source_type)
    if entry is None:
        logger.warning("ingest_one: unsupported source_type=%s", source_type)
        return result

    collection, needs_server_meta = entry
    # Lazy import: app.tools imports nothing from app.rag, so this direction is
    # safe, but keep it local rather than widening module-level coupling.
    from app.tools._common import to_oid

    oid = to_oid(source_id)
    if oid is None:
        logger.warning("ingest_one: invalid source_id=%s", source_id)
        return result

    await _bootstrap()
    db = get_db()
    doc = await db[collection].find_one({"_id": oid})
    if not doc:
        # Already deleted, or never existed — delete-sync owns that path.
        return result

    if needs_server_meta:
        meta_map = await _fetch_server_meta([doc["server"]] if doc.get("server") else [])
        meta = meta_map.get(
            str(doc.get("server", "")), {"visibility": "private", "member_ids": []}
        )
        normalize = (
            chunker.normalize_channel if source_type == "channel" else chunker.normalize_thread
        )
        raw_chunks = normalize(doc, meta["visibility"], meta["member_ids"])
    else:
        normalize = {
            "server": chunker.normalize_server,
            "reel": chunker.normalize_reel,
            "user": chunker.normalize_user,
        }[source_type]
        raw_chunks = normalize(doc)

    result.docs = 1
    result.chunks = len(raw_chunks)

    # No chunks is a legitimate outcome, not an error: the chunker enforces its
    # own visibility rules (a private profile, an unpublished reel), so this is
    # how "indexed nothing, on purpose" looks.
    if not raw_chunks:
        return result

    await _embed_and_upsert(raw_chunks, result)
    return result


async def ingest_voice_session(
    doc: dict,
    server_visibility: str,
    server_member_ids: list[str],
    channel_name: str = "",
) -> IngestResult:
    """Immediate, single-document ingestion for one just-summarized voice
    session — called right after /internal/v1/voice/summarize generates its
    summary (app.api.internal), NOT part of ingest_all()'s periodic batch
    scan, so a fresh summary is searchable the moment it exists rather than
    waiting for the next scheduled/triggered ingest pass. No content-hash
    dedup check (unlike every ingest_* below) — a session is summarized
    exactly once, so there is no "unchanged since last ingest" case to skip.
    `channel_name` is optional (only affects the chunk's display title) — the
    gateway already has it on hand from the session's own channel document.
    """
    result = IngestResult(source_type="voice_session")
    await _bootstrap()
    chunks = chunker.normalize_voice_session(
        doc, server_visibility, server_member_ids, channel_name
    )
    result.docs = 1
    result.chunks = len(chunks)
    if chunks:
        await _embed_and_upsert(chunks, result)
    return result


async def ingest_voice_sessions(
    since: datetime | None = None, limit: int = 500
) -> IngestResult:
    """Periodic/batch re-ingestion path (ingest_all()) — a resilience net in
    case Qdrant data is ever lost/rebuilt; the primary path for a fresh
    summary is ingest_voice_session()'s immediate call, above."""
    result = IngestResult(source_type="voice_session")
    await _bootstrap()
    db = get_db()

    query: dict[str, Any] = {"status": "summarized"}
    if since:
        query["endedAt"] = {"$gt": since}
    docs = await db["voicesessions"].find(query).limit(limit).to_list(length=limit)
    if not docs:
        return result

    server_meta = await _fetch_server_meta(
        list({d["server"] for d in docs if d.get("server")})
    )
    channel_names = await _fetch_channel_names(
        list({d["channel"] for d in docs if d.get("channel")})
    )
    source_ids = [str(d["_id"]) for d in docs]
    existing = await get_existing_hashes("voice_session", source_ids)

    raw_chunks: list[RagChunk] = []
    for doc in docs:
        meta = server_meta.get(str(doc.get("server", "")), {"visibility": "private", "member_ids": []})
        ch_name = channel_names.get(str(doc.get("channel", "")), "")
        raw_chunks.extend(
            chunker.normalize_voice_session(doc, meta["visibility"], meta["member_ids"], ch_name)
        )

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
        ingest_voice_sessions,
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

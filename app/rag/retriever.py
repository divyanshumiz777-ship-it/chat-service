from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.rag.embedder import embed_text
from app.rag.qdrant_client import get_client

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.retriever")

# ── Re-rank weights (must sum to 1.0) ─────────────────────────────────────────
_W_SIMILARITY = 0.70
_W_RECENCY = 0.20
_W_POPULARITY = 0.10

_DEFAULT_TOP_K = 10
_OVERRETRIEVE_FACTOR = 3  # fetch 3× top_k from Qdrant, rerank, then slice to top_k
# Hard cap on offset+top_k passed to Qdrant — bounds the per-collection fetch
# cost of a "Load more" pagination chain regardless of how deep a caller pages.
_MAX_EFFECTIVE_K = 100


# ── Source-type filtering ───────────────────────────────────────────────────
_CANONICAL_TYPES = frozenset({
    "server", "channel", "thread", "reel", "user", "message", "notification",
    "voice_session",
    # Product-help topics (content/help/*.md). Always public, no owner — see
    # chunker.normalize_doc.
    "docs",
})
_TYPE_ALIASES: dict[str, list[str]] = {
    "discordserver": ["server"],
    "server": ["server"],
    "servers": ["server"],
    "community": ["server", "channel", "thread"],
    "communities": ["server", "channel", "thread"],
    "channel": ["channel"],
    "channels": ["channel"],
    "group": ["message"],
    "groups": ["message"],
    "thread": ["thread"],
    "threads": ["thread"],
    "creator": ["user"],
    "creators": ["user"],
    "user": ["user"],
    "users": ["user"],
    "reel": ["reel"],
    "reels": ["reel"],
    "message": ["message"],
    "messages": ["message"],
    "notification": ["notification"],
    "notifications": ["notification"],
    "voice_session": ["voice_session"],
    "voice_sessions": ["voice_session"],
    "voice": ["voice_session"],
    "call": ["voice_session"],
    "calls": ["voice_session"],
    "docs": ["docs"],
    "doc": ["docs"],
    "help": ["docs"],
    "guide": ["docs"],
}


def _resolve_source_types(types: list[str] | None) -> set[str]:
    if not types:
        return set()
    resolved: set[str] = set()
    for raw in types:
        key = str(raw).strip().lower()
        if not key:
            continue
        mapped = _TYPE_ALIASES.get(key)
        if mapped:
            resolved.update(mapped)
        elif key in _CANONICAL_TYPES:
            resolved.add(key)
        else:
            logger.warning("ignoring unknown search source_type: %r", raw)
    return resolved


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:

    id: str           
    source_type: str 
    source_id: str  
    title: str       
    score: float      
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _recency_score(updated_at: str) -> float:
    """Exponential decay with ~43-day half-life. Returns 0.5 when unknown."""
    if not updated_at:
        return 0.5
    try:
        dt = datetime.fromisoformat(updated_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - dt).total_seconds() / 86400
        return math.exp(-age_days / 43.0)
    except (ValueError, TypeError, OverflowError):
        return 0.5


def _popularity_score(payload: dict) -> float:
    return 0.5


def _rerank_score(similarity: float, payload: dict) -> float:
    return (
        _W_SIMILARITY * similarity
        + _W_RECENCY * _recency_score(payload.get("updated_at", ""))
        + _W_POPULARITY * _popularity_score(payload)
    )


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def _build_acl_filter(
    user_id: str | None,
    collection: str,
    source_types: set[str] | None = None,
) -> Any:
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must: list[Any] = []
    if collection == settings.qdrant_collection_public:
        must.append(FieldCondition(key="visibility", match=MatchValue(value="public")))
    else:
        if not user_id:
            raise ValueError("user_id is required for private collection search")
        must.append(
            FieldCondition(key="allowed_user_ids", match=MatchAny(any=[user_id]))
        )

    if source_types:
        must.append(
            FieldCondition(key="source_type", match=MatchAny(any=sorted(source_types)))
        )

    return Filter(must=must)


async def _query_collection(
    collection: str,
    query_vector: list[float],
    acl_filter: Any,
    limit: int,
) -> list[Any]:
    try:
        client = get_client()
        return await client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=acl_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("qdrant search failed for %s: %s", collection, exc)
        return []


def _to_result(point: Any) -> SearchResult:
    payload = point.payload or {}
    title = payload.get("title") or (
        f"{payload.get('source_type', '')} {str(payload.get('source_id', ''))[:8]}"
    )
    score = _rerank_score(float(point.score), payload)
    metadata = {
        k: v for k, v in payload.items()
        if k != "allowed_user_ids"  
    }
    return SearchResult(
        id=str(point.id),
        source_type=payload.get("source_type", ""),
        source_id=payload.get("source_id", ""),
        title=title,
        score=score,
        metadata=metadata,
    )


def _dedup_by_source(results: list[SearchResult]) -> list[SearchResult]:
    best: dict[str, SearchResult] = {}
    for r in results:
        if r.source_id not in best or r.score > best[r.source_id].score:
            best[r.source_id] = r
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


# ── Public API ─────────────────────────────────────────────────────────────────

async def search_public(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    types: list[str] | None = None,
) -> list[SearchResult]:
    source_types = _resolve_source_types(types)
    query_vector = await embed_text(query)
    raw = await _query_collection(
        settings.qdrant_collection_public,
        query_vector,
        _build_acl_filter(None, settings.qdrant_collection_public, source_types),
        limit=top_k * _OVERRETRIEVE_FACTOR,
    )
    return _dedup_by_source([_to_result(p) for p in raw])[:top_k]


async def search_private(
    user_id: str,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    types: list[str] | None = None,
) -> list[SearchResult]:
    source_types = _resolve_source_types(types)
    query_vector = await embed_text(query)
    raw = await _query_collection(
        settings.qdrant_collection_private,
        query_vector,
        _build_acl_filter(user_id, settings.qdrant_collection_private, source_types),
        limit=top_k * _OVERRETRIEVE_FACTOR,
    )
    return _dedup_by_source([_to_result(p) for p in raw])[:top_k]


async def _drop_blocked(user_id: str, results: list[SearchResult]) -> list[SearchResult]:
    """Remove results authored by someone on either side of a block.

    Qdrant's own ACL filter (visibility + allowed_user_ids) is baked at ingest
    time and knows nothing about blocks, so without this a blocked user's
    public profile, reels and public-server messages still surfaced in /search
    and in RAG answers — even though every MongoDB-backed tool filters blocks.

    Bidirectional, matching the pattern in tools/catch_me_up.py: hide content
    from people the caller blocked AND from people who blocked the caller.
    Fails CLOSED on error (drops nothing only when there is nothing to check).
    """
    if not user_id or not results:
        return results

    owner_ids = {
        str(r.metadata.get("owner_id"))
        for r in results
        if r.metadata.get("owner_id")
    }
    if not owner_ids:
        return results

    try:
        from app.core.mongo import get_db
        from app.tools._common import oid_set, to_oid

        db = get_db()
        user_oid = to_oid(user_id)
        if user_oid is None:
            return results

        owner_oids = list(oid_set(owner_ids))
        if not owner_oids:
            return results

        me, reverse_blocked_docs = await asyncio.gather(
            db["users"].find_one({"_id": user_oid}, {"blockedUsers": 1}),
            db["users"]
            .find({"_id": {"$in": owner_oids}, "blockedUsers": user_oid}, {"_id": 1})
            .to_list(length=None),
        )

        hidden = oid_set((me or {}).get("blockedUsers"))
        hidden |= {d["_id"] for d in reverse_blocked_docs}
        if not hidden:
            return results

        hidden_str = {str(o) for o in hidden}
        return [
            r for r in results
            if str(r.metadata.get("owner_id") or "") not in hidden_str
        ]
    except Exception:
        # A blocklist lookup failure must not silently widen visibility.
        logger.exception("block filter failed user=%s — dropping owned results", user_id)
        return [r for r in results if not r.metadata.get("owner_id")]


_INACTIVE_ACCOUNT_STATUSES = ["deactivated", "deleted"]


def _needs_access_check(r: SearchResult) -> bool:
    md = r.metadata
    return bool(
        md.get("server_id")
        or md.get("conversation_id")
        or md.get("group_id")
        or r.source_type in ("user", "reel")
    )


async def _drop_inaccessible(user_id: str, results: list[SearchResult]) -> list[SearchResult]:
    """Re-check every result against CURRENT MongoDB state.

    Qdrant's ACL (visibility + allowed_user_ids) is a snapshot taken when the
    chunk was indexed, and re-indexing skips unchanged text — so it never sees
    a member leaving or being banned from a private server, a server switching
    from public to private, someone leaving a group, a profile going private,
    or an account being deactivated/deleted. Without this, all of those stayed
    searchable through the assistant indefinitely.

    Rules (mirroring the backend's own access checks):
      server_id        → server exists AND (public OR caller owns it OR caller
                         has a non-banned ServerMember row — the collection the
                         backend's join-server/canJoinVoiceChannel trust)
      conversation_id  → caller is in conversations.participants
      group_id         → caller is in groups.participants
      source_type user → profile is public AND account is active
      source_type reel → reel exists, not deleted, not private

    One batched query per collection. FAILS CLOSED: if any lookup errors,
    every result that needed a check is dropped.
    """
    if not results:
        return results
    to_check = [r for r in results if _needs_access_check(r)]
    if not to_check:
        return results

    try:
        from app.core.mongo import get_db
        from app.tools._common import oid_set, to_oid

        user_oid = to_oid(user_id)
        if user_oid is None:
            return [r for r in results if not _needs_access_check(r)]

        def ids(key: str) -> set:
            return oid_set(r.metadata.get(key) for r in to_check)

        server_oids = ids("server_id")
        conv_oids = ids("conversation_id")
        group_oids = ids("group_id")
        profile_oids = oid_set(r.source_id for r in to_check if r.source_type == "user")
        reel_oids = oid_set(r.source_id for r in to_check if r.source_type == "reel")

        db = get_db()

        async def visible_servers() -> set[str]:
            if not server_oids:
                return set()
            servers = await db["discordservers"].find(
                {"_id": {"$in": list(server_oids)}}, {"visibility": 1, "owner": 1}
            ).to_list(length=None)
            ok = {
                str(s["_id"]) for s in servers
                if s.get("visibility") == "public" or s.get("owner") == user_oid
            }
            restricted = [s["_id"] for s in servers if str(s["_id"]) not in ok]
            if restricted:
                rows = await db["servermembers"].find(
                    {
                        "server": {"$in": restricted},
                        "user": user_oid,
                        "banned.isBanned": {"$ne": True},
                    },
                    {"server": 1},
                ).to_list(length=None)
                ok |= {str(m["server"]) for m in rows}
            return ok

        async def matching(collection: str, oids: set, extra: dict) -> set[str]:
            if not oids:
                return set()
            docs = await db[collection].find(
                {"_id": {"$in": list(oids)}, **extra}, {"_id": 1}
            ).to_list(length=None)
            return {str(d["_id"]) for d in docs}

        ok_servers, ok_convs, ok_groups, ok_profiles, ok_reels = await asyncio.gather(
            visible_servers(),
            matching("conversations", conv_oids, {"participants": user_oid}),
            matching("groups", group_oids, {"participants": user_oid}),
            matching(
                "users",
                profile_oids,
                {
                    "settings.privacy.profileVisibility": "public",
                    "accountStatus": {"$nin": _INACTIVE_ACCOUNT_STATUSES},
                },
            ),
            matching("reels", reel_oids, {"isDeleted": {"$ne": True}, "isPublic": {"$ne": False}}),
        )
    except Exception:
        logger.exception("access re-check failed user=%s — dropping checked results", user_id)
        return [r for r in results if not _needs_access_check(r)]

    def allowed(r: SearchResult) -> bool:
        md = r.metadata
        if md.get("server_id") and str(md["server_id"]) not in ok_servers:
            return False
        if md.get("conversation_id") and str(md["conversation_id"]) not in ok_convs:
            return False
        if md.get("group_id") and str(md["group_id"]) not in ok_groups:
            return False
        if r.source_type == "user" and str(r.source_id) not in ok_profiles:
            return False
        if r.source_type == "reel" and str(r.source_id) not in ok_reels:
            return False
        return True

    return [r for r in results if allowed(r)]


async def search_all(
    user_id: str,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    types: list[str] | None = None,
    offset: int = 0,
) -> list[SearchResult]:
    """`offset` paginates the FINAL deduped+reranked list, not Qdrant's own
    native result ordering — Qdrant's search() call is pre-rerank, so pushing
    offset down to it would paginate the wrong ordering (and risks skipping
    or duplicating items once merged/deduped/reranked here). Instead this
    over-fetches `offset + top_k` (still scaled by _OVERRETRIEVE_FACTOR) from
    each collection, reranks/dedupes the full set as usual, then slices the
    requested page out of that final list. _MAX_EFFECTIVE_K bounds how deep
    a pagination chain can push the underlying Qdrant fetch.
    """
    source_types = _resolve_source_types(types)
    query_vector = await embed_text(query)
    effective_k = min(offset + top_k, _MAX_EFFECTIVE_K)
    per_coll = effective_k * _OVERRETRIEVE_FACTOR

    pub_raw, priv_raw = await asyncio.gather(
        _query_collection(
            settings.qdrant_collection_public,
            query_vector,
            _build_acl_filter(None, settings.qdrant_collection_public, source_types),
            limit=per_coll,
        ),
        _query_collection(
            settings.qdrant_collection_private,
            query_vector,
            _build_acl_filter(user_id, settings.qdrant_collection_private, source_types),
            limit=per_coll,
        ),
    )

    merged = [_to_result(p) for p in pub_raw + priv_raw]
    deduped = _dedup_by_source(merged)
    visible = await _drop_blocked(user_id, deduped)
    visible = await _drop_inaccessible(user_id, visible)
    return visible[offset:offset + top_k]

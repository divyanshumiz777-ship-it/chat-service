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


# ── Source-type filtering ───────────────────────────────────────────────────
_CANONICAL_TYPES = frozenset({
    "server", "channel", "thread", "reel", "user", "message", "notification",
    "voice_session",
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


async def search_all(
    user_id: str,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    types: list[str] | None = None,
) -> list[SearchResult]:
    source_types = _resolve_source_types(types)
    query_vector = await embed_text(query)
    per_coll = top_k * _OVERRETRIEVE_FACTOR

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
    return _dedup_by_source(merged)[:top_k]

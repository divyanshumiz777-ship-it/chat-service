from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.qdrant")

_client: Any | None = None


def get_client() -> Any:
    global _client
    if _client is None:
        from qdrant_client import AsyncQdrantClient

        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


def _distance() -> Any:
    from qdrant_client.models import Distance

    return getattr(Distance, settings.qdrant_distance.upper(), Distance.COSINE)


async def collection_exists(name: str) -> bool:
    return bool(await get_client().collection_exists(name))


# Payload keys that retriever._build_acl_filter filters on. Qdrant REQUIRES a
# payload index for every filtered key — without one it rejects the search with
# HTTP 400 ("Index required but not found for ..."), which _query_collection
# catches and turns into an empty list. The observable symptom is that
# semantic search silently returns nothing, always, for everyone: the two ACL
# keys below are on the critical path of every single query.
_FILTERED_PAYLOAD_KEYS = ("visibility", "allowed_user_ids", "source_type")


async def _ensure_payload_indexes(client: Any, collection: str) -> None:
    """Create the payload indexes the ACL filter depends on (idempotent)."""
    from qdrant_client.models import PayloadSchemaType

    for key in _FILTERED_PAYLOAD_KEYS:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info("qdrant payload index ready: %s.%s", collection, key)
        except Exception as exc:
            # Already-exists is the common, benign case; anything else is worth
            # seeing, but must not stop startup — the service is still useful
            # for every non-RAG capability.
            logger.debug("payload index %s.%s not created: %s", collection, key, exc)


async def bootstrap_collections() -> None:
    from qdrant_client.models import VectorParams

    client = get_client()
    vectors_config = VectorParams(
        size=settings.qdrant_vector_size, distance=_distance()
    )
    for name in (
        settings.qdrant_collection_public,
        settings.qdrant_collection_private,
    ):
        if await client.collection_exists(name):
            logger.debug("qdrant collection %s already exists", name)
        else:
            await client.create_collection(
                collection_name=name, vectors_config=vectors_config
            )
            logger.info("created qdrant collection %s", name)

        # Deliberately OUTSIDE the exists check: the collections in this
        # deployment already existed but had no payload indexes, so an
        # early-continue meant they were never backfilled and every search
        # kept 400ing. Creating an existing index is a no-op.
        await _ensure_payload_indexes(client, name)


async def get_qdrant_status() -> str:
    if not settings.qdrant_url:
        return "not_wired"
    try:
        client = get_client()
    except ImportError:
        return "not_wired"
    except Exception:
        return "unhealthy"
    try:
        await client.get_collections()
        return "healthy"
    except Exception:
        return "unhealthy"


async def upsert_points(collection_name: str, points: list[dict]) -> None:
    from qdrant_client.models import PointStruct

    client = get_client()
    qdrant_points = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points
    ]
    await client.upsert(collection_name=collection_name, points=qdrant_points)
    logger.debug("upserted %d point(s) into %s", len(qdrant_points), collection_name)


async def delete_by_fields(collection_name: str, conditions: dict[str, str]) -> None:
    """Delete every point in `collection_name` whose payload matches ALL of
    `conditions` (field -> exact value, ANDed).

    Refuses an empty/blank-valued conditions dict rather than silently
    building a match-everything filter — Qdrant's delete-by-filter has no
    separate "are you sure" step, so a bug that produced e.g. {"server_id":
    ""} must fail loudly here instead of wiping the whole collection.
    """
    if not conditions or not all(conditions.values()):
        raise ValueError("delete_by_fields requires non-empty field=value conditions")

    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    client = get_client()
    filter_ = Filter(
        must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in conditions.items()]
    )
    await client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=filter_),
    )
    logger.debug("deleted points matching %s from %s", conditions, collection_name)


async def close_client() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            logger.exception("error closing qdrant client")
        _client = None

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
            continue
        await client.create_collection(
            collection_name=name, vectors_config=vectors_config
        )
        logger.info("created qdrant collection %s", name)


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


async def close_client() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            logger.exception("error closing qdrant client")
        _client = None

"""Async Redis client for the AI Chat Service.

Lazy singleton over redis.asyncio, sharing the backend's Redis (via REDIS_URL).
No connection is opened at import time. Used by the readiness probe now; caching
is added in later batches.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return a lazily-created shared async Redis client."""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _client


async def check_health() -> bool:
    """Readiness: ping Redis. Returns True if reachable, else False (never raises)."""
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


async def close_redis() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

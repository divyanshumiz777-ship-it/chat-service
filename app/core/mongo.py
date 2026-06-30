"""Async MongoDB client for the AI Chat Service.

Lazy singleton over motor with connection pooling, sharing the same MongoDB as
the backend (via MONGO_URI). No connection is opened at import time. Used by the
readiness probe now; RAG data access is added in later batches.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.mongo")

# Matches the backend's mongoose default: when MONGO_URI carries no database in
# its path and MONGO_DB is unset, the data lives in "test".
_DEFAULT_DB = "test"

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return a lazily-created, pooled async Mongo client."""
    global _client
    if _client is None:
        if not settings.mongo_uri:
            raise RuntimeError("MONGO_URI is not configured")
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            maxPoolSize=settings.mongo_max_pool_size,
            minPoolSize=settings.mongo_min_pool_size,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the target database.

    Resolution order:
      1. MONGO_DB setting, when set (explicit override).
      2. The default database encoded in MONGO_URI's path.
      3. Fallback to "test" — the backend uses mongoose, which defaults to the
         "test" database when the connection string has no database in its path.

    Previously, an SRV URI with no database path (…mongodb.net/?retryWrites=…)
    made get_default_database() raise ConfigurationError, which silently aborted
    every ingest run before any document was read.
    """
    from pymongo.errors import ConfigurationError

    client = get_client()
    if settings.mongo_db:
        return client[settings.mongo_db]
    try:
        return client.get_default_database()
    except ConfigurationError:
        logger.warning(
            "MONGO_URI has no default database and MONGO_DB is unset; "
            "falling back to %r (set MONGO_DB to silence this).",
            _DEFAULT_DB,
        )
        return client[_DEFAULT_DB]


async def check_health() -> bool:
    """Readiness: ping MongoDB. Returns True if reachable, else False (never raises)."""
    if not settings.mongo_uri:
        return False
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False


async def close_mongo() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None

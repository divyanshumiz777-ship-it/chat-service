from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from app.config import get_settings
from app.core.redis import get_redis

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.embedder")

_CACHE_PREFIX = "chat:embed:"
_model: Any | None = None


def get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def reset_model() -> None:
    global _model
    _model = None


def _cache_key(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


async def _cache_get_many(keys: list[str]) -> list[list[float] | None]:
    if not keys:
        return []
    try:
        raw = await get_redis().mget(keys)
    except Exception:
        return [None] * len(keys)

    out: list[list[float] | None] = []
    for value in raw:
        if value is None:
            out.append(None)
            continue
        try:
            out.append(json.loads(value))
        except (ValueError, TypeError):
            out.append(None)
    return out


async def _cache_set(key: str, vector: list[float]) -> None:
    try:
        await get_redis().set(
            key, json.dumps(vector), ex=settings.embedding_cache_ttl
        )
    except Exception:
        logger.debug("embedding cache write failed for %s", key)


def _encode(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


async def embed_text(text: str) -> list[float]:
    """Embed a single string (cache-aware)."""
    result = await embed_batch([text])
    return result[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    keys = [_cache_key(t) for t in texts]
    cached = await _cache_get_many(keys)

    missing_indices = [i for i, vector in enumerate(cached) if vector is None]
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        encoded = await asyncio.to_thread(_encode, missing_texts)
        for idx, vector in zip(missing_indices, encoded):
            cached[idx] = vector
            await _cache_set(keys[idx], vector)

    return [vector if vector is not None else [] for vector in cached]

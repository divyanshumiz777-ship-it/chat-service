"""Unit tests for the embedding engine's cache-aware batching.

Redis and the model are both mocked, so these run without a live Redis or the
sentence-transformers package.
"""

import json

from app.rag import embedder


def test_cache_key_is_deterministic_and_prefixed():
    assert embedder._cache_key("hello") == embedder._cache_key("hello")
    assert embedder._cache_key("hello").startswith("chat:embed:")
    assert embedder._cache_key("hello") != embedder._cache_key("world")


async def test_embed_batch_uses_cache_and_encodes_only_misses(monkeypatch):
    cached_vec = [0.1, 0.2]
    store = {embedder._cache_key("a"): json.dumps(cached_vec)}
    writes: dict[str, tuple] = {}

    class _FakeRedis:
        async def mget(self, keys):
            return [store.get(k) for k in keys]

        async def set(self, key, value, ex=None):
            writes[key] = (value, ex)

    encode_calls: list[list[str]] = []

    def _fake_encode(texts):
        encode_calls.append(list(texts))
        return [[9.0, 9.0] for _ in texts]

    monkeypatch.setattr(embedder, "get_redis", lambda: _FakeRedis())
    monkeypatch.setattr(embedder, "_encode", _fake_encode)

    result = await embedder.embed_batch(["a", "b"])

    assert result[0] == cached_vec        # served from cache
    assert result[1] == [9.0, 9.0]        # encoded miss
    assert encode_calls == [["b"]]        # only the miss was encoded
    # the miss was written back with the configured TTL
    b_key = embedder._cache_key("b")
    assert b_key in writes
    assert writes[b_key][1] == embedder.settings.embedding_cache_ttl


async def test_embed_text_returns_single_vector(monkeypatch):
    class _FakeRedis:
        async def mget(self, keys):
            return [None for _ in keys]

        async def set(self, key, value, ex=None):
            pass

    monkeypatch.setattr(embedder, "get_redis", lambda: _FakeRedis())
    monkeypatch.setattr(embedder, "_encode", lambda texts: [[1.0, 2.0, 3.0]])

    assert await embedder.embed_text("hi") == [1.0, 2.0, 3.0]


async def test_embed_batch_empty():
    assert await embedder.embed_batch([]) == []

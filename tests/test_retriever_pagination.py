"""Unit tests for search_all's offset/top_k pagination.

Qdrant and the embedding model are both mocked — these verify the
over-fetch-then-slice logic in isolation: offset paginates the FINAL
deduped+reranked list, not Qdrant's own native ordering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag import retriever


class _FakePoint:
    def __init__(self, source_id: str, score: float):
        self.id = f"point-{source_id}"
        self.score = score
        self.payload = {
            "source_type": "message",
            "source_id": source_id,
            "title": f"Message {source_id}",
            "updated_at": "",
        }


@pytest.fixture(autouse=True)
def _mock_embed(monkeypatch):
    async def _fake_embed(_query: str) -> list[float]:
        return [0.0]

    monkeypatch.setattr(retriever, "embed_text", _fake_embed)


def _install_fake_points(monkeypatch, points: list[_FakePoint]):
    """All requested points come back from the public collection; private
    returns empty — the pagination logic under test doesn't care which
    collection they came from, just the final merged/deduped/sliced order."""

    async def _fake_query_collection(collection, query_vector, acl_filter, limit):
        if collection == retriever.settings.qdrant_collection_public:
            return points[:limit]
        return []

    monkeypatch.setattr(retriever, "_query_collection", _fake_query_collection)


@pytest.mark.asyncio
async def test_offset_zero_returns_first_page_highest_scores_first(monkeypatch):
    # 5 distinct sources, descending score order 0.9 .. 0.5
    points = [_FakePoint(str(i), score) for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])]
    _install_fake_points(monkeypatch, points)

    results = await retriever.search_all(user_id="u1", query="q", top_k=2, offset=0)

    assert [r.source_id for r in results] == ["0", "1"]


@pytest.mark.asyncio
async def test_offset_pages_into_the_next_slice_of_the_reranked_list(monkeypatch):
    points = [_FakePoint(str(i), score) for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])]
    _install_fake_points(monkeypatch, points)

    results = await retriever.search_all(user_id="u1", query="q", top_k=2, offset=2)

    assert [r.source_id for r in results] == ["2", "3"]


@pytest.mark.asyncio
async def test_offset_past_the_end_returns_empty_not_an_error(monkeypatch):
    points = [_FakePoint(str(i), score) for i, score in enumerate([0.9, 0.8, 0.7])]
    _install_fake_points(monkeypatch, points)

    results = await retriever.search_all(user_id="u1", query="q", top_k=2, offset=10)

    assert results == []


@pytest.mark.asyncio
async def test_per_collection_fetch_scales_with_offset_plus_top_k(monkeypatch):
    captured_limits: list[int] = []

    async def _capture_limit(collection, query_vector, acl_filter, limit):
        captured_limits.append(limit)
        return []

    monkeypatch.setattr(retriever, "_query_collection", _capture_limit)

    await retriever.search_all(user_id="u1", query="q", top_k=10, offset=20)

    # effective_k = min(20 + 10, _MAX_EFFECTIVE_K) = 30; per_coll = 30 * _OVERRETRIEVE_FACTOR
    expected = min(30, retriever._MAX_EFFECTIVE_K) * retriever._OVERRETRIEVE_FACTOR
    assert all(limit == expected for limit in captured_limits)


@pytest.mark.asyncio
async def test_per_collection_fetch_is_capped_by_max_effective_k(monkeypatch):
    captured_limits: list[int] = []

    async def _capture_limit(collection, query_vector, acl_filter, limit):
        captured_limits.append(limit)
        return []

    monkeypatch.setattr(retriever, "_query_collection", _capture_limit)

    await retriever.search_all(user_id="u1", query="q", top_k=50, offset=200)

    expected = retriever._MAX_EFFECTIVE_K * retriever._OVERRETRIEVE_FACTOR
    assert all(limit == expected for limit in captured_limits)

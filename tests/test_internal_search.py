"""Integration tests: POST /internal/v1/search — limit/offset validation and
the has_more heuristic. search_all() itself is mocked; these verify the
HTTP-layer contract only.

Run:
    pytest services/chat-service/tests/test_internal_search.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import require_internal_token
from app.rag.retriever import SearchResult


@pytest.fixture()
def authed_client() -> TestClient:
    async def _always_ok() -> None:
        return

    app.dependency_overrides[require_internal_token] = _always_ok
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.pop(require_internal_token, None)


def _fake_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(id=f"p{i}", source_type="message", source_id=str(i), title=f"m{i}", score=1.0)
        for i in range(n)
    ]


def test_defaults_limit_10_offset_0(authed_client: TestClient) -> None:
    with patch("app.api.internal.search_all", new=AsyncMock(return_value=_fake_results(3))) as mock_search:
        resp = authed_client.post(
            "/internal/v1/search",
            json={"query": "hello"},
            headers={"X-User-Id": "u1"},
        )
    assert resp.status_code == 200, resp.text
    mock_search.assert_awaited_once()
    _, kwargs = mock_search.await_args
    assert kwargs["top_k"] == 10
    assert kwargs["offset"] == 0


def test_custom_limit_and_offset_forwarded(authed_client: TestClient) -> None:
    with patch("app.api.internal.search_all", new=AsyncMock(return_value=_fake_results(5))):
        resp = authed_client.post(
            "/internal/v1/search",
            json={"query": "hello", "limit": 5, "offset": 20},
            headers={"X-User-Id": "u1"},
        )
    assert resp.status_code == 200, resp.text


def test_has_more_true_when_full_page_returned(authed_client: TestClient) -> None:
    with patch("app.api.internal.search_all", new=AsyncMock(return_value=_fake_results(5))):
        resp = authed_client.post(
            "/internal/v1/search",
            json={"query": "hello", "limit": 5},
            headers={"X-User-Id": "u1"},
        )
    assert resp.json()["has_more"] is True


def test_has_more_false_when_partial_page_returned(authed_client: TestClient) -> None:
    with patch("app.api.internal.search_all", new=AsyncMock(return_value=_fake_results(3))):
        resp = authed_client.post(
            "/internal/v1/search",
            json={"query": "hello", "limit": 5},
            headers={"X-User-Id": "u1"},
        )
    assert resp.json()["has_more"] is False


def test_limit_above_50_is_rejected(authed_client: TestClient) -> None:
    resp = authed_client.post(
        "/internal/v1/search",
        json={"query": "hello", "limit": 500},
        headers={"X-User-Id": "u1"},
    )
    assert resp.status_code == 422


def test_negative_offset_is_rejected(authed_client: TestClient) -> None:
    resp = authed_client.post(
        "/internal/v1/search",
        json={"query": "hello", "offset": -1},
        headers={"X-User-Id": "u1"},
    )
    assert resp.status_code == 422

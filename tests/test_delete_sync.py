"""Tests for the Qdrant deletion/tombstone path — specifically the two new
bulk scopes (delete_by_conversation, delete_by_group) added to close the
DM/group-DM tombstoning gap, plus the /internal/v1/delete endpoint's
handling of the two new request scopes.

Run:
    pytest services/chat-service/tests/test_delete_sync.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import require_internal_token
from app.rag import delete_sync

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# delete_sync.delete_by_conversation / delete_by_group — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_by_conversation_filters_on_conversation_id() -> None:
    with patch.object(delete_sync, "delete_by_fields", new=AsyncMock()) as mock_delete:
        await delete_sync.delete_by_conversation("conv-123")

    assert mock_delete.await_count == 2  # public + private collections
    for call in mock_delete.await_args_list:
        _, field_filter = call.args
        assert field_filter == {"conversation_id": "conv-123"}


@pytest.mark.asyncio
async def test_delete_by_group_filters_on_group_id() -> None:
    with patch.object(delete_sync, "delete_by_fields", new=AsyncMock()) as mock_delete:
        await delete_sync.delete_by_group("group-456")

    assert mock_delete.await_count == 2
    for call in mock_delete.await_args_list:
        _, field_filter = call.args
        assert field_filter == {"group_id": "group-456"}


@pytest.mark.asyncio
async def test_delete_by_conversation_swallows_errors() -> None:
    """Best-effort semantics — a Qdrant failure must never raise, matching
    delete_by_server/delete_by_channel's existing try/except-log pattern."""
    with patch.object(
        delete_sync, "delete_by_fields", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await delete_sync.delete_by_conversation("conv-err")  # must not raise


@pytest.mark.asyncio
async def test_delete_by_group_swallows_errors() -> None:
    with patch.object(
        delete_sync, "delete_by_fields", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await delete_sync.delete_by_group("group-err")  # must not raise


# ---------------------------------------------------------------------------
# POST /internal/v1/delete — endpoint-level scope handling
# ---------------------------------------------------------------------------


def test_delete_conversation_scope_returns_ok(authed_client: TestClient) -> None:
    with patch("app.api.internal.delete_by_conversation", new=AsyncMock()) as mock_fn:
        resp = authed_client.post(
            "/internal/v1/delete", json={"scope": "conversation", "id": "conv-1"}
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok"}
    mock_fn.assert_awaited_once_with("conv-1")


def test_delete_group_scope_returns_ok(authed_client: TestClient) -> None:
    with patch("app.api.internal.delete_by_group", new=AsyncMock()) as mock_fn:
        resp = authed_client.post(
            "/internal/v1/delete", json={"scope": "group", "id": "group-1"}
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok"}
    mock_fn.assert_awaited_once_with("group-1")


def test_delete_unknown_scope_still_400s(authed_client: TestClient) -> None:
    resp = authed_client.post(
        "/internal/v1/delete", json={"scope": "not-a-real-scope", "id": "x"}
    )
    assert resp.status_code == 400, resp.text

"""Index sync must actually REMOVE things.

* ingest_one used to only upsert: a profile switched to private (zero chunks),
  a server switched public → private (new point lands in the other
  collection), or shortened text (fewer chunks) all left the old points
  searchable. It now clears the source first, always.
* scope "user" on /internal/v1/delete removes an account's profile point and
  its notifications on deactivate/delete.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.core.security import require_internal_token
from app.main import app
from app.rag import delete_sync, ingest_sync
from fake_mongo import FakeCollection, FakeDb


@pytest.fixture()
def authed_client():
    async def _always_ok() -> None:
        return

    app.dependency_overrides[require_internal_token] = _always_ok
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.pop(require_internal_token, None)


def _install(monkeypatch, db):
    monkeypatch.setattr(ingest_sync, "get_db", lambda: db)
    monkeypatch.setattr(ingest_sync, "_bootstrap", AsyncMock())


async def test_ingest_one_deletes_stale_points_when_profile_went_private(monkeypatch):
    uid = ObjectId()
    db = FakeDb(users=FakeCollection([
        {"_id": uid, "name": "Sam", "about": "hi",
         "settings": {"privacy": {"profileVisibility": "private"}}},
    ]))
    _install(monkeypatch, db)
    upsert = AsyncMock()
    monkeypatch.setattr(ingest_sync, "_embed_and_upsert", upsert)
    with patch.object(delete_sync, "delete_source", new=AsyncMock()) as mock_delete:
        result = await ingest_sync.ingest_one("user", str(uid))

    mock_delete.assert_awaited_once_with("user", str(uid))
    upsert.assert_not_awaited()
    assert result.chunks == 0


async def test_ingest_one_deletes_before_reupserting(monkeypatch):
    uid = ObjectId()
    db = FakeDb(users=FakeCollection([
        {"_id": uid, "name": "Sam", "about": "hello there",
         "settings": {"privacy": {"profileVisibility": "public"}}},
    ]))
    _install(monkeypatch, db)
    order: list[str] = []

    async def fake_delete(*_a, **_k):
        order.append("delete")

    async def fake_upsert(chunks, result):
        order.append("upsert")

    monkeypatch.setattr(ingest_sync, "_embed_and_upsert", fake_upsert)
    with patch.object(delete_sync, "delete_source", new=fake_delete):
        await ingest_sync.ingest_one("user", str(uid))

    assert order == ["delete", "upsert"]


async def test_ingest_one_never_indexes_a_deactivated_profile(monkeypatch):
    uid = ObjectId()
    db = FakeDb(users=FakeCollection([
        {"_id": uid, "name": "Sam", "accountStatus": "deactivated",
         "settings": {"privacy": {"profileVisibility": "public"}}},
    ]))
    _install(monkeypatch, db)
    upsert = AsyncMock()
    monkeypatch.setattr(ingest_sync, "_embed_and_upsert", upsert)
    with patch.object(delete_sync, "delete_source", new=AsyncMock()) as mock_delete:
        await ingest_sync.ingest_one("user", str(uid))

    mock_delete.assert_awaited_once()
    upsert.assert_not_awaited()


async def test_ingest_one_clears_index_for_a_vanished_document(monkeypatch):
    _install(monkeypatch, FakeDb(users=FakeCollection([])))
    with patch.object(delete_sync, "delete_source", new=AsyncMock()) as mock_delete:
        await ingest_sync.ingest_one("user", str(ObjectId()))
    mock_delete.assert_awaited_once()


async def test_delete_by_user_removes_profile_and_notifications():
    with patch.object(delete_sync, "delete_by_fields", new=AsyncMock()) as mock_delete, \
         patch.object(delete_sync, "scroll_source_ids",
                      new=AsyncMock(side_effect=[["n1", "n2"], ["n3"]])), \
         patch.object(delete_sync, "delete_index_entries", new=AsyncMock()) as mock_idx, \
         patch.object(delete_sync, "delete_index_entries_many", new=AsyncMock()) as mock_idx_many:
        await delete_sync.delete_by_user("u-1")

    conditions = [c.args[1] for c in mock_delete.await_args_list]
    # Both collections, both kinds.
    assert conditions.count({"source_type": "user", "source_id": "u-1"}) == 2
    assert conditions.count({"source_type": "notification", "allowed_user_ids": "u-1"}) == 2
    mock_idx.assert_awaited_once_with("user", "u-1")
    mock_idx_many.assert_awaited_once_with("notification", ["n1", "n2", "n3"])


async def test_delete_by_user_rejects_blank_id():
    with pytest.raises(ValueError):
        await delete_sync.delete_by_user("")


def test_delete_endpoint_user_scope(authed_client):
    with patch("app.api.internal.delete_by_user", new=AsyncMock()) as mock_user:
        resp = authed_client.post("/internal/v1/delete", json={"scope": "user", "id": "u-9"})
    assert resp.status_code == 200
    mock_user.assert_awaited_once_with("u-9")


def test_delete_endpoint_user_scope_requires_id(authed_client):
    with patch("app.api.internal.delete_by_user", new=AsyncMock()) as mock_user:
        resp = authed_client.post("/internal/v1/delete", json={"scope": "user", "id": "  "})
    assert resp.status_code == 400
    mock_user.assert_not_awaited()

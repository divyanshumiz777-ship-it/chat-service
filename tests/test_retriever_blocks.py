"""RAG must respect blocks.

Every MongoDB-backed tool filters blocked users in both directions, but
search_all did not: Qdrant's ACL filter (visibility + allowed_user_ids) is
baked at ingest time and has no concept of a block. A blocked user's public
profile, public reels, and public-server messages therefore stayed fully
searchable and could be cited in an answer.
"""

from __future__ import annotations

import pytest

from app.rag import retriever
from app.rag.retriever import SearchResult, _drop_blocked


def result(source_id: str, owner_id: str | None) -> SearchResult:
    return SearchResult(
        id=f"pt-{source_id}",
        source_type="user",
        source_id=source_id,
        title=source_id,
        score=0.9,
        metadata={"owner_id": owner_id} if owner_id else {},
    )


class FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return self._docs


class FakeUsers:
    def __init__(self, my_blocked, blocked_me):
        self._my_blocked = my_blocked
        self._blocked_me = blocked_me

    async def find_one(self, *_args, **_kwargs):
        return {"blockedUsers": self._my_blocked}

    def find(self, *_args, **_kwargs):
        return FakeCursor([{"_id": oid} for oid in self._blocked_me])


class FakeDb:
    def __init__(self, users):
        self._users = users

    def __getitem__(self, name):
        assert name == "users"
        return self._users


ME = "507f1f77bcf86cd799439011"
BLOCKED = "507f1f77bcf86cd799439012"
BLOCKER = "507f1f77bcf86cd799439013"
NEUTRAL = "507f1f77bcf86cd799439014"


def install_db(monkeypatch, my_blocked=(), blocked_me=()):
    from bson import ObjectId

    users = FakeUsers(
        [ObjectId(x) for x in my_blocked],
        [ObjectId(x) for x in blocked_me],
    )
    import app.core.mongo as mongo

    monkeypatch.setattr(mongo, "get_db", lambda: FakeDb(users))


@pytest.mark.asyncio
class TestDropBlocked:
    async def test_hides_users_the_caller_blocked(self, monkeypatch):
        install_db(monkeypatch, my_blocked=[BLOCKED])
        out = await _drop_blocked(ME, [result("a", BLOCKED), result("b", NEUTRAL)])
        assert [r.source_id for r in out] == ["b"]

    async def test_hides_users_who_blocked_the_caller(self, monkeypatch):
        """The reverse direction — otherwise someone who blocked you can still
        be surfaced to you by the assistant."""
        install_db(monkeypatch, blocked_me=[BLOCKER])
        out = await _drop_blocked(ME, [result("a", BLOCKER), result("b", NEUTRAL)])
        assert [r.source_id for r in out] == ["b"]

    async def test_keeps_everything_when_no_blocks_exist(self, monkeypatch):
        install_db(monkeypatch)
        results = [result("a", NEUTRAL), result("b", BLOCKED)]
        assert await _drop_blocked(ME, results) == results

    async def test_ownerless_results_are_untouched(self, monkeypatch):
        """Server/channel chunks may carry no owner_id; they must not vanish."""
        install_db(monkeypatch, my_blocked=[BLOCKED])
        out = await _drop_blocked(ME, [result("a", None), result("b", BLOCKED)])
        assert [r.source_id for r in out] == ["a"]

    async def test_fails_closed_when_the_blocklist_lookup_errors(self, monkeypatch):
        """A DB failure must not widen visibility — owned content is dropped."""
        import app.core.mongo as mongo

        def boom():
            raise RuntimeError("mongo down")

        monkeypatch.setattr(mongo, "get_db", boom)
        out = await _drop_blocked(ME, [result("a", NEUTRAL), result("b", None)])
        assert [r.source_id for r in out] == ["b"]

    async def test_noop_without_a_user_id(self, monkeypatch):
        results = [result("a", BLOCKED)]
        assert await _drop_blocked("", results) == results

"""Tests for catch_me_up.py's DM/reel highlight extensions — specifically the
bidirectional-block ACL (copied from conversation_insights.py/list_creators.py)
and the early-return fix that previously skipped DM/reel highlights entirely
for a user with zero servers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId

from app.tools import catch_me_up

SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, *_a, **_k):
        return self

    async def to_list(self, length=None):
        return self._docs[:length] if length is not None else self._docs


class _FakeCollection:
    def __init__(self, spec):
        self._find_one_fn = spec.get("find_one")
        self._find_fn = spec.get("find")

    async def find_one(self, query, *_a, **_k):
        if self._find_one_fn is None:
            return None
        return self._find_one_fn(query)

    def find(self, query, *_a, **_k):
        docs = self._find_fn(query) if self._find_fn else []
        return _Cursor(docs)


class _FakeDb:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        return _FakeCollection(self._collections.get(name, {}))


def _oid():
    return ObjectId()


async def test_dm_highlights_excludes_users_i_blocked():
    me, blocked_other, ok_other = _oid(), _oid(), _oid()
    conv_blocked, conv_ok = _oid(), _oid()

    db = _FakeDb({
        "users": {
            "find_one": lambda q: {"_id": me, "blockedUsers": [blocked_other]},
            "find": lambda q: (
                [] if "blockedUsers" in q else [{"_id": ok_other, "name": "Friend"}]
            ),
        },
        "conversations": {
            "find": lambda q: [
                {"_id": conv_blocked, "participants": [me, blocked_other]},
                {"_id": conv_ok, "participants": [me, ok_other]},
            ],
        },
        "messages": {
            "find": lambda q: (
                [{"sender": ok_other, "conversationId": conv_ok, "plainText": "hi there", "createdAt": SINCE}]
                if conv_ok in q.get("conversationId", {}).get("$in", [])
                and conv_blocked not in q.get("conversationId", {}).get("$in", [])
                else []
            ),
        },
    })

    result = await catch_me_up._get_dm_highlights(db, me, SINCE)

    assert len(result) == 1
    assert result[0]["conversation_id"] == str(conv_ok)
    assert result[0]["sender"] == "Friend"


async def test_dm_highlights_excludes_users_who_blocked_me():
    me, other_blocked_me = _oid(), _oid()
    conv1 = _oid()

    db = _FakeDb({
        "users": {
            "find_one": lambda q: {"_id": me, "blockedUsers": []},
            "find": lambda q: ([{"_id": other_blocked_me}] if "blockedUsers" in q else []),
        },
        "conversations": {
            "find": lambda q: [{"_id": conv1, "participants": [me, other_blocked_me]}],
        },
        "messages": {
            "find": lambda q: [
                {"sender": other_blocked_me, "conversationId": conv1, "plainText": "hey", "createdAt": SINCE}
            ],
        },
    })

    result = await catch_me_up._get_dm_highlights(db, me, SINCE)

    assert result == []


async def test_dm_highlights_empty_when_no_conversations():
    me = _oid()
    db = _FakeDb({
        "users": {"find_one": lambda q: {"_id": me, "blockedUsers": []}},
        "conversations": {"find": lambda q: []},
    })

    result = await catch_me_up._get_dm_highlights(db, me, SINCE)
    assert result == []


async def test_reel_highlights_excludes_blocked_creators():
    me, blocked_creator, ok_creator = _oid(), _oid(), _oid()
    reel1 = _oid()

    db = _FakeDb({
        "follows": {
            "find": lambda q: [{"followee": blocked_creator}, {"followee": ok_creator}],
        },
        "users": {
            "find_one": lambda q: {"_id": me, "blockedUsers": [blocked_creator]},
            "find": lambda q: [{"_id": ok_creator, "name": "Creator", "blockedUsers": []}],
        },
        "reels": {
            "find": lambda q: [
                {"_id": reel1, "creator_id": ok_creator, "caption": "hello", "created_at": SINCE}
            ],
        },
    })

    result = await catch_me_up._get_reel_highlights(db, me, SINCE)

    assert len(result) == 1
    assert result[0]["reel_id"] == str(reel1)
    assert result[0]["creator_name"] == "Creator"


async def test_reel_highlights_empty_when_no_follows():
    me = _oid()
    db = _FakeDb({
        "follows": {"find": lambda q: []},
        "users": {"find_one": lambda q: {"_id": me, "blockedUsers": []}},
    })

    result = await catch_me_up._get_reel_highlights(db, me, SINCE)
    assert result == []


async def test_zero_servers_still_returns_dm_and_reel_highlights(monkeypatch):
    """Regression test for the early-return bug: a user with no servers must
    still get their DM/reel highlights, not an empty CatchMeUpData."""
    me, friend, creator = _oid(), _oid(), _oid()
    conv1, reel1 = _oid(), _oid()

    db = _FakeDb({
        "discordservers": {"find": lambda q: []},
        "users": {
            "find_one": lambda q: {"_id": me, "blockedUsers": []},
            "find": lambda q: (
                [{"_id": creator, "name": "Creator", "blockedUsers": []}]
                if "blockedUsers" in q
                else [{"_id": friend, "name": "Friend"}]
            ),
        },
        "conversations": {"find": lambda q: [{"_id": conv1, "participants": [me, friend]}]},
        "messages": {
            "find": lambda q: [{"sender": friend, "conversationId": conv1, "plainText": "hi", "createdAt": SINCE}]
        },
        "follows": {"find": lambda q: [{"followee": creator}]},
        "reels": {
            "find": lambda q: [{"_id": reel1, "creator_id": creator, "caption": "new reel", "created_at": SINCE}]
        },
    })

    monkeypatch.setattr(catch_me_up, "get_db", lambda: db)

    result = await catch_me_up._catch_me_up_impl(str(me), SINCE)

    assert result is not None
    assert result.servers == []
    assert len(result.dm_highlights) == 1
    assert len(result.reel_highlights) == 1

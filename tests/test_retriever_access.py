"""Query-time access re-check (_drop_inaccessible).

Qdrant's ACL is a snapshot from indexing time and re-indexing skips unchanged
text, so membership loss, a server going private, a profile going private and
account deactivation/deletion never reached the index. These tests pin the
re-check against CURRENT Mongo state that now runs on every search.
"""

from __future__ import annotations

from bson import ObjectId

from app.rag.retriever import SearchResult, _drop_inaccessible
from fake_mongo import FakeCollection, FakeDb

ME = ObjectId()
OTHER = ObjectId()

PUBLIC_SRV = ObjectId()
PRIVATE_MEMBER_SRV = ObjectId()
PRIVATE_LEFT_SRV = ObjectId()
PRIVATE_BANNED_SRV = ObjectId()
PRIVATE_OWNED_SRV = ObjectId()
FLIPPED_SRV = ObjectId()  # was public at index time, private now
MY_CONV = ObjectId()
OTHERS_CONV = ObjectId()
MY_GROUP = ObjectId()
LEFT_GROUP = ObjectId()
PUBLIC_USER = ObjectId()
PRIVATE_USER = ObjectId()
DEACTIVATED_USER = ObjectId()
DELETED_USER = ObjectId()
LIVE_REEL = ObjectId()
DELETED_REEL = ObjectId()


def r(source_type: str, source_id, **md) -> SearchResult:
    return SearchResult(
        id=f"pt-{source_id}",
        source_type=source_type,
        source_id=str(source_id),
        title=str(source_id),
        score=0.9,
        metadata={k: str(v) for k, v in md.items()},
    )


def make_db(**overrides) -> FakeDb:
    collections = dict(
        discordservers=FakeCollection([
            {"_id": PUBLIC_SRV, "visibility": "public", "owner": OTHER},
            {"_id": PRIVATE_MEMBER_SRV, "visibility": "private", "owner": OTHER},
            {"_id": PRIVATE_LEFT_SRV, "visibility": "private", "owner": OTHER},
            {"_id": PRIVATE_BANNED_SRV, "visibility": "invite-only", "owner": OTHER},
            {"_id": PRIVATE_OWNED_SRV, "visibility": "private", "owner": ME},
            {"_id": FLIPPED_SRV, "visibility": "private", "owner": OTHER},
        ]),
        servermembers=FakeCollection([
            {"server": PRIVATE_MEMBER_SRV, "user": ME},
            {"server": PRIVATE_BANNED_SRV, "user": ME, "banned": {"isBanned": True}},
        ]),
        conversations=FakeCollection([
            {"_id": MY_CONV, "participants": [ME, OTHER]},
            {"_id": OTHERS_CONV, "participants": [OTHER]},
        ]),
        groups=FakeCollection([
            {"_id": MY_GROUP, "participants": [ME, OTHER]},
            {"_id": LEFT_GROUP, "participants": [OTHER]},
        ]),
        users=FakeCollection([
            {"_id": PUBLIC_USER, "settings": {"privacy": {"profileVisibility": "public"}}},
            {"_id": PRIVATE_USER, "settings": {"privacy": {"profileVisibility": "private"}}},
            {"_id": DEACTIVATED_USER, "accountStatus": "deactivated",
             "settings": {"privacy": {"profileVisibility": "public"}}},
            {"_id": DELETED_USER, "accountStatus": "deleted",
             "settings": {"privacy": {"profileVisibility": "public"}}},
        ]),
        reels=FakeCollection([
            {"_id": LIVE_REEL},
            {"_id": DELETED_REEL, "isDeleted": True},
        ]),
    )
    collections.update(overrides)
    return FakeDb(**collections)


def install(monkeypatch, db: FakeDb) -> None:
    import app.core.mongo as mongo

    monkeypatch.setattr(mongo, "get_db", lambda: db)


def ids(results: list[SearchResult]) -> set[str]:
    return {x.source_id for x in results}


async def test_server_scoped_results_follow_current_membership(monkeypatch):
    install(monkeypatch, make_db())
    results = [
        r("message", "m-public", server_id=PUBLIC_SRV),
        r("message", "m-member", server_id=PRIVATE_MEMBER_SRV),
        r("channel", "c-left", server_id=PRIVATE_LEFT_SRV),
        r("voice_session", "v-banned", server_id=PRIVATE_BANNED_SRV),
        r("server", "s-owned", server_id=PRIVATE_OWNED_SRV),
        r("thread", "t-flipped", server_id=FLIPPED_SRV),
        r("message", "m-gone", server_id=ObjectId()),  # server deleted
    ]
    out = await _drop_inaccessible(str(ME), results)
    assert ids(out) == {"m-public", "m-member", "s-owned"}


async def test_dm_and_group_results_require_current_participation(monkeypatch):
    install(monkeypatch, make_db())
    results = [
        r("message", "dm-mine", conversation_id=MY_CONV),
        r("message", "dm-not-mine", conversation_id=OTHERS_CONV),
        r("message", "g-mine", group_id=MY_GROUP),
        r("message", "g-left", group_id=LEFT_GROUP),
    ]
    out = await _drop_inaccessible(str(ME), results)
    assert ids(out) == {"dm-mine", "g-mine"}


async def test_profiles_must_be_public_and_active(monkeypatch):
    install(monkeypatch, make_db())
    results = [
        r("user", PUBLIC_USER),
        r("user", PRIVATE_USER),
        r("user", DEACTIVATED_USER),
        r("user", DELETED_USER),
        r("user", ObjectId()),  # account document gone
    ]
    out = await _drop_inaccessible(str(ME), results)
    assert ids(out) == {str(PUBLIC_USER)}


async def test_reels_must_still_exist_and_not_be_deleted(monkeypatch):
    install(monkeypatch, make_db())
    out = await _drop_inaccessible(
        str(ME), [r("reel", LIVE_REEL), r("reel", DELETED_REEL), r("reel", ObjectId())]
    )
    assert ids(out) == {str(LIVE_REEL)}


async def test_unscoped_results_pass_through(monkeypatch):
    install(monkeypatch, make_db())
    results = [r("docs", "help-1"), r("notification", "n-1")]
    out = await _drop_inaccessible(str(ME), results)
    assert ids(out) == {"help-1", "n-1"}


async def test_fails_closed_when_a_lookup_errors(monkeypatch):
    install(monkeypatch, make_db(conversations=FakeCollection(fail=True)))
    results = [
        r("docs", "help-1"),
        r("message", "m-public", server_id=PUBLIC_SRV),
        r("message", "dm-mine", conversation_id=MY_CONV),
    ]
    out = await _drop_inaccessible(str(ME), results)
    # Only the result that never needed a check survives.
    assert ids(out) == {"help-1"}


async def test_invalid_caller_id_drops_everything_that_needs_a_check(monkeypatch):
    install(monkeypatch, make_db())
    out = await _drop_inaccessible(
        "not-an-object-id", [r("docs", "help-1"), r("message", "m", server_id=PUBLIC_SRV)]
    )
    assert ids(out) == {"help-1"}

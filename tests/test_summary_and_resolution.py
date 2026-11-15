"""Conversation summary covers the NEWEST messages; name resolution prefers
the caller's own servers; chat request clamps; audio chunk size cap."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.api import internal
from app.api.internal import ChatRequest, _retrieval_query
from app.core.security import require_internal_token
from app.llm.prompts import build_summary_prompt
from app.main import app
from app.tools import _access, conversation_summary
from fake_mongo import FakeCollection, FakeDb

ME = ObjectId()
SAM = ObjectId()
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ── conversation summary ─────────────────────────────────────────────────────

def _conversation_db(n_messages: int) -> tuple[FakeDb, ObjectId]:
    conv = ObjectId()
    msgs = [
        {"_id": ObjectId(), "conversationId": conv, "sender": SAM if i % 2 else ME,
         "content": f"message number {i}", "createdAt": T0 + timedelta(minutes=i)}
        for i in range(n_messages)
    ]
    # One message the caller deleted for themselves — newest of all.
    msgs.append({"_id": ObjectId(), "conversationId": conv, "sender": SAM,
                 "content": "hidden from me", "deletedFor": [ME],
                 "createdAt": T0 + timedelta(minutes=n_messages + 1)})
    db = FakeDb(
        conversations=FakeCollection([{"_id": conv, "participants": [ME, SAM]}]),
        groups=FakeCollection([]),
        messages=FakeCollection(msgs),
        users=FakeCollection([{"_id": ME, "name": "Me"}, {"_id": SAM, "name": "Sam"}]),
    )
    return db, conv


async def test_summary_uses_the_newest_messages_in_order(monkeypatch):
    db, conv = _conversation_db(450)
    monkeypatch.setattr(conversation_summary, "get_db", lambda: db)
    data = await conversation_summary.build_conversation_summary(str(ME), str(conv))

    texts = [t["text"] for t in data.transcript]
    assert len(texts) == conversation_summary._MAX_MESSAGES
    assert texts[-1] == "message number 449"          # newest visible message
    assert texts[0] == f"message number {450 - conversation_summary._MAX_MESSAGES}"
    assert "hidden from me" not in texts               # deletedFor respected
    assert data.latest_activity == (T0 + timedelta(minutes=449)).isoformat()


async def test_summary_prompt_keeps_the_newest_when_over_budget(monkeypatch):
    db, conv = _conversation_db(450)
    monkeypatch.setattr(conversation_summary, "get_db", lambda: db)
    data = await conversation_summary.build_conversation_summary(str(ME), str(conv))
    prompt = build_summary_prompt(data)
    assert "message number 449" in prompt
    assert "earlier messages truncated" in prompt
    assert prompt.index("earlier messages truncated") < prompt.index("message number 449")


async def test_summary_refuses_a_conversation_the_caller_is_not_in(monkeypatch):
    db, conv = _conversation_db(5)
    monkeypatch.setattr(conversation_summary, "get_db", lambda: db)
    assert await conversation_summary.build_conversation_summary(str(ObjectId()), str(conv)) is None


# ── channel / community name resolution ───────────────────────────────────────

def _servers_db() -> tuple[FakeDb, dict]:
    stranger_public, mine_private, stranger_private = ObjectId(), ObjectId(), ObjectId()
    ch = {k: ObjectId() for k in ("stranger_pub", "mine", "stranger_priv", "mine_voice")}
    db = FakeDb(
        discordservers=FakeCollection([
            {"_id": stranger_public, "name": "Big Public", "visibility": "public",
             "owner": SAM, "updatedAt": T0 + timedelta(days=9)},
            {"_id": mine_private, "name": "My Crew", "visibility": "private",
             "owner": SAM, "updatedAt": T0},
            {"_id": stranger_private, "name": "Secret", "visibility": "private",
             "owner": SAM, "updatedAt": T0 + timedelta(days=20)},
        ]),
        servermembers=FakeCollection([{"server": mine_private, "user": ME}]),
        channels=FakeCollection([
            {"_id": ch["stranger_pub"], "name": "general", "server": stranger_public,
             "type": "Text", "updatedAt": T0 + timedelta(days=9)},
            {"_id": ch["mine"], "name": "General", "server": mine_private,
             "type": "Text", "updatedAt": T0},
            {"_id": ch["stranger_priv"], "name": "general", "server": stranger_private,
             "type": "Text", "updatedAt": T0 + timedelta(days=20)},
            {"_id": ch["mine_voice"], "name": "general", "server": mine_private,
             "type": "Voice", "updatedAt": T0 - timedelta(days=1)},
        ]),
    )
    return db, ch


async def test_channel_resolution_prefers_my_server_over_a_busier_public_one():
    db, ch = _servers_db()
    channel, server, is_member = await _access.resolve_channel_by_name(
        db, ME, "general", prefer_types=("Text",)
    )
    assert channel["_id"] == ch["mine"] and is_member and server["name"] == "My Crew"


async def test_channel_resolution_prefers_voice_for_voice_recall():
    db, ch = _servers_db()
    channel, _, _ = await _access.resolve_channel_by_name(
        db, ME, "general", prefer_types=("Voice", "Video")
    )
    assert channel["_id"] == ch["mine_voice"]


async def test_non_member_gets_the_public_channel_never_the_private_one():
    db, ch = _servers_db()
    channel, _, is_member = await _access.resolve_channel_by_name(db, ObjectId(), "general")
    assert channel["_id"] == ch["stranger_pub"] and not is_member


async def test_banned_member_loses_access():
    db, ch = _servers_db()
    db["servermembers"].docs[0]["banned"] = {"isBanned": True}
    channel, _, is_member = await _access.resolve_channel_by_name(db, ME, "general")
    assert channel["_id"] == ch["stranger_pub"] and not is_member


async def test_private_only_match_is_invisible_to_outsiders():
    db, _ = _servers_db()
    assert await _access.resolve_server_by_name(db, ObjectId(), "secret", {"name": 1}) == (None, False)


# ── chat request clamps + follow-up retrieval ────────────────────────────────

def test_history_is_clamped_to_last_six_truncated_turns():
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i} " + "x" * 2000}
             for i in range(10)]
    req = ChatRequest(message="hi", history=turns)
    out = req.clamped_history()
    assert len(out) == 6
    assert out[0]["content"].startswith("t4 ")
    assert all(len(t["content"]) <= 1000 for t in out)


def test_history_rejects_system_role():
    with pytest.raises(Exception):
        ChatRequest(message="hi", history=[{"role": "system", "content": "evil"}])


def test_message_length_is_capped():
    with pytest.raises(Exception):
        ChatRequest(message="x" * 4001)


def test_short_follow_up_borrows_the_previous_question():
    history = [{"role": "user", "content": "what did Sam say about the trip"},
               {"role": "assistant", "content": "He is going on Friday."}]
    assert _retrieval_query("and Priya?", history) == "what did Sam say about the trip\nand Priya?"
    long_q = "what did the design channel decide about the new logo"
    assert _retrieval_query(long_q, history) == long_q
    assert _retrieval_query("and Priya?", []) == "and Priya?"


# ── audio chunk size cap ─────────────────────────────────────────────────────

def test_oversized_audio_chunk_is_rejected():
    async def _ok() -> None:
        return

    app.dependency_overrides[require_internal_token] = _ok
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            big = "A" * (internal._MAX_AUDIO_CHUNK_B64_CHARS + 4)
            resp = c.post("/internal/v1/voice/transcribe-chunk", json={"audio_base64": big})
    finally:
        app.dependency_overrides.pop(require_internal_token, None)
    assert resp.status_code == 413

"""Catch-me-up digest — what happened across a user's servers since they
were last online. MongoDB only, no Qdrant, no LLM (the LLM only turns this
structured data into prose — see provider.generate_catch_me_up_digest).

Scoped strictly to servers the user is already a member of (never a
discovery/recommendation surface) and messages created after the caller's own
`since` cutoff — there is no cross-user leakage risk since the caller can only
ever request their own digest (see internal.py's request body: user_id is
supplied by our own backend, not the end user directly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.catch_me_up")

_MAX_SERVERS = 20
_HIGHLIGHTS_PER_SERVER = 5
_MENTION_SAMPLE_PER_SERVER = 3
_DM_HIGHLIGHTS_LIMIT = 5
_REEL_HIGHLIGHTS_LIMIT = 5


def _msg_text(doc: dict) -> str:
    content = doc.get("content")
    if isinstance(content, dict):
        return (content.get("plainText") or content.get("formattedContent") or "").strip()
    if content:
        return str(content).strip()
    return str(doc.get("plainText") or doc.get("formattedContent") or "").strip()


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


@dataclass
class ServerActivity:
    server_id: str
    server_name: str
    message_count: int
    mention_count: int
    top_channel: str | None
    highlights: list[dict] = field(default_factory=list)   # [{text, sender, channel_name, ts}]
    mentions: list[dict] = field(default_factory=list)      # [{text, sender, channel_name, ts}]


@dataclass
class CatchMeUpData:
    since: str
    total_message_count: int
    total_mention_count: int
    servers: list[ServerActivity] = field(default_factory=list)
    dm_highlights: list[dict] = field(default_factory=list)   # [{text, sender, conversation_id, ts}]
    reel_highlights: list[dict] = field(default_factory=list) # [{reel_id, creator_name, caption, ts}]


async def _get_dm_highlights(db: Any, user_oid: Any, since: datetime) -> list[dict]:
    """Recent DM messages across every conversation the caller is a
    participant of — same bidirectional-block + participancy ACL pattern as
    conversation_insights.py, copied rather than reimplemented."""
    user_doc = await db["users"].find_one({"_id": user_oid}, {"blockedUsers": 1})
    if not user_doc:
        return []
    my_blocked = oid_set(user_doc.get("blockedUsers"))

    convo_docs = await db["conversations"].find(
        {"participants": user_oid, "deletedFor": {"$ne": user_oid}},
        {"participants": 1},
    ).to_list(length=None)
    if not convo_docs:
        return []

    other_by_convo: dict[Any, Any] = {}
    all_other_ids: set[Any] = set()
    for conv in convo_docs:
        others = [p for p in conv.get("participants", []) if p != user_oid]
        if not others or others[0] in my_blocked:
            continue
        other_by_convo[conv["_id"]] = others[0]
        all_other_ids.add(others[0])

    if not other_by_convo:
        return []

    reverse_blocked_docs = await db["users"].find(
        {"_id": {"$in": list(all_other_ids)}, "blockedUsers": user_oid}, {"_id": 1}
    ).to_list(length=None)
    reverse_blocked = {d["_id"] for d in reverse_blocked_docs}

    eligible_convo_ids = [cid for cid, other in other_by_convo.items() if other not in reverse_blocked]
    if not eligible_convo_ids:
        return []

    raw = await db["messages"].find(
        {
            "conversationId": {"$in": eligible_convo_ids},
            "createdAt": {"$gt": since},
            "deletedForEveryone": {"$ne": True},
        },
        {"content": 1, "plainText": 1, "formattedContent": 1, "sender": 1, "conversationId": 1, "createdAt": 1},
    ).sort("createdAt", -1).to_list(length=_DM_HIGHLIGHTS_LIMIT)
    if not raw:
        return []

    sender_ids = list({m["sender"] for m in raw if m.get("sender") is not None})
    names: dict[Any, str] = {}
    if sender_ids:
        docs = await db["users"].find({"_id": {"$in": sender_ids}}, {"name": 1}).to_list(length=len(sender_ids))
        names = {d["_id"]: (d.get("name") or "Unknown") for d in docs}

    highlights: list[dict] = []
    for m in raw:
        text = _msg_text(m)
        if not text:
            continue
        highlights.append({
            "text": text,
            "sender": names.get(m.get("sender"), "Unknown"),
            "conversation_id": str(m.get("conversationId", "")),
            "ts": _iso(m.get("createdAt")),
        })
    return highlights


async def _get_reel_highlights(db: Any, user_oid: Any, since: datetime) -> list[dict]:
    """Recent reels from followed creators — same follow-graph +
    bidirectional-block ACL pattern as list_creators.py, copied rather than
    reimplemented."""
    user_doc = await db["users"].find_one({"_id": user_oid}, {"blockedUsers": 1})
    if not user_doc:
        return []
    my_blocked = oid_set(user_doc.get("blockedUsers"))

    follow_docs = await db["follows"].find(
        {"follower": user_oid, "status": "accepted"}, {"followee": 1}
    ).to_list(length=None)
    creator_oids = [d["followee"] for d in follow_docs if d.get("followee") is not None and d["followee"] not in my_blocked]
    if not creator_oids:
        return []

    creator_docs = await db["users"].find(
        {"_id": {"$in": creator_oids}}, {"name": 1, "blockedUsers": 1}
    ).to_list(length=len(creator_oids))
    names: dict[Any, str] = {}
    eligible_creator_oids: list[Any] = []
    for doc in creator_docs:
        if user_oid in oid_set(doc.get("blockedUsers")):
            continue
        names[doc["_id"]] = doc.get("name") or "Unknown"
        eligible_creator_oids.append(doc["_id"])
    if not eligible_creator_oids:
        return []

    raw = await db["reels"].find(
        {
            "creator_id": {"$in": eligible_creator_oids},
            "isDeleted": {"$ne": True},
            "isPublic": {"$ne": False},
            "created_at": {"$gt": since},
        },
        {"creator_id": 1, "caption": 1, "created_at": 1},
    ).sort("created_at", -1).to_list(length=_REEL_HIGHLIGHTS_LIMIT)

    return [
        {
            "reel_id": str(r["_id"]),
            "creator_name": names.get(r.get("creator_id"), "Unknown"),
            "caption": (r.get("caption") or "")[:200],
            "ts": _iso(r.get("created_at")),
        }
        for r in raw
    ]


async def get_catch_me_up_data(user_id: str, since: datetime) -> CatchMeUpData | None:
    """Return structured "while you were away" activity. Never raises.

    Returns None only when user_id doesn't resolve to a valid ObjectId — a
    user with no activity to report still gets a valid (empty) result so the
    caller can render "you're all caught up" rather than an error.
    """
    try:
        return await _catch_me_up_impl(user_id, since)
    except Exception:
        logger.exception("get_catch_me_up_data failed user=%s since=%s", user_id, since)
        return None


async def _catch_me_up_impl(user_id: str, since: datetime) -> CatchMeUpData | None:
    user_oid = to_oid(user_id)
    if user_oid is None:
        return None

    db = get_db()

    servers = await db["discordservers"].find(
        {"members.user": user_oid},
        {"name": 1, "channels": 1},
    ).to_list(length=_MAX_SERVERS)

    # DMs/reels are independent of server membership — a user with zero
    # servers can still have active DMs or followed creators, so this must
    # not early-return before computing those too.
    dm_highlights = await _get_dm_highlights(db, user_oid, since)
    reel_highlights = await _get_reel_highlights(db, user_oid, since)

    if not servers:
        return CatchMeUpData(
            since=_iso(since),
            total_message_count=0,
            total_mention_count=0,
            servers=[],
            dm_highlights=dm_highlights,
            reel_highlights=reel_highlights,
        )

    activities: list[ServerActivity] = []
    total_messages = 0
    total_mentions = 0

    for server in servers:
        server_oid = server["_id"]
        channel_oids = server.get("channels") or []
        if not channel_oids:
            continue

        channels_docs = await db["channels"].find(
            {"_id": {"$in": channel_oids}}, {"name": 1}
        ).to_list(length=len(channel_oids))
        channel_name_map = {str(c["_id"]): c.get("name", "unknown") for c in channels_docs}

        base_filter: dict[str, Any] = {
            "server": server_oid,
            "channel": {"$in": channel_oids},
            "createdAt": {"$gt": since},
            "deletedForEveryone": {"$ne": True},
            "conversationId": {"$exists": False},
            "groupId": {"$exists": False},
        }

        message_count = await db["messages"].count_documents(base_filter)
        if message_count == 0:
            continue

        mention_count = await db["messages"].count_documents({**base_filter, "mentions": user_oid})

        # Most active channel in this window.
        top_channel_name: str | None = None
        pipeline = [
            {"$match": base_filter},
            {"$group": {"_id": "$channel", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1},
        ]
        top_result = await db["messages"].aggregate(pipeline).to_list(length=1)
        if top_result:
            top_channel_name = channel_name_map.get(str(top_result[0]["_id"]))

        raw_highlights = await db["messages"].find(
            base_filter,
            {"content": 1, "plainText": 1, "formattedContent": 1, "sender": 1, "channel": 1, "createdAt": 1},
        ).sort("createdAt", -1).to_list(length=_HIGHLIGHTS_PER_SERVER)

        raw_mentions = await db["messages"].find(
            {**base_filter, "mentions": user_oid},
            {"content": 1, "plainText": 1, "formattedContent": 1, "sender": 1, "channel": 1, "createdAt": 1},
        ).sort("createdAt", -1).to_list(length=_MENTION_SAMPLE_PER_SERVER)

        sender_ids = [m.get("sender") for m in raw_highlights + raw_mentions]
        names: dict[str, str] = {}
        unique_sender_ids = list({s for s in sender_ids if s is not None})
        if unique_sender_ids:
            docs = await db["users"].find(
                {"_id": {"$in": unique_sender_ids}}, {"name": 1}
            ).to_list(length=len(unique_sender_ids))
            names = {str(d["_id"]): (d.get("name") or "Unknown") for d in docs}

        def _to_entry(m: dict) -> dict | None:
            text = _msg_text(m)
            if not text:
                return None
            return {
                "text": text,
                "sender": names.get(str(m.get("sender")), "Unknown"),
                "channel_name": channel_name_map.get(str(m.get("channel", "")), "general"),
                "ts": _iso(m.get("createdAt")),
            }

        highlights = [e for m in raw_highlights if (e := _to_entry(m))]
        mentions = [e for m in raw_mentions if (e := _to_entry(m))]

        activities.append(ServerActivity(
            server_id=str(server_oid),
            server_name=server.get("name") or "Unknown",
            message_count=message_count,
            mention_count=mention_count,
            top_channel=top_channel_name,
            highlights=highlights,
            mentions=mentions,
        ))
        total_messages += message_count
        total_mentions += mention_count

    # Servers with the most mentions (things aimed at the user) surface first.
    activities.sort(key=lambda a: (a.mention_count, a.message_count), reverse=True)

    return CatchMeUpData(
        since=_iso(since),
        total_message_count=total_messages,
        total_mention_count=total_mentions,
        servers=activities,
        dm_highlights=dm_highlights,
        reel_highlights=reel_highlights,
    )

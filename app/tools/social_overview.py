"""Social overview — aggregate social stats for a user, MongoDB only, no Qdrant.

Returns a single dict with counts, most-active peers, and recent-activity bullets
derived deterministically from MongoDB data. The LLM never sees raw data here —
internal.py formats this dict directly into the reply string.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid
from app.tools.conversation_insights import _format_activity

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.social_overview")

_ACTIVITY_DAYS = 30


async def get_social_overview(user_id: str) -> dict | None:
    """Return a social stats dict, or None on hard error (logged).

    Guaranteed keys: friends_count, communities_count, messages_count,
    most_active_friend, most_active_community, most_active_creator,
    recent_bullets.
    """
    try:
        return await _get_social_overview_impl(user_id)
    except Exception:
        logger.exception("get_social_overview failed for user %s", user_id)
        return None


async def _get_social_overview_impl(user_id: str) -> dict:
    me_oid = to_oid(user_id)
    if not me_oid:
        return _empty()

    db = get_db()
    since = datetime.now(tz=timezone.utc) - timedelta(days=_ACTIVITY_DAYS)

    # ── User doc ────────────────────────────────────────────────────────────────
    me = await db["users"].find_one(
        {"_id": me_oid},
        {"friends": 1, "servers": 1, "blockedUsers": 1},
    )
    if not me:
        return _empty()

    friend_oids = list(oid_set(me.get("friends")))
    blocked_oids = oid_set(me.get("blockedUsers"))
    # Drop blocked friends from count.
    visible_friends = [o for o in friend_oids if o not in blocked_oids]
    friends_count = len(visible_friends)

    logger.info("USER=%s FRIENDS=%d", user_id, friends_count)

    # ── Communities ─────────────────────────────────────────────────────────────
    communities_count = await db["discordservers"].count_documents(
        {"members.user": me_oid}
    )
    logger.info("USER=%s COMMUNITIES=%d", user_id, communities_count)

    # ── Messages sent (last 30 days) ─────────────────────────────────────────────
    messages_count = await db["messages"].count_documents(
        {"sender": me_oid, "createdAt": {"$gte": since}}
    )
    logger.info("USER=%s MESSAGES_30D=%d", user_id, messages_count)

    # ── Most active friend (by most recent DM conversation) ──────────────────────
    most_active_friend: str | None = None
    most_active_friend_time: str | None = None

    conv = await db["conversations"].find_one(
        {"participants": me_oid, "deletedFor": {"$nin": [me_oid]}},
        {"participants": 1, "updatedAt": 1},
        sort=[("updatedAt", -1)],
    )
    if conv:
        other_oids = [p for p in conv.get("participants", []) if p != me_oid]
        if other_oids:
            friend_doc = await db["users"].find_one(
                {"_id": other_oids[0]}, {"name": 1}
            )
            if friend_doc:
                most_active_friend = friend_doc.get("name") or "Unknown"
                most_active_friend_time = _format_activity(conv.get("updatedAt"))

    # ── Most active community (by most recent channel message from the user) ──────
    most_active_community: str | None = None

    recent_msg = await db["messages"].find_one(
        {"sender": me_oid, "server": {"$exists": True}},
        {"server": 1},
        sort=[("createdAt", -1)],
    )
    if recent_msg:
        server_doc = await db["discordservers"].find_one(
            {"_id": recent_msg.get("server")}, {"name": 1}
        )
        if server_doc:
            most_active_community = server_doc.get("name")

    # ── Most active creator (creator of the reel the user most recently engaged) ──
    most_active_creator: str | None = None

    recent_interaction = await db["userreelinteractions"].find_one(
        {"user_id": me_oid},
        {"reel_id": 1},
        sort=[("updatedAt", -1)],
    )
    if recent_interaction:
        reel_doc = await db["reels"].find_one(
            {"_id": to_oid(recent_interaction.get("reel_id"))}, {"creator_id": 1}
        )
        if reel_doc:
            creator_doc = await db["users"].find_one(
                {"_id": reel_doc.get("creator_id")}, {"name": 1}
            )
            if creator_doc:
                most_active_creator = creator_doc.get("name")

    # ── Followed creators count (real Follow graph) ───────────────────────────────
    creators_count = await db["follows"].count_documents(
        {"follower": me_oid, "status": "accepted"}
    )

    # ── Recent activity bullets (deterministic) ──────────────────────────────────
    bullets: list[str] = []
    if messages_count:
        bullets.append(
            f"Sent {messages_count} message{'s' if messages_count != 1 else ''} in the last 30 days"
        )
    if friends_count:
        bullets.append(
            f"Connected with {friends_count} friend{'s' if friends_count != 1 else ''}"
        )
    if communities_count:
        bullets.append(
            f"Active in {communities_count} communit{'ies' if communities_count != 1 else 'y'}"
        )
    if creators_count:
        bullets.append(
            f"Following {creators_count} creator{'s' if creators_count != 1 else ''}"
        )
    if most_active_friend:
        label = f" ({most_active_friend_time})" if most_active_friend_time else ""
        bullets.append(f"Last chatted with {most_active_friend}{label}")
    if most_active_community:
        bullets.append(f"Last active in {most_active_community}")
    if most_active_creator:
        bullets.append(f"Recently engaged with {most_active_creator}'s reels")

    if not bullets:
        bullets.append("No recent activity — start a conversation or join a community!")

    logger.info("USER=%s BULLETS=%d", user_id, len(bullets))

    return {
        "friends_count": friends_count,
        "communities_count": communities_count,
        "messages_count": messages_count,
        "creators_count": creators_count,
        "most_active_friend": most_active_friend,
        "most_active_community": most_active_community,
        "most_active_creator": most_active_creator,
        "recent_bullets": bullets,
    }


def _empty() -> dict:
    return {
        "friends_count": 0,
        "communities_count": 0,
        "messages_count": 0,
        "creators_count": 0,
        "most_active_friend": None,
        "most_active_community": None,
        "most_active_creator": None,
        "recent_bullets": ["No activity data found. Start exploring the platform!"],
    }

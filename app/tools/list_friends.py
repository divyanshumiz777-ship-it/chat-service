"""List the user's existing friends — MongoDB only, no Qdrant, no LLM.

Returns ACTUAL connected friends, NOT recommendations.
Each entry includes the friend's profile, mutual server count, and last activity.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid
from app.tools.conversation_insights import _format_activity

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.list_friends")


async def get_my_friends(user_id: str, limit: int = 20) -> list[dict]:
    """Return the user's actual friend list. Never raises.

    Returns []:
      - user not found
      - no friends added yet
      - unexpected error (logged)
    """
    try:
        return await _get_my_friends_impl(user_id, limit)
    except Exception:
        logger.exception("get_my_friends failed for user %s", user_id)
        return []


async def _get_my_friends_impl(user_id: str, limit: int) -> list[dict]:
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    db = get_db()

    total_users = await db["users"].count_documents({})
    logger.info("USER=%s MONGO_TOTAL=%d", user_id, total_users)

    me = await db["users"].find_one(
        {"_id": me_oid}, {"friends": 1, "servers": 1, "blockedUsers": 1}
    )
    if not me:
        logger.info("USER=%s SKIP_REASON=user_not_found", user_id)
        return []

    friend_oids = list(oid_set(me.get("friends")))
    blocked_oids = oid_set(me.get("blockedUsers"))
    my_server_oids = oid_set(me.get("servers"))

    logger.info("USER=%s FILTERED=%d friends_in_db", user_id, len(friend_oids))

    if not friend_oids:
        logger.info("USER=%s RETURNED=0 SKIP_REASON=no_friends", user_id)
        return []

    # Bulk-fetch friend profiles.
    friend_docs = await db["users"].find(
        {"_id": {"$in": friend_oids}},
        {"name": 1, "profilePic": 1, "servers": 1, "lastSeen": 1, "blockedUsers": 1},
    ).to_list(length=len(friend_oids))

    # Drop reverse-blocked friends.
    skipped_blocked = 0
    friends_by_oid: dict[Any, dict] = {}
    for fd in friend_docs:
        if me_oid in oid_set(fd.get("blockedUsers")):
            skipped_blocked += 1
            logger.debug("USER=%s SKIP_REASON=reverse_block friend=%s", user_id, fd["_id"])
            continue
        if fd["_id"] in blocked_oids:
            skipped_blocked += 1
            logger.debug("USER=%s SKIP_REASON=blocked friend=%s", user_id, fd["_id"])
            continue
        friends_by_oid[fd["_id"]] = fd

    if skipped_blocked:
        logger.info("USER=%s SKIP_REASON=blocked count=%d", user_id, skipped_blocked)

    if not friends_by_oid:
        logger.info("USER=%s RETURNED=0 SKIP_REASON=all_blocked", user_id)
        return []

    eligible_oids = list(friends_by_oid.keys())

    # Last conversation activity with each friend (one bulk query).
    convs = await db["conversations"].find(
        {"participants": me_oid, "deletedFor": {"$nin": [me_oid]}},
        {"participants": 1, "updatedAt": 1},
    ).sort("updatedAt", -1).to_list(length=500)

    friend_last_activity: dict[Any, Any] = {}
    for conv in convs:
        others = [p for p in conv.get("participants", []) if p != me_oid]
        for other in others:
            if other not in friend_last_activity:
                friend_last_activity[other] = conv.get("updatedAt")

    results: list[dict] = []
    for foid, fd in list(friends_by_oid.items())[:limit]:
        friend_server_oids = oid_set(fd.get("servers"))
        mutual = len(my_server_oids & friend_server_oids)
        last_activity = _format_activity(friend_last_activity.get(foid))

        results.append({
            "friend_id": str(foid),
            "name": fd.get("name") or "Unknown",
            "profile_pic": fd.get("profilePic") or "",
            "mutual_servers": mutual,
            "last_activity": last_activity,
            "last_seen": _format_activity(fd.get("lastSeen")),
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

"""List creators the user follows — MongoDB only, no Qdrant, no LLM.

Returns ACTUAL followed creators from the real `follows` collection
(follower/followee/status — see backend/src/models/Follow.ts), NOT
recommendations. Each entry includes the creator's profile and reel count.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.list_creators")


async def get_followed_creators(user_id: str, limit: int = 20) -> list[dict]:
    """Return creators the user follows. Never raises.

    Returns []:
      - user has no accepted follows
      - user not found
      - unexpected error (logged)
    """
    try:
        return await _get_followed_creators_impl(user_id, limit)
    except Exception:
        logger.exception("get_followed_creators failed for user %s", user_id)
        return []


async def _get_followed_creators_impl(user_id: str, limit: int) -> list[dict]:
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    db = get_db()

    me = await db["users"].find_one({"_id": me_oid}, {"blockedUsers": 1})
    if not me:
        logger.info("USER=%s SKIP_REASON=user_not_found", user_id)
        return []
    my_blocked = oid_set(me.get("blockedUsers"))

    follow_docs = await db["follows"].find(
        {"follower": me_oid, "status": "accepted"},
        {"followee": 1},
    ).to_list(length=None)
    creator_oids = [d["followee"] for d in follow_docs if d.get("followee") is not None]

    if not creator_oids:
        logger.info("USER=%s SKIP_REASON=no_followed_creators RETURNED=0", user_id)
        return []

    logger.info("USER=%s FILTERED=%d followed_creators", user_id, len(creator_oids))

    # Bulk-fetch creator profiles — also pull blockedUsers so a follow that
    # predates a block (either direction) doesn't leak through. list_friends.py
    # already does this both-direction check; this tool previously did neither.
    creator_docs = await db["users"].find(
        {"_id": {"$in": creator_oids}},
        {"name": 1, "profilePic": 1, "about": 1, "blockedUsers": 1},
    ).to_list(length=len(creator_oids))

    profiles_by_oid: dict = {}
    skipped_blocked = 0
    for doc in creator_docs:
        cid = doc["_id"]
        if cid in my_blocked or me_oid in oid_set(doc.get("blockedUsers")):
            skipped_blocked += 1
            logger.debug("USER=%s SKIP_REASON=blocked creator=%s", user_id, cid)
            continue
        profiles_by_oid[cid] = doc

    if skipped_blocked:
        logger.info("USER=%s SKIP_REASON=blocked count=%d", user_id, skipped_blocked)

    eligible_oids = [c for c in creator_oids if c in profiles_by_oid][:limit]
    if not eligible_oids:
        logger.info("USER=%s RETURNED=0 SKIP_REASON=all_blocked", user_id)
        return []

    # Reel counts per creator.
    reel_counts: dict = {}
    pipeline = [
        {
            "$match": {
                "creator_id": {"$in": eligible_oids},
                "isDeleted": {"$ne": True},
                "isPublic": {"$ne": False},
            }
        },
        {"$group": {"_id": "$creator_id", "count": {"$sum": 1}}},
    ]
    async for row in db["reels"].aggregate(pipeline):
        reel_counts[row["_id"]] = row["count"]

    results: list[dict] = []
    for coid in eligible_oids:
        profile = profiles_by_oid[coid]
        results.append({
            "creator_id": str(coid),
            "name": profile.get("name") or "Unknown",
            "profile_pic": profile.get("profilePic") or "",
            "about": profile.get("about") or "",
            "reel_count": reel_counts.get(coid, 0),
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

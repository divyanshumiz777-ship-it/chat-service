"""List creators the user follows — MongoDB only, no Qdrant, no LLM.

Returns ACTUAL followed creators from `userreelpreferences.followed_creators`,
NOT recommendations. Each entry includes the creator's profile and reel count.
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
      - user has no followed creators
      - userreelpreferences doc missing
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

    total_prefs = await db["userreelpreferences"].count_documents({})
    logger.info("USER=%s MONGO_TOTAL=%d", user_id, total_prefs)

    pref = await db["userreelpreferences"].find_one(
        {"user_id": me_oid},
        {"followed_creators": 1},
    )
    if not pref:
        logger.info("USER=%s SKIP_REASON=no_preferences RETURNED=0", user_id)
        return []

    creator_oids = list(oid_set(pref.get("followed_creators")))
    if not creator_oids:
        logger.info("USER=%s SKIP_REASON=no_followed_creators RETURNED=0", user_id)
        return []

    logger.info("USER=%s FILTERED=%d followed_creators", user_id, len(creator_oids))

    # Bulk-fetch creator profiles.
    creator_docs = await db["users"].find(
        {"_id": {"$in": creator_oids[:limit]}},
        {"name": 1, "profilePic": 1, "about": 1},
    ).to_list(length=limit)

    profiles_by_oid: dict = {doc["_id"]: doc for doc in creator_docs}

    # Reel counts per creator.
    reel_counts: dict = {}
    if creator_oids:
        pipeline = [
            {
                "$match": {
                    "creator_id": {"$in": creator_oids[:limit]},
                    "isDeleted": {"$ne": True},
                    "isPublic": {"$ne": False},
                }
            },
            {"$group": {"_id": "$creator_id", "count": {"$sum": 1}}},
        ]
        async for row in db["reels"].aggregate(pipeline):
            reel_counts[row["_id"]] = row["count"]

    results: list[dict] = []
    for coid in creator_oids[:limit]:
        profile = profiles_by_oid.get(coid)
        if not profile:
            logger.debug("USER=%s SKIP_REASON=profile_not_found creator=%s", user_id, coid)
            continue
        results.append({
            "creator_id": str(coid),
            "name": profile.get("name") or "Unknown",
            "profile_pic": profile.get("profilePic") or "",
            "about": profile.get("about") or "",
            "reel_count": reel_counts.get(coid, 0),
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

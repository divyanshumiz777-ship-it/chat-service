"""Trending creators — MongoDB only, no Qdrant, no LLM.

Ranks creators by recent follower growth (new accepted follows in the last
`_TRENDING_WINDOW_DAYS` days) using the real Follow graph — this is only
possible now that Follow.ts exists; there was no follow data to rank by
before. Falls back to lifetime follower count when nothing has grown
recently (e.g. right after the follow graph launches) so the command
doesn't return an empty list forever.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.trending_creators")

_TRENDING_WINDOW_DAYS = 7


async def get_trending_creators(user_id: str, limit: int = 10) -> list[dict]:
    """Return up to `limit` trending creators (highest recent growth first). Never raises."""
    try:
        return await _get_trending_creators_impl(user_id, limit)
    except Exception:
        logger.exception("get_trending_creators failed for user %s", user_id)
        return []


async def _rank_by_new_followers(db, since: datetime | None, limit: int) -> list[dict]:
    match: dict = {"status": "accepted"}
    if since is not None:
        match["createdAt"] = {"$gte": since}
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$followee", "new_followers": {"$sum": 1}}},
        {"$sort": {"new_followers": -1}},
        {"$limit": limit * 3},  # buffer for self/block filtering
    ]
    return [row async for row in db["follows"].aggregate(pipeline)]


async def _get_trending_creators_impl(user_id: str, limit: int) -> list[dict]:
    db = get_db()
    me_oid = to_oid(user_id)

    my_blocked: set = set()
    if me_oid:
        me = await db["users"].find_one({"_id": me_oid}, {"blockedUsers": 1})
        my_blocked = oid_set(me.get("blockedUsers")) if me else set()

    since = datetime.now(tz=timezone.utc) - timedelta(days=_TRENDING_WINDOW_DAYS)
    ranked = await _rank_by_new_followers(db, since, limit)
    if not ranked:
        ranked = await _rank_by_new_followers(db, None, limit)
    if not ranked:
        logger.info("USER=%s RETURNED=0 SKIP_REASON=no_follow_data", user_id)
        return []

    candidate_ids = [row["_id"] for row in ranked if row.get("_id")]

    reverse_blocked: set = set()
    if me_oid:
        async for u in db["users"].find({"blockedUsers": me_oid}, {"_id": 1}):
            reverse_blocked.add(u["_id"])

    excluded = my_blocked | reverse_blocked | ({me_oid} if me_oid else set())
    eligible_ids = [c for c in candidate_ids if c not in excluded][:limit]
    if not eligible_ids:
        return []

    creator_docs = await db["users"].find(
        {"_id": {"$in": eligible_ids}},
        {"name": 1, "profilePic": 1, "about": 1},
    ).to_list(length=len(eligible_ids))
    profiles = {d["_id"]: d for d in creator_docs}
    growth_by_id = {row["_id"]: row["new_followers"] for row in ranked}

    results: list[dict] = []
    for cid in eligible_ids:
        profile = profiles.get(cid)
        if not profile:
            continue
        growth = growth_by_id.get(cid, 0)
        reason = (
            f"{growth} new follower{'s' if growth != 1 else ''} this week"
            if growth > 0
            else "among the platform's most-followed creators"
        )
        results.append({
            "creator_id": str(cid),
            "name": profile.get("name") or "Unknown",
            "profile_pic": profile.get("profilePic") or "",
            "about": profile.get("about") or "",
            "new_followers": growth,
            "reasons": [reason],
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

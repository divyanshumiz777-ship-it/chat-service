"""Community recommender — MongoDB only (no Qdrant, no semantic search).

Suggests communities (DiscordServers) the user may want to join, scored from:

  +5 per friend already inside   (friend ∈ `members.user`)
  +4 shared interests            (user's preferred tags appear in name/description)
  +3 per shared category         (community categories ∩ the user's communities' categories)
  +2 active community            (recent activity or members online)
  +1 trending community          (member count near the top of the candidate set)

Excludes: communities the user already joined and deleted communities.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import is_recent, oid_set, top_n, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.community_recommender")

_MAX_CANDIDATES = 200


async def suggest_communities(user_id: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` community suggestions (highest score first). Never raises."""
    db = get_db()
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    me = await db["users"].find_one({"_id": me_oid}, {"servers": 1, "friends": 1})
    if not me:
        return []

    my_servers = oid_set(me.get("servers"))
    my_friends = oid_set(me.get("friends"))

    pref = await db["userreelpreferences"].find_one(
        {"user_id": me_oid}, {"preferred_tags": 1}
    )
    my_tags = {t.lower() for t in (pref.get("preferred_tags") or [])} if pref else set()

    # Categories of the communities the user already belongs to.
    my_categories: set = set()
    if my_servers:
        async for s in db["discordservers"].find(
            {"_id": {"$in": list(my_servers)}}, {"categories": 1}
        ):
            my_categories |= oid_set(s.get("categories"))

    # ── Candidate pool: public, not already joined, not deleted ──────────────
    query: dict = {"visibility": "public", "isDeleted": {"$ne": True}}
    if my_servers:
        query["_id"] = {"$nin": list(my_servers)}

    cands: list[dict] = []
    async for s in db["discordservers"].find(
        query,
        {"name": 1, "description": 1, "members": 1, "categories": 1,
         "onlineCount": 1, "updatedAt": 1},
    ).sort("updatedAt", -1).limit(_MAX_CANDIDATES):
        cands.append(s)
    if not cands:
        return []

    member_counts = {s["_id"]: len(s.get("members") or []) for s in cands}
    max_members = max(member_counts.values()) if member_counts else 0

    # ── Score ─────────────────────────────────────────────────────────────────
    results: list[dict] = []
    for s in cands:
        sid = s["_id"]
        score = 0
        reasons: list[str] = []

        member_ids = oid_set(m.get("user") for m in (s.get("members") or []))
        n = len(my_friends & member_ids)
        if n:
            score += 5 * n
            reasons.append(f"{'1 friend is' if n == 1 else f'{n} friends are'} already inside")

        text = f"{s.get('name', '')} {s.get('description', '')}".lower()
        matched = [t for t in my_tags if t and t in text]
        if matched:
            score += 4
            reasons.append("matches your interests (" + ", ".join(matched[:3]) + ")")

        n = len(my_categories & oid_set(s.get("categories")))
        if n:
            score += 3 * n
            reasons.append("1 shared category" if n == 1 else f"{n} shared categories")

        if (s.get("onlineCount") or 0) > 0 or is_recent(s.get("updatedAt"), 14):
            score += 2
            reasons.append("active community")

        mc = member_counts.get(sid, 0)
        if mc > 0 and mc >= max(2, max_members * 0.5):
            score += 1
            reasons.append("trending")

        if score > 0:
            results.append({
                "community_id": str(sid),
                "name": s.get("name", "Unknown"),
                "description": s.get("description", ""),
                "score": score,
                "reasons": reasons,
            })

    return top_n(results, limit)

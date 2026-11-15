"""Creator recommender — MongoDB only (no Qdrant, no semantic search).

Suggests reel creators the user may want to follow, scored from:

  +5 creator affinity        (you positively engaged with their reels)
  +4 watched similar reels    (their reel tags overlap tags of reels you watched)
  +3 shared tags              (their reel tags overlap your preferred tags)
  +2 high engagement creator  (high aggregate likes+shares across their reels)
  +1 trending creator         (a recent reel with real views)

Excludes: creators you already follow, muted creators, blocked users, and self.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import active_account_filter, is_recent, oid_set, top_n, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.creator_recommender")

_MAX_REELS_SCAN = 5000
_POSITIVE = {
    "$or": [
        {"liked": True}, {"saved": True}, {"completed": True},
        {"shared": True}, {"follow_creator": True},
    ]
}


async def suggest_creators(user_id: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` creator suggestions (highest score first). Never raises."""
    db = get_db()
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    me = await db["users"].find_one({"_id": me_oid}, {"blockedUsers": 1})
    if not me:
        return []
    my_blocked = oid_set(me.get("blockedUsers"))

    pref = await db["userreelpreferences"].find_one(
        {"user_id": me_oid},
        {"preferred_tags": 1, "muted_creators": 1},
    )
    my_tags = {t.lower() for t in (pref.get("preferred_tags") or [])} if pref else set()
    muted = oid_set(pref.get("muted_creators")) if pref else set()

    # Real follow graph (Follow.ts) — userreelpreferences.followed_creators
    # was a dead field with no writer anywhere in the codebase.
    following: set = set()
    async for doc in db["follows"].find({"follower": me_oid}, {"followee": 1}):
        f = to_oid(doc.get("followee"))
        if f:
            following.add(f)

    excluded = {me_oid} | following | muted | my_blocked

    # ── What I've watched: reel ids → their creators + tags ──────────────────
    my_reel_ids: set = set()
    async for it in db["userreelinteractions"].find(
        {"user_id": me_oid, **_POSITIVE}, {"reel_id": 1}
    ):
        r = to_oid(it.get("reel_id"))
        if r:
            my_reel_ids.add(r)

    my_watch_tags: set = set()
    affinity: dict = defaultdict(int)  # creator_oid -> count of my positive reels
    if my_reel_ids:
        async for r in db["reels"].find(
            {"_id": {"$in": list(my_reel_ids)}}, {"creator_id": 1, "tags": 1}
        ):
            for t in (r.get("tags") or []):
                my_watch_tags.add(str(t).lower())
            c = to_oid(r.get("creator_id"))
            if c:
                affinity[c] += 1

    # ── Candidate creators from public, non-deleted reels ────────────────────
    stats: dict = defaultdict(
        lambda: {"likes": 0, "shares": 0, "views": 0, "tags": set(), "recent": False}
    )
    async for r in db["reels"].find(
        {"isDeleted": {"$ne": True}, "isPublic": {"$ne": False}},
        {"creator_id": 1, "tags": 1, "likeCount": 1, "shareCount": 1,
         "viewCount": 1, "created_at": 1},
    ).limit(_MAX_REELS_SCAN):
        c = to_oid(r.get("creator_id"))
        if not c or c in excluded:
            continue
        st = stats[c]
        st["likes"] += int(r.get("likeCount") or 0)
        st["shares"] += int(r.get("shareCount") or 0)
        st["views"] += int(r.get("viewCount") or 0)
        for t in (r.get("tags") or []):
            st["tags"].add(str(t).lower())
        if is_recent(r.get("created_at"), 14):
            st["recent"] = True

    if not stats:
        return []

    max_engagement = max((s["likes"] + s["shares"] for s in stats.values()), default=0)

    names: dict = {}
    async for u in db["users"].find(
        {"_id": {"$in": list(stats.keys())}, **active_account_filter()},
        {"name": 1, "profilePic": 1, "blockedUsers": 1},
    ):
        # Reverse-block: this candidate has blocked me. My_blocked (candidate
        # blocked by me) is already excluded earlier via `excluded`.
        if me_oid in oid_set(u.get("blockedUsers")):
            continue
        names[u["_id"]] = u

    # ── Score ─────────────────────────────────────────────────────────────────
    results: list[dict] = []
    for cid, st in stats.items():
        if cid not in names:
            continue
        score = 0
        reasons: list[str] = []

        aff = affinity.get(cid, 0)
        if aff:
            score += 5 * aff
            reasons.append("you engage with their reels")

        if my_watch_tags & st["tags"]:
            score += 4
            reasons.append("makes reels like ones you watch")

        shared = my_tags & st["tags"]
        if shared:
            score += 3
            reasons.append("shares your interests (" + ", ".join(list(shared)[:3]) + ")")

        engagement = st["likes"] + st["shares"]
        if engagement > 0 and engagement >= max(1, max_engagement * 0.5):
            score += 2
            reasons.append("highly engaging creator")

        if st["recent"] and st["views"] > 0:
            score += 1
            reasons.append("trending recently")

        if score > 0:
            u = names.get(cid, {})
            results.append({
                "creator_id": str(cid),
                "name": u.get("name", "Unknown"),
                "profile_pic": u.get("profilePic", ""),
                "score": score,
                "reasons": reasons,
            })

    return top_n(results, limit)

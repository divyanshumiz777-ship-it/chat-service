"""Friend recommender — MongoDB only (no Qdrant, no semantic search).

Suggests people the user may want to connect with, scored from explicit social
and interest signals read directly from MongoDB:

  +5 per mutual community     (shared `users.servers`)
  +4 per mutual friend        (shared `users.friends`)
  +3 shared reel interests    (reels both engaged with positively)
  +3 shared creators          (overlap in the real Follow graph)
  +2 per shared tag           (shared `preferred_tags`)
  +1 similar activity         (recently active)

Excludes: the current user, existing friends, blocked users (either direction),
and users with a pending friend request (sent or received).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import active_account_filter, is_recent, oid_set, plural, to_oid, top_n

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.friend_recommender")

_MAX_CANDIDATES = 200
_POSITIVE = {"$or": [{"liked": True}, {"saved": True}, {"completed": True}, {"shared": True}]}


async def _positive_reels(db, user_filter) -> dict:
    """Map {user_oid: {reel_oid, ...}} of positively-interacted reels."""
    out: dict = defaultdict(set)
    cursor = db["userreelinteractions"].find(
        {**user_filter, **_POSITIVE}, {"user_id": 1, "reel_id": 1}
    )
    async for it in cursor:
        u = to_oid(it.get("user_id"))
        r = to_oid(it.get("reel_id"))
        if u and r:
            out[u].add(r)
    return out


async def suggest_friends(user_id: str, limit: int = 5) -> list[dict]:
    """Return up to `limit` friend suggestions (highest score first).

    Never raises — any unexpected error degrades to an empty list so the caller
    renders a friendly "no suggestions yet" message instead of surfacing an
    "assistant unavailable" error to the user.
    """
    try:
        return await _suggest_friends_impl(user_id, limit)
    except Exception:  # noqa: BLE001 — graceful degradation boundary
        logger.exception("suggest_friends failed for user %s", user_id)
        return []


async def _suggest_friends_impl(user_id: str, limit: int) -> list[dict]:
    """Core scoring logic. May raise on unexpected data shapes; wrapped above."""
    db = get_db()
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    me = await db["users"].find_one(
        {"_id": me_oid}, {"friends": 1, "blockedUsers": 1, "servers": 1}
    )
    if not me:
        return []

    my_friends = oid_set(me.get("friends"))
    my_servers = oid_set(me.get("servers"))
    my_blocked = oid_set(me.get("blockedUsers"))

    pref = await db["userreelpreferences"].find_one(
        {"user_id": me_oid}, {"preferred_tags": 1}
    )
    my_tags = set(pref.get("preferred_tags") or []) if pref else set()

    # Real follow graph (Follow.ts) — userreelpreferences.followed_creators
    # was a dead field with no writer anywhere in the codebase.
    my_creators: set = set()
    async for doc in db["follows"].find(
        {"follower": me_oid, "status": "accepted"}, {"followee": 1}
    ):
        f = to_oid(doc.get("followee"))
        if f:
            my_creators.add(f)

    my_reels = (await _positive_reels(db, {"user_id": me_oid})).get(me_oid, set())

    # ── Exclusions ──────────────────────────────────────────────────────────
    excluded = {me_oid} | my_friends | my_blocked
    async for fr in db["friendrequests"].find(
        {"status": "pending", "$or": [{"sender": me_oid}, {"receiver": me_oid}]},
        {"sender": 1, "receiver": 1},
    ):
        for key in ("sender", "receiver"):
            o = to_oid(fr.get(key))
            if o and o != me_oid:
                excluded.add(o)
    async for u in db["users"].find({"blockedUsers": me_oid}, {"_id": 1}):
        excluded.add(u["_id"])  # users who blocked me

    # ── Candidate generation (targeted, not a full scan) ─────────────────────
    cand: set = set()
    if my_servers:
        async for u in db["users"].find({"servers": {"$in": list(my_servers)}}, {"_id": 1}):
            cand.add(u["_id"])
    if my_friends:
        async for u in db["users"].find({"friends": {"$in": list(my_friends)}}, {"_id": 1}):
            cand.add(u["_id"])
        async for u in db["users"].find({"_id": {"$in": list(my_friends)}}, {"friends": 1}):
            cand |= oid_set(u.get("friends"))
    if my_tags:
        async for p in db["userreelpreferences"].find(
            {"preferred_tags": {"$in": list(my_tags)}}, {"user_id": 1}
        ):
            o = to_oid(p.get("user_id"))
            if o:
                cand.add(o)
    if my_creators:
        async for doc in db["follows"].find(
            {"followee": {"$in": list(my_creators)}, "status": "accepted"}, {"follower": 1}
        ):
            f = to_oid(doc.get("follower"))
            if f:
                cand.add(f)

    cand -= excluded
    if not cand:
        return []
    cand_list = list(cand)[:_MAX_CANDIDATES]

    # ── Bulk-load candidate signals ──────────────────────────────────────────
    users_map: dict = {}
    async for u in db["users"].find(
        {"_id": {"$in": cand_list}, **active_account_filter()},
        {"name": 1, "profilePic": 1, "friends": 1, "servers": 1, "lastSeen": 1},
    ):
        users_map[u["_id"]] = u

    prefs_map: dict = {}
    async for p in db["userreelpreferences"].find(
        {"user_id": {"$in": cand_list}},
        {"user_id": 1, "preferred_tags": 1},
    ):
        o = to_oid(p.get("user_id"))
        if o:
            prefs_map[o] = p

    cand_creators_map: dict = defaultdict(set)
    if my_creators:
        async for doc in db["follows"].find(
            {"follower": {"$in": cand_list}, "status": "accepted"},
            {"follower": 1, "followee": 1},
        ):
            f = to_oid(doc.get("follower"))
            c = to_oid(doc.get("followee"))
            if f and c:
                cand_creators_map[f].add(c)

    cand_reels = await _positive_reels(db, {"user_id": {"$in": cand_list}}) if my_reels else {}

    # ── Score ─────────────────────────────────────────────────────────────────
    results: list[dict] = []
    for cid in cand_list:
        u = users_map.get(cid)
        if not u:
            continue
        score = 0
        reasons: list[str] = []

        n = len(my_servers & oid_set(u.get("servers")))
        if n:
            score += 5 * n
            reasons.append("1 mutual community" if n == 1 else f"{n} mutual communities")
        n = len(my_friends & oid_set(u.get("friends")))
        if n:
            score += 4 * n
            reasons.append(plural(n, "mutual friend"))
        n = len(my_reels & cand_reels.get(cid, set()))
        if n:
            score += 3 * n
            reasons.append("shared reel interests")
        p = prefs_map.get(cid) or {}
        n = len(my_creators & cand_creators_map.get(cid, set()))
        if n:
            score += 3 * n
            reasons.append("follows creators you follow")
        n = len(my_tags & set(p.get("preferred_tags") or []))
        if n:
            score += 2 * n
            reasons.append(plural(n, "shared interest"))
        if is_recent(u.get("lastSeen"), 14):
            score += 1
            reasons.append("recently active")

        if score > 0:
            results.append({
                "user_id": str(cid),
                "name": u.get("name", "Unknown"),
                "profile_pic": u.get("profilePic", ""),
                "score": score,
                "reasons": reasons,
            })

    return top_n(results, limit)

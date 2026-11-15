"""Global entity search for platform users — MongoDB only, no Qdrant, no LLM.

Powers "users", "list users", "who is on the platform", and "find <name>".
Returns ACTUAL platform users, ACL-filtered, NEVER invented.

ACL rules (Part 9 — never expose private data):
  • Exclude the caller themselves.
  • Exclude users the caller blocked, and users who blocked the caller.
  • Visibility: a user appears only if their profileVisibility is "public"
    (or unset, which defaults to public) OR they are already the caller's friend.
    "private" and "friends"-only profiles never surface to non-friends.

Each result carries: name, avatar, mutual-community count, friend status,
creator badge (derived from owning ≥1 public reel), and a profile link.
"""

from __future__ import annotations

import logging
import re

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import active_account_filter, oid_set, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.user_search")

_MAX_FETCH = 60  # candidate cap before final trim to `limit`


async def search_users(user_id: str, term: str | None = None, limit: int = 10) -> list[dict]:
    """Return platform users matching `term` (or recent users when term is None).

    Never raises — any unexpected error degrades to [] so the caller renders the
    "couldn't find any matching data" fallback instead of an error.
    """
    try:
        return await _search_users_impl(user_id, term, limit)
    except Exception:
        logger.exception("search_users failed for user %s (term=%r)", user_id, term)
        return []


async def _search_users_impl(user_id: str, term: str | None, limit: int) -> list[dict]:
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    db = get_db()

    total_users = await db["users"].estimated_document_count()
    logger.info("USER=%s TERM=%r MONGO_TOTAL=%d", user_id, term, total_users)

    me = await db["users"].find_one(
        {"_id": me_oid}, {"friends": 1, "servers": 1, "blockedUsers": 1}
    )
    if not me:
        logger.info("USER=%s SKIP_REASON=user_not_found", user_id)
        return []

    my_friends = oid_set(me.get("friends"))
    my_servers = oid_set(me.get("servers"))
    my_blocked = oid_set(me.get("blockedUsers"))

    # Reverse blocks: users who blocked the caller must never surface.
    excluded = {me_oid} | my_blocked
    async for u in db["users"].find({"blockedUsers": me_oid}, {"_id": 1}):
        excluded.add(u["_id"])

    # ── Build the query ──────────────────────────────────────────────────────────
    # Visibility gate: public (or unset) OR an existing friend of the caller.
    visibility_or = [
        {"settings.privacy.profileVisibility": {"$nin": ["private", "friends"]}},
    ]
    if my_friends:
        visibility_or.append({"_id": {"$in": list(my_friends)}})

    query: dict = {
        "_id": {"$nin": list(excluded)},
        "$or": visibility_or,
        **active_account_filter(),
    }
    if term:
        query["name"] = {"$regex": re.escape(term.strip()), "$options": "i"}

    projection = {
        "name": 1,
        "profilePic": 1,
        "about": 1,
        "customStatus": 1,
        "servers": 1,
        "lastSeen": 1,
    }

    docs = await (
        db["users"]
        .find(query, projection)
        .sort("lastSeen", -1)
        .limit(_MAX_FETCH)
        .to_list(length=_MAX_FETCH)
    )

    logger.info("USER=%s FILTERED=%d", user_id, len(docs))

    if not docs:
        logger.info("USER=%s RETURNED=0 SKIP_REASON=no_match", user_id)
        return []

    docs = docs[:limit]
    result_oids = [d["_id"] for d in docs]

    # Creator badge: a user is a creator iff they own at least one public reel.
    creator_oids: set = set()
    creator_ids = await db["reels"].distinct(
        "creator_id",
        {"creator_id": {"$in": result_oids}, "isDeleted": {"$ne": True}},
    )
    creator_oids = oid_set(creator_ids)

    # Pending friend requests in either direction → "pending" status.
    pending: set = set()
    async for fr in db["friendrequests"].find(
        {
            "status": "pending",
            "$or": [
                {"sender": me_oid, "receiver": {"$in": result_oids}},
                {"receiver": me_oid, "sender": {"$in": result_oids}},
            ],
        },
        {"sender": 1, "receiver": 1},
    ):
        for key in ("sender", "receiver"):
            o = to_oid(fr.get(key))
            if o and o != me_oid:
                pending.add(o)

    results: list[dict] = []
    for doc in docs:
        uid = doc["_id"]
        if uid in my_friends:
            friend_status = "friend"
        elif uid in pending:
            friend_status = "pending"
        else:
            friend_status = "none"

        status_text = ""
        cs = doc.get("customStatus") or {}
        if isinstance(cs, dict):
            status_text = (cs.get("text") or "").strip()

        results.append({
            "user_id": str(uid),
            "name": doc.get("name") or "Unknown",
            "profile_pic": doc.get("profilePic") or "",
            "about": (doc.get("about") or "").strip(),
            "custom_status": status_text,
            "mutual_communities": len(my_servers & oid_set(doc.get("servers"))),
            "friend_status": friend_status,
            "is_creator": uid in creator_oids,
            "profile_url": f"/profile/{uid}",
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

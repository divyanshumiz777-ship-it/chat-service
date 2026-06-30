"""List the communities the user has joined — MongoDB only, no Qdrant, no LLM.

Returns ACTUAL memberships from `discordservers.members.user`, NOT recommendations.
Each entry includes the server name, the user's role, member count, and visibility.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.list_communities")


def _extract_role(doc: dict, me_oid: Any) -> str:
    """Determine the user's highest role in the server (owner > admin > moderator > member)."""
    if doc.get("owner") == me_oid:
        return "owner"
    for m in doc.get("members") or []:
        if m.get("user") == me_oid:
            roles: list[str] = [r.lower() for r in (m.get("roles") or [])]
            if "admin" in roles:
                return "admin"
            if "moderator" in roles or "mod" in roles:
                return "moderator"
            return roles[0] if roles else "member"
    return "member"


async def get_my_communities(user_id: str, limit: int = 20) -> list[dict]:
    """Return all communities the user has joined. Never raises.

    Returns []:
      - user has joined no communities
      - unexpected error (logged)
    """
    try:
        return await _get_my_communities_impl(user_id, limit)
    except Exception:
        logger.exception("get_my_communities failed for user %s", user_id)
        return []


async def _get_my_communities_impl(user_id: str, limit: int) -> list[dict]:
    me_oid = to_oid(user_id)
    if not me_oid:
        return []

    db = get_db()

    total = await db["discordservers"].count_documents({})
    logger.info("USER=%s MONGO_TOTAL=%d", user_id, total)

    docs = await db["discordservers"].find(
        {"members.user": me_oid},
        {
            "name": 1,
            "description": 1,
            "visibility": 1,
            "createdAt": 1,
            "imageUrl": 1,
            "owner": 1,
            "members": 1,
        },
    ).limit(limit).to_list(length=limit)

    logger.info("USER=%s FILTERED=%d RETURNED=%d", user_id, total, len(docs))

    results: list[dict] = []
    for doc in docs:
        members = doc.get("members") or []
        role = _extract_role(doc, me_oid)

        # member count from embedded array length
        member_count = len(members)

        created_at = doc.get("createdAt")
        created_str = ""
        if created_at:
            try:
                created_str = created_at.strftime("%-d %b %Y")
            except Exception:
                created_str = str(created_at)[:10]

        results.append({
            "community_id": str(doc["_id"]),
            "name": doc.get("name") or "Unnamed",
            "description": doc.get("description") or "",
            "visibility": doc.get("visibility") or "public",
            "member_count": member_count,
            "role": role,
            "created_at": created_str,
            "image_url": doc.get("imageUrl") or "",
            "is_owner": doc.get("owner") == me_oid,
        })

    logger.info("USER=%s RETURNED=%d", user_id, len(results))
    return results

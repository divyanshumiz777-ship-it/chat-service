"""Look up a specific community the user belongs to — MongoDB only, no Qdrant.

Thin wrapper around list_communities.get_my_communities().
When a name is given, filters the user's memberships for that community.
When no name is given, returns all memberships (same as get_my_communities).
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.tools.list_communities import get_my_communities

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.community_lookup")


async def lookup_my_community(
    user_id: str,
    name: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return communities the user belongs to, optionally filtered by name.

    Never raises — any error from get_my_communities() is already caught there.
    If name is given, does case-insensitive substring matching against community names.
    """
    communities = await get_my_communities(user_id, limit=limit)

    if not name:
        return communities

    needle = name.lower().strip()
    matched = [c for c in communities if needle in c.get("name", "").lower()]
    logger.info(
        "USER=%s COMMUNITY_LOOKUP name=%r TOTAL=%d MATCHED=%d",
        user_id, name, len(communities), len(matched),
    )
    return matched

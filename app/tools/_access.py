"""Shared access + name-resolution helpers for the server/channel tools.

Membership here mirrors the backend's own access checks (server.ts
join-server / canJoinVoiceChannel): the server OWNER, or a ServerMember row
for (server, user) that is not banned. The embedded DiscordServer.members array
is not authoritative (seed scripts and some flows only write one of the two).

Name resolution: channel and community names are NOT unique across the
platform ("general" exists in nearly every server). Resolving "summarize
#general" with a bare find_one picked an arbitrary server's channel — often a
stranger's public server — or, if that happened to be a private server, told
the caller the channel didn't exist even though their own server has one.
Candidates are now ranked: servers the caller belongs to first, then public
servers, most recently active first within each group. Servers the caller
can't see are never returned.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_MAX_CANDIDATES = 50


def _name_filter(name: str) -> dict:
    return {"$regex": f"^{re.escape(name.strip())}$", "$options": "i"}


async def member_server_oids(db: Any, user_oid: Any, server_oids: Iterable[Any]) -> set:
    """Subset of `server_oids` the caller belongs to (owner or non-banned member)."""
    ids = [s for s in server_oids if s is not None]
    if user_oid is None or not ids:
        return set()
    owned = await db["discordservers"].find(
        {"_id": {"$in": ids}, "owner": user_oid}, {"_id": 1}
    ).to_list(length=None)
    rows = await db["servermembers"].find(
        {"server": {"$in": ids}, "user": user_oid, "banned.isBanned": {"$ne": True}},
        {"server": 1},
    ).to_list(length=None)
    return {d["_id"] for d in owned} | {r["server"] for r in rows}


def can_view(server: dict | None, member_oids: set) -> bool:
    """Public servers are visible to everyone; others only to members."""
    if not server:
        return False
    return server.get("visibility") == "public" or server.get("_id") in member_oids


def _recency(doc: dict) -> Any:
    from datetime import datetime, timezone

    return doc.get("updatedAt") or doc.get("createdAt") or datetime.min.replace(tzinfo=timezone.utc)


def _sort_key(doc: dict) -> float:
    value = _recency(doc)
    try:
        return value.timestamp()
    except Exception:  # noqa: BLE001 — non-datetime values sort last
        return float("-inf")


async def resolve_server_by_name(
    db: Any, user_oid: Any, name: str, projection: dict
) -> tuple[dict | None, bool]:
    """Best visible server named `name`: (server, caller_is_member)."""
    proj = {**projection, "visibility": 1, "owner": 1, "updatedAt": 1}
    candidates = await db["discordservers"].find(
        {"name": _name_filter(name)}, proj
    ).to_list(length=_MAX_CANDIDATES)
    if not candidates:
        return None, False
    members = await member_server_oids(db, user_oid, [c["_id"] for c in candidates])
    mine = sorted((c for c in candidates if c["_id"] in members), key=_sort_key, reverse=True)
    if mine:
        return mine[0], True
    public = sorted(
        (c for c in candidates if c.get("visibility") == "public"), key=_sort_key, reverse=True
    )
    return (public[0], False) if public else (None, False)


async def resolve_channel_by_name(
    db: Any,
    user_oid: Any,
    channel_name: str,
    *,
    community_name: str | None = None,
    channel_projection: dict | None = None,
    prefer_types: tuple[str, ...] = (),
) -> tuple[dict, dict, bool] | None:
    """Best visible channel named `channel_name`: (channel, server, caller_is_member).

    Ranking: caller's own servers before public ones; within a group, channels
    of a preferred type (e.g. Voice/Video for voice recall) first, then the
    most recently active. Returns None when no visible match exists.
    """
    ch_filter: dict = {"name": _name_filter(channel_name)}
    if community_name:
        server, _ = await resolve_server_by_name(db, user_oid, community_name, {"name": 1})
        if server is None:
            return None
        ch_filter["server"] = server["_id"]

    proj = {**(channel_projection or {}), "name": 1, "server": 1, "type": 1, "updatedAt": 1}
    channels = await db["channels"].find(ch_filter, proj).to_list(length=_MAX_CANDIDATES)
    if not channels:
        return None

    server_ids = list({c.get("server") for c in channels if c.get("server") is not None})
    servers = await db["discordservers"].find(
        {"_id": {"$in": server_ids}}, {"name": 1, "visibility": 1, "owner": 1, "updatedAt": 1}
    ).to_list(length=None)
    by_id = {s["_id"]: s for s in servers}
    members = await member_server_oids(db, user_oid, server_ids)

    def rank(ch: dict) -> tuple:
        server = by_id.get(ch.get("server"))
        is_member = server is not None and server["_id"] in members
        preferred = bool(prefer_types) and ch.get("type") in prefer_types
        return (1 if is_member else 0, 1 if preferred else 0, _sort_key(ch))

    visible = [c for c in channels if can_view(by_id.get(c.get("server")), members)]
    if not visible:
        return None
    best = max(visible, key=rank)
    server = by_id[best["server"]]
    return best, server, server["_id"] in members

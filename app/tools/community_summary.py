"""Community and channel summarization — MongoDB only, no Qdrant, no LLM.

Retrieves structured data used by the LLM to generate community or channel
summaries.  ACL: public communities are visible to all authenticated users;
private / invite-only communities are visible only to their members.  Channels
inherit their server's visibility rule.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._access import (
    can_view,
    member_server_oids,
    resolve_channel_by_name,
    resolve_server_by_name,
)
from app.tools._common import to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.community_summary")

_MAX_MESSAGES = 30
_MAX_THREADS = 10
_MAX_CHANNELS = 30
_TOPIC_COUNT = 5
_MIN_TOPIC_LEN = 4

_STOPWORDS = frozenset({
    "the", "and", "you", "your", "yours", "for", "are", "was", "were", "this",
    "that", "with", "have", "has", "had", "but", "not", "can", "will", "would",
    "should", "could", "from", "they", "them", "their", "what", "when", "where",
    "who", "why", "how", "about", "just", "like", "yeah", "yes", "okay", "ok",
    "hey", "hello", "there", "then", "into", "out", "got", "get", "going",
    "really", "much", "some", "any", "all", "now", "one", "good", "great",
    "lol", "haha", "gonna", "wanna",
})


# ── Structured result types ────────────────────────────────────────────────────

@dataclass
class CommunitySummaryData:
    server_id: str
    name: str
    description: str
    visibility: str
    member_count: int
    channel_names: list[str]
    recent_messages: list[dict]   # [{text, sender, channel_name, ts}]
    recent_threads: list[dict]    # [{title, ts}]
    topics: list[str]
    is_member: bool


@dataclass
class ChannelSummaryData:
    channel_id: str
    name: str
    server_name: str
    message_count: int
    recent_messages: list[dict]   # [{text, sender, ts}]
    pinned_messages: list[dict]   # [{text, sender, ts}]
    thread_count: int
    topics: list[str]
    is_quiet: bool                # True when message_count == 0


# ── Private helpers ────────────────────────────────────────────────────────────

def _msg_text(doc: dict) -> str:
    """Extract plain text from a Message document."""
    content = doc.get("content")
    if isinstance(content, dict):
        return (content.get("plainText") or content.get("formattedContent") or "").strip()
    if content:
        return str(content).strip()
    return str(doc.get("plainText") or doc.get("formattedContent") or "").strip()


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


def _extract_topics(texts: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()):
            if len(tok) >= _MIN_TOPIC_LEN and tok not in _STOPWORDS:
                counts[tok] += 1
    return [w for w, _ in counts.most_common(_TOPIC_COUNT)]


async def _load_names(db: Any, ids: list[Any]) -> dict[str, str]:
    unique = list({i for i in ids if i is not None})
    if not unique:
        return {}
    docs = await db["users"].find(
        {"_id": {"$in": unique}}, {"name": 1}
    ).to_list(length=len(unique))
    return {str(d["_id"]): (d.get("name") or "Unknown") for d in docs}


# Access checks live in app/tools/_access.py (owner or non-banned ServerMember,
# mirroring the backend) — shared with voice_session_recall.py.


# ── Community summary ──────────────────────────────────────────────────────────

async def get_community_summary(
    user_id: str, community_name: str | None
) -> CommunitySummaryData | None:
    """Return structured community data for LLM summarization.  Never raises.

    Returns None when the community does not exist or the caller lacks access.
    When community_name is None, falls back to the user's most-recently-active
    community.
    """
    try:
        return await _community_summary_impl(user_id, community_name)
    except Exception:
        logger.exception(
            "get_community_summary failed user=%s name=%r", user_id, community_name
        )
        return None


async def _community_summary_impl(
    user_id: str, community_name: str | None
) -> CommunitySummaryData | None:
    user_oid = to_oid(user_id)
    db = get_db()

    proj = {
        "name": 1, "description": 1, "visibility": 1, "owner": 1,
        "members": 1, "channels": 1, "createdAt": 1,
    }

    if community_name:
        # Names aren't unique: prefer a server the caller belongs to, then a
        # public one — never an arbitrary (possibly private) match.
        server, is_member = await resolve_server_by_name(db, user_oid, community_name, proj)
    else:
        if not user_oid:
            return None
        server = await db["discordservers"].find_one(
            {"members.user": user_oid},
            proj,
            sort=[("updatedAt", -1)],
        )
        member_oids = await member_server_oids(
            db, user_oid, [server["_id"]] if server else []
        )
        if not can_view(server, member_oids):
            return None
        is_member = server["_id"] in member_oids

    if server is None:
        return None

    server_oid = server["_id"]
    members = server.get("members") or []

    # Channels
    channel_oids = server.get("channels") or []
    channels_docs: list[dict] = []
    if channel_oids:
        channels_docs = await db["channels"].find(
            {"_id": {"$in": channel_oids}},
            {"name": 1, "type": 1},
        ).to_list(length=_MAX_CHANNELS)
    channel_names = [c.get("name", "unknown") for c in channels_docs]

    # Recent channel messages
    channel_id_oids = [c["_id"] for c in channels_docs]
    msg_filter: dict[str, Any] = {
        "server": server_oid,
        "deletedForEveryone": {"$ne": True},
        "conversationId": {"$exists": False},
        "groupId": {"$exists": False},
    }
    if channel_id_oids:
        msg_filter["channel"] = {"$in": channel_id_oids}

    raw_msgs = await db["messages"].find(
        msg_filter,
        {"content": 1, "plainText": 1, "formattedContent": 1,
         "sender": 1, "channel": 1, "createdAt": 1},
    ).sort("createdAt", -1).to_list(length=_MAX_MESSAGES)

    sender_ids = [m.get("sender") for m in raw_msgs]
    names = await _load_names(db, sender_ids)
    channel_name_map = {str(c["_id"]): c.get("name", "unknown") for c in channels_docs}

    recent_messages: list[dict] = []
    for m in raw_msgs:
        text = _msg_text(m)
        if not text:
            continue
        recent_messages.append({
            "text": text,
            "sender": names.get(str(m.get("sender")), "Unknown"),
            "channel_name": channel_name_map.get(str(m.get("channel", "")), "general"),
            "ts": _iso(m.get("createdAt")),
        })

    # Recent threads
    threads = await db["threads"].find(
        {"server": server_oid},
        {"title": 1, "createdAt": 1},
    ).sort("createdAt", -1).to_list(length=_MAX_THREADS)

    recent_threads = [
        {"title": t.get("title") or "Untitled thread", "ts": _iso(t.get("createdAt"))}
        for t in threads
    ]

    topics = _extract_topics([m["text"] for m in recent_messages])

    return CommunitySummaryData(
        server_id=str(server_oid),
        name=server.get("name") or "Unknown",
        description=server.get("description") or "",
        visibility=server.get("visibility") or "public",
        member_count=len(members),
        channel_names=channel_names,
        recent_messages=recent_messages,
        recent_threads=recent_threads,
        topics=topics,
        is_member=is_member,
    )


# ── Channel summary ────────────────────────────────────────────────────────────

async def get_channel_summary(
    user_id: str, channel_name: str, community_name: str | None = None
) -> ChannelSummaryData | None:
    """Return structured channel data for LLM summarization.  Never raises.

    Returns None when the channel does not exist or the caller lacks access.
    Optionally scoped to a specific community via community_name.
    """
    try:
        return await _channel_summary_impl(user_id, channel_name, community_name)
    except Exception:
        logger.exception(
            "get_channel_summary failed user=%s channel=%r", user_id, channel_name
        )
        return None


async def _channel_summary_impl(
    user_id: str, channel_name: str, community_name: str | None
) -> ChannelSummaryData | None:
    user_oid = to_oid(user_id)
    db = get_db()

    # Channel names repeat across servers ("general" is everywhere): resolve to
    # the caller's own server first, then a public one. Access is enforced by
    # the resolver — a channel whose server the caller can't see is never
    # returned (and a channel with no server document is no longer allowed).
    resolved = await resolve_channel_by_name(
        db,
        user_oid,
        channel_name,
        community_name=community_name,
        channel_projection={"thread": 1},
        prefer_types=("Text",),
    )
    if resolved is None:
        return None
    channel, server_doc, _is_member = resolved

    channel_oid = channel["_id"]
    server_name = server_doc.get("name") or "Unknown Server"

    # Messages in this channel
    raw_msgs = await db["messages"].find(
        {"channel": channel_oid, "deletedForEveryone": {"$ne": True}},
        {"content": 1, "plainText": 1, "formattedContent": 1,
         "sender": 1, "createdAt": 1, "pinned": 1},
    ).sort("createdAt", -1).to_list(length=_MAX_MESSAGES)

    sender_ids = [m.get("sender") for m in raw_msgs]
    names = await _load_names(db, sender_ids)

    all_msgs: list[dict] = []
    pinned_msgs: list[dict] = []
    for m in raw_msgs:
        text = _msg_text(m)
        if not text:
            continue
        entry = {
            "text": text,
            "sender": names.get(str(m.get("sender")), "Unknown"),
            "ts": _iso(m.get("createdAt")),
        }
        all_msgs.append(entry)
        if m.get("pinned"):
            pinned_msgs.append(entry)

    thread_count = len(channel.get("thread") or [])
    topics = _extract_topics([m["text"] for m in all_msgs])

    return ChannelSummaryData(
        channel_id=str(channel_oid),
        name=channel.get("name") or channel_name,
        server_name=server_name,
        message_count=len(all_msgs),
        recent_messages=all_msgs,
        pinned_messages=pinned_msgs,
        thread_count=thread_count,
        topics=topics,
        is_quiet=len(all_msgs) == 0,
    )

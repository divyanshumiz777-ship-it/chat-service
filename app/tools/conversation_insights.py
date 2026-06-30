"""Direct-message conversation insights — reads MongoDB only, no Qdrant.

Returns a ranked list (newest first) of the user's DM conversations, each with
the other participant's name/avatar, visible message count, latest message
snippet, and a lightweight deterministic summary generated from content patterns.

No semantic search, no Qdrant, no LLM invocations in this module.
The LLM in api/internal._format_conversation_insights() may format the output
but must never invent conversations that this tool did not return.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.conversation_insights")

# Messages fetched per conversation for summary generation (not the visible count).
_MAX_SCAN_MESSAGES = 30

# ── Summary signal patterns ────────────────────────────────────────────────────

_GREETING_PAT = re.compile(
    r"\b(hi+|hey+|hello+|howdy|sup|yo+|good\s+(morning|evening|afternoon|night))\b",
    re.IGNORECASE,
)
_WORK_WORDS = frozenset({
    "project", "code", "coding", "work", "deploy", "build", "develop",
    "bug", "fix", "feature", "git", "design", "meeting", "deadline",
    "server", "backend", "frontend", "api", "database", "pr", "commit",
    "review", "debug", "script", "task", "issue", "ticket",
})
_PLAN_WORDS = frozenset({
    "meet", "plan", "tomorrow", "tonight", "weekend", "later",
    "schedule", "come", "going", "outing", "movie", "party", "hang",
    "catch", "coffee", "lunch", "dinner", "trip", "visit",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg_text(doc: dict) -> str:
    content = doc.get("content")
    if isinstance(content, dict):
        return (content.get("plainText") or content.get("formattedContent") or "").strip()
    if content:
        return str(content).strip()
    return str(doc.get("plainText") or doc.get("formattedContent") or "").strip()


from datetime import datetime, timezone
from typing import Any


def _format_activity(dt: Any) -> str:
    """Human-readable relative timestamp (e.g. 'Today 3:57 PM')."""

    if not isinstance(dt, datetime):
        return ""

    now = datetime.now(tz=timezone.utc)

    dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    delta = now - dt_utc

    hour = dt_utc.strftime("%I").lstrip("0") or "12"
    t = f"{hour}:{dt_utc.strftime('%M %p')}"

    if delta.days == 0:
        return f"Today {t}"

    if delta.days == 1:
        return f"Yesterday {t}"

    if delta.days < 7:
        return f"{dt_utc.strftime('%A')} {t}"

    return dt_utc.strftime("%b %d")


def _summarize(messages: list[dict]) -> list[str]:
    """Deterministic bullet-point summary from message content patterns.

    Inspects message text and attachment types; produces 1–4 short bullets.
    No LLM, no Qdrant — pure in-memory heuristics.
    """
    if not messages:
        return []

    greeting_hits = 0
    question_hits = 0
    work_hits = 0
    plan_hits = 0
    media_types: set[str] = set()
    total_chars = 0

    for msg in messages:
        text = _msg_text(msg)
        lower = text.lower()
        words = frozenset(re.findall(r"\b\w+\b", lower))

        if _GREETING_PAT.search(lower):
            greeting_hits += 1
        if "?" in text:
            question_hits += 1
        if words & _WORK_WORDS:
            work_hits += 1
        if words & _PLAN_WORDS:
            plan_hits += 1

        for att in msg.get("attachmentsV2") or []:
            atype = (att.get("type") or "").strip().lower()
            if atype:
                media_types.add(atype)

        total_chars += len(text)

    n = len(messages)
    bullets: list[str] = []

    if greeting_hits >= max(1, n // 4):
        bullets.append("Greetings exchanged")

    if "gif" in media_types:
        bullets.append("GIFs shared")
    if {"image", "video"} & media_types:
        bullets.append("Media shared")
    if "sticker" in media_types:
        bullets.append("Stickers exchanged")

    if work_hits >= 2:
        bullets.append("Development or project discussion")

    if plan_hits >= 2:
        bullets.append("Plans or meetups mentioned")

    if question_hits >= max(1, n // 3):
        bullets.append("Questions and answers")

    if not bullets:
        avg = total_chars // max(n, 1)
        bullets.append("Casual conversation" if avg < 60 else "Extended discussion")

    return bullets[:4]


# ── Public API ─────────────────────────────────────────────────────────────────

async def get_conversation_insights(
    user_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return the user's most-recent DM conversations with insights.

    Access-controlled: only conversations where the caller is a participant.
    Excludes conversations the user deleted, empty conversations, and blocked
    contacts (both directions).

    Returns a list of dicts sorted by latest_activity DESC, each containing:
        conversation_id, friend_id, friend_name, friend_profile_pic,
        message_count, latest_message, latest_activity, participants, summary
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        user_oid = ObjectId(user_id)
    except (InvalidId, TypeError):
        logger.warning("invalid user_id for conversation insights: %r", user_id)
        return []

    db = get_db()

    # Load caller's blocked users (one-directional; reverse block checked below).
    user_doc = await db["users"].find_one({"_id": user_oid}, {"blockedUsers": 1})
    blocked_oids = oid_set(user_doc.get("blockedUsers") if user_doc else None)

    # Fetch conversations the caller participates in and has not deleted.
    # Over-fetch (limit * 3) to account for filtered-out entries.
    convs = await db["conversations"].find(
        {
            "participants": user_oid,
            "deletedFor": {"$nin": [user_oid]},
        },
        {
            "participants": 1,
            "hiddenFor": 1,
            "deletedFor": 1,
            "updatedAt": 1,
            "createdAt": 1,
        },
    ).sort("updatedAt", -1).to_list(length=limit * 3)

    # Map conv_id → friend ObjectId; collect all friend ids for bulk lookup.
    friend_id_map: dict[str, Any] = {}
    all_friend_oids: list[Any] = []

    for conv in convs:
        others = [p for p in conv.get("participants", []) if p != user_oid]
        if not others:
            continue
        friend_oid = others[0]
        if friend_oid in blocked_oids:
            continue
        friend_id_map[str(conv["_id"])] = friend_oid
        all_friend_oids.append(friend_oid)

    if not all_friend_oids:
        return []

    # Check which friends have blocked the current user (reverse-block).
    blocker_docs = await db["users"].find(
        {"_id": {"$in": all_friend_oids}, "blockedUsers": user_oid},
        {"_id": 1},
    ).to_list(length=len(all_friend_oids))
    reverse_blocked = {d["_id"] for d in blocker_docs}

    # Bulk-fetch friend profiles (only non-blocked friends).
    eligible_friend_oids = [
        v for v in friend_id_map.values() if v not in reverse_blocked
    ]
    if not eligible_friend_oids:
        return []

    friend_docs = await db["users"].find(
        {"_id": {"$in": eligible_friend_oids}},
        {"name": 1, "profilePic": 1},
    ).to_list(length=len(eligible_friend_oids))
    friend_profile: dict[str, dict] = {str(d["_id"]): d for d in friend_docs}

    results: list[dict] = []

    for conv in convs:
        if len(results) >= limit:
            break

        conv_id = conv["_id"]
        conv_id_str = str(conv_id)
        friend_oid = friend_id_map.get(conv_id_str)
        if friend_oid is None or friend_oid in reverse_blocked:
            continue
        profile = friend_profile.get(str(friend_oid))
        if not profile:
            continue

        msg_filter: dict[str, Any] = {
            "conversationId": conv_id,
            "deletedForEveryone": {"$ne": True},
            "deletedFor": {"$nin": [user_oid]},
        }

        # Accurate visible message count (fast index scan on conversationId).
        visible_count = await db["messages"].count_documents(msg_filter)
        if visible_count == 0:
            continue  # exclude empty conversations

        # Recent messages for summary and latest snippet.
        raw_msgs = await db["messages"].find(
            msg_filter,
            {
                "content": 1,
                "plainText": 1,
                "formattedContent": 1,
                "attachmentsV2": 1,
                "createdAt": 1,
            },
        ).sort("createdAt", -1).to_list(length=_MAX_SCAN_MESSAGES)

        latest_text = _msg_text(raw_msgs[0]) if raw_msgs else ""
        if len(latest_text) > 120:
            latest_text = latest_text[:117] + "…"

        # Participants list (display names, excluding caller).
        participant_names: list[str] = []
        for pid in conv.get("participants", []):
            if pid == user_oid:
                continue
            fp = friend_profile.get(str(pid))
            if fp:
                participant_names.append(fp.get("name") or "Unknown")

        results.append({
            "conversation_id": conv_id_str,
            "friend_id": str(friend_oid),
            "friend_name": profile.get("name") or "Unknown",
            "friend_profile_pic": profile.get("profilePic") or "",
            "message_count": visible_count,
            "latest_message": latest_text,
            "latest_activity": _format_activity(conv.get("updatedAt")),
            "participants": participant_names,
            "summary": _summarize(raw_msgs),
        })

    return results

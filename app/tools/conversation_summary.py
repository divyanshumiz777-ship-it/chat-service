"""Current-conversation summarizer — reads MongoDB directly.

Builds the structured data needed to summarize the conversation a user currently
has open (a 1:1 DM `Conversation` or a `Group` DM). Used by the
SUMMARIZE_CONVERSATION intent.

This tool deliberately does NOT use Qdrant or search_all() — conversation
content is read straight from the `conversations`/`groups` and `messages`
collections. The LLM only formats the prose summary afterwards (see
provider.generate_conversation_summary).

Access control: a conversation is only ever loaded when the caller is one of its
participants. The fallback path (no conversation_id given) queries exclusively
by `participants: <user>`, so a user can never reach a conversation they are not
part of.
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

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.conversation_summary")

_MAX_MESSAGES = 200            # transcript cap sent onward to the LLM
_TOPIC_COUNT = 6               # number of keyword "topics" to surface
_MIN_TOPIC_LEN = 4             # ignore short tokens when mining topics

# Minimal English stopword set for topic mining (kept local — no NLP dep).
_STOPWORDS = frozenset({
    "the", "and", "you", "your", "yours", "for", "are", "was", "were", "this",
    "that", "with", "have", "has", "had", "but", "not", "can", "will", "would",
    "should", "could", "from", "they", "them", "their", "what", "when", "where",
    "who", "why", "how", "about", "just", "like", "yeah", "yes", "okay", "ok",
    "hey", "hello", "hii", "hiii", "heyy", "lol", "haha", "gonna", "wanna",
    "there", "here", "then", "than", "into", "out", "got", "get", "going",
    "really", "much", "some", "any", "all", "now", "one", "good", "great",
})


# ── Structured result ───────────────────────────────────────────────────────

@dataclass
class ConversationSummaryData:
    conversation_id: str
    kind: str                       # "dm" | "group"
    participants: list[str]         # display names
    message_count: int
    latest_activity: str            # ISO 8601, or "" if unknown
    topics: list[str]
    transcript: list[dict] = field(default_factory=list)  # [{sender,text,ts}]
    title: str = ""                 # group name when kind == "group"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg_text(doc: dict) -> str:
    """Extract plain text from a Message (content may be str or {plainText})."""
    content = doc.get("content")
    if isinstance(content, dict):
        return (content.get("plainText") or content.get("formattedContent") or "").strip()
    if content:
        return str(content).strip()
    # Fall back to top-level plainText/formattedContent (schema allows both).
    return str(doc.get("plainText") or doc.get("formattedContent") or "").strip()


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else ""


def _extract_topics(texts: list[str]) -> list[str]:
    """Cheap keyword mining: most frequent non-stopword tokens across messages."""
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()):
            if len(tok) >= _MIN_TOPIC_LEN and tok not in _STOPWORDS:
                counts[tok] += 1
    return [word for word, _ in counts.most_common(_TOPIC_COUNT)]


async def _load_names(db: Any, ids: list[Any]) -> dict[str, str]:
    """Map {str(user_id): display name} for the given user ObjectIds."""
    unique = list({i for i in ids if i is not None})
    if not unique:
        return {}
    docs = await db["users"].find(
        {"_id": {"$in": unique}}, {"name": 1}
    ).to_list(length=len(unique))
    return {str(d["_id"]): (d.get("name") or "Unknown") for d in docs}


async def _resolve_conversation(
    db: Any, user_oid: Any, conversation_id: str | None
) -> tuple[dict | None, str]:
    """Locate the target conversation (DM or group), scoped to the caller.

    With an explicit id → that conversation/group, but only if the user is a
    participant. Without one → the user's most-recently-active conversation
    (DM first, then group).
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    if conversation_id:
        try:
            oid = ObjectId(conversation_id)
        except (InvalidId, TypeError):
            logger.warning("invalid conversation_id: %r", conversation_id)
            return None, ""
        conv = await db["conversations"].find_one(
            {"_id": oid, "participants": user_oid}
        )
        if conv:
            return conv, "dm"
        grp = await db["groups"].find_one({"_id": oid, "participants": user_oid})
        if grp:
            return grp, "group"
        return None, ""

    # Fallback: most recently updated conversation the user belongs to.
    conv = await db["conversations"].find_one(
        {"participants": user_oid}, sort=[("updatedAt", -1)]
    )
    if conv:
        return conv, "dm"
    grp = await db["groups"].find_one(
        {"participants": user_oid}, sort=[("updatedAt", -1)]
    )
    if grp:
        return grp, "group"
    return None, ""


# ── Public API ──────────────────────────────────────────────────────────────

async def build_conversation_summary(
    user_id: str, conversation_id: str | None = None
) -> ConversationSummaryData | None:
    """Assemble structured summary data for the user's conversation.

    Returns None when no accessible conversation is found (caller renders a
    friendly fallback message). Never raises for normal "not found" cases.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        user_oid = ObjectId(user_id)
    except (InvalidId, TypeError):
        logger.warning("invalid user_id for conversation summary: %r", user_id)
        return None

    db = get_db()
    conv, kind = await _resolve_conversation(db, user_oid, conversation_id)
    if not conv:
        return None

    conv_oid = conv["_id"]
    msg_filter: dict[str, Any] = {"deletedForEveryone": {"$ne": True}}
    if kind == "group":
        msg_filter["groupId"] = conv_oid
    else:
        msg_filter["conversationId"] = conv_oid

    raw_msgs = await db["messages"].find(msg_filter).sort("createdAt", 1).to_list(
        length=_MAX_MESSAGES * 2
    )
    # Drop messages this user deleted for themselves.
    raw_msgs = [m for m in raw_msgs if user_oid not in (m.get("deletedFor") or [])]

    participant_ids = conv.get("participants", [])
    sender_ids = [m.get("sender") for m in raw_msgs]
    names = await _load_names(db, list(participant_ids) + sender_ids)

    transcript: list[dict] = []
    for m in raw_msgs:
        text = _msg_text(m)
        if not text:
            continue
        transcript.append({
            "sender": names.get(str(m.get("sender")), "Unknown"),
            "text": text,
            "ts": _iso(m.get("createdAt")),
        })

    participants = [names.get(str(pid), "Unknown") for pid in participant_ids]
    latest_activity = (
        transcript[-1]["ts"] if transcript else _iso(conv.get("updatedAt"))
    )

    return ConversationSummaryData(
        conversation_id=str(conv_oid),
        kind=kind,
        participants=participants,
        message_count=len(transcript),
        latest_activity=latest_activity,
        topics=_extract_topics([t["text"] for t in transcript]),
        transcript=transcript[:_MAX_MESSAGES],
        title=str(conv.get("name", "")) if kind == "group" else "",
    )

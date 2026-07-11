"""Voice-session recall — MongoDB only, no Qdrant, no LLM for the lookup.

Finds the most recently *summarized* VoiceSession for a named channel and
returns its stored summary/keyPoints/actionItems verbatim. These were already
generated once by generate_voice_session_summary() when the call ended
(app.api.internal.summarize_voice_session) — this tool never re-calls the
LLM, it only retrieves what was already written to MongoDB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import to_oid

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.voice_session_recall")


@dataclass
class VoiceSessionRecallData:
    channel_id: str
    channel_name: str
    server_name: str
    session_id: str
    summary: str
    key_points: list[str]
    action_items: list[str]
    ended_at: str | None
    participant_count: int


# Mirrors community_summary.py's _can_view rule (public visible to all,
# private visible only to members) — duplicated rather than imported since
# each tool module here is deliberately self-contained.
def _can_view(server: dict, user_oid: Any) -> bool:
    if server.get("visibility") == "public":
        return True
    if user_oid is None:
        return False
    return any(m.get("user") == user_oid for m in (server.get("members") or []))


async def get_voice_session_recall(
    user_id: str, channel_name: str
) -> VoiceSessionRecallData | None:
    """Return the most recent summarized voice session for `channel_name`.

    Returns None when the channel doesn't exist, the caller lacks access, or
    no session has been summarized yet for that channel. Never raises.
    """
    try:
        return await _voice_session_recall_impl(user_id, channel_name)
    except Exception:
        logger.exception(
            "get_voice_session_recall failed user=%s channel=%r", user_id, channel_name
        )
        return None


async def _voice_session_recall_impl(
    user_id: str, channel_name: str
) -> VoiceSessionRecallData | None:
    user_oid = to_oid(user_id)
    db = get_db()

    channel = await db["channels"].find_one(
        {"name": {"$regex": f"^{re.escape(channel_name)}$", "$options": "i"}},
        {"name": 1, "server": 1},
    )
    if channel is None:
        return None

    channel_oid = channel["_id"]
    server_oid = channel.get("server")

    server_doc: dict | None = None
    if server_oid:
        server_doc = await db["discordservers"].find_one(
            {"_id": server_oid}, {"name": 1, "visibility": 1, "members": 1}
        )
    if server_doc and not _can_view(server_doc, user_oid):
        return None

    server_name = (server_doc or {}).get("name") or "Unknown Server"

    session = await db["voicesessions"].find_one(
        {"channel": channel_oid, "status": "summarized"},
        {"summary": 1, "keyPoints": 1, "actionItems": 1, "endedAt": 1, "participants": 1},
        sort=[("endedAt", -1)],
    )
    if session is None:
        return None

    ended_at = session.get("endedAt")
    return VoiceSessionRecallData(
        channel_id=str(channel_oid),
        channel_name=channel.get("name") or channel_name,
        server_name=server_name,
        session_id=str(session["_id"]),
        summary=session.get("summary") or "",
        key_points=list(session.get("keyPoints") or []),
        action_items=list(session.get("actionItems") or []),
        ended_at=ended_at.isoformat() if isinstance(ended_at, datetime) else None,
        participant_count=len(session.get("participants") or []),
    )

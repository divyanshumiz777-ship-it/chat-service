"""Voice-session recall — MongoDB only, no Qdrant, no LLM for the lookup.

Finds the most recently *summarized* VoiceSession for a named channel and
returns its stored summary/keyPoints/actionItems verbatim. These were already
generated once by generate_voice_session_summary() when the call ended
(app.api.internal.summarize_voice_session) — this tool never re-calls the
LLM, it only retrieves what was already written to MongoDB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._access import resolve_channel_by_name
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

    # Shared resolver (app/tools/_access.py): the caller's own servers first,
    # then public ones, voice/video channels preferred over a same-named text
    # channel; servers the caller can't see are never returned.
    resolved = await resolve_channel_by_name(
        db, user_oid, channel_name, prefer_types=("Voice", "Video")
    )
    if resolved is None:
        return None
    channel, server_doc, _is_member = resolved

    channel_oid = channel["_id"]
    server_name = server_doc.get("name") or "Unknown Server"

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

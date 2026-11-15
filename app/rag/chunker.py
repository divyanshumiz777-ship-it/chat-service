from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_CHARS = 512    
_OVERLAP_CHARS = 64  


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AclPayload:
    """Qdrant point payload encoding identity + access-control for one chunk."""

    source_type: str  
    source_id: str   
    visibility: str    
    server_id: str | None = None
    channel_id: str | None = None
    conversation_id: str | None = None
    group_id: str | None = None
    owner_id: str | None = None
    allowed_user_ids: list[str] = field(default_factory=list)
    updated_at: str = "" 
    title: str = ""  

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "visibility": self.visibility,
            "allowed_user_ids": self.allowed_user_ids,
        }
        for key in (
            "title", "server_id", "channel_id", "conversation_id",
            "group_id", "owner_id", "updated_at",
        ):
            val = getattr(self, key)
            if val:
                d[key] = val
        return d


@dataclass
class RagChunk:
    text: str
    payload: AclPayload
    content_hash: str
    chunk_index: int = 0


# ── Internal helpers ───────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _str_id(val: Any) -> str:
    return str(val) if val is not None else ""


def _str_ids(vals: list[Any]) -> list[str]:
    return [str(v) for v in vals if v is not None]


def _str_date(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _split_text(text: str) -> list[str]:
    """Split into sentence-boundary-aware chunks of ≤ _MAX_CHARS chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= _MAX_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _MAX_CHARS, len(text))
        if end < len(text):
            for sep in (".", "?", "!", "\n", " "):
                pos = text.rfind(sep, start + 1, end)
                if pos > start:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Guarantee forward progress. `end` is always > start, but
        # `end - _OVERLAP_CHARS` need not be: when the chosen sentence boundary
        # lands within _OVERLAP_CHARS of `start` (a long passage whose only "."
        # sits near the top of the window), the next iteration restarts at or
        # BEFORE the current position and the loop never terminates — it
        # appends the same chunk until the process runs out of memory. Give up
        # the overlap for that one step rather than stall.
        next_start = end - _OVERLAP_CHARS
        start = next_start if next_start > start else end
    return chunks


def _make_chunks(text: str, payload: AclPayload) -> list[RagChunk]:
    return [
        RagChunk(text=part, payload=payload, content_hash=_hash(part), chunk_index=i)
        for i, part in enumerate(_split_text(text))
    ]


# ── Source normalizers ─────────────────────────────────────────────────────────

def normalize_doc(slug: str, title: str, body: str) -> list[RagChunk]:
    """A product-help topic (content/help/*.md) — the only source that is NOT a
    Mongo document.

    These exist so the assistant can answer "how does X work on TheSocial?".
    Previously it could not: the chat prompt forbids explaining anything, so
    product questions got the same refusal as questions about the weather —
    while the UI advertised "Explain the platform" as a capability.

    Always public and owner-less: the corpus is hand-written, version
    controlled, and identical for every user, so there is nothing to
    access-control and nothing to attribute.
    """
    text = f"{title}\n{body}".strip()
    if not text:
        return []

    payload = AclPayload(
        source_type="docs",
        source_id=slug,
        visibility="public",
        allowed_user_ids=[],
        title=title,
    )
    return _make_chunks(text, payload)


def normalize_server(doc: dict) -> list[RagChunk]:
    name = (doc.get("name") or "").strip()
    desc = (doc.get("description") or "").strip()
    text = f"{name}\n{desc}" if desc else name
    if not text:
        return []

    is_public = doc.get("visibility") == "public"
    allowed: list[str] = []
    if not is_public:
        allowed = _str_ids([m.get("user") for m in doc.get("members", [])])

    payload = AclPayload(
        source_type="server",
        source_id=_str_id(doc.get("_id")),
        visibility="public" if is_public else "private",
        server_id=_str_id(doc.get("_id")),
        owner_id=_str_id(doc.get("owner")),
        allowed_user_ids=allowed,
        updated_at=_str_date(doc.get("updatedAt")),
        title=name,
    )
    return _make_chunks(text, payload)


def normalize_channel(
    doc: dict, server_visibility: str, server_member_ids: list[str]
) -> list[RagChunk]:
    """Channel — inherits visibility from its parent server."""
    text = doc.get("name", "").strip()
    if not text:
        return []

    is_public = server_visibility == "public"
    payload = AclPayload(
        source_type="channel",
        source_id=_str_id(doc.get("_id")),
        visibility="public" if is_public else "private",
        server_id=_str_id(doc.get("server")),
        channel_id=_str_id(doc.get("_id")),
        allowed_user_ids=[] if is_public else server_member_ids,
        updated_at=_str_date(doc.get("updatedAt")),
        title=text,
    )
    return _make_chunks(text, payload)


def normalize_thread(
    doc: dict, server_visibility: str, server_member_ids: list[str]
) -> list[RagChunk]:
    """Thread — inherits visibility from its parent server."""
    text = doc.get("title", "").strip()
    if not text:
        return []

    is_public = server_visibility == "public"
    payload = AclPayload(
        source_type="thread",
        source_id=_str_id(doc.get("_id")),
        visibility="public" if is_public else "private",
        server_id=_str_id(doc.get("server")),
        channel_id=_str_id(doc.get("channel")),
        allowed_user_ids=[] if is_public else server_member_ids,
        updated_at=_str_date(doc.get("updatedAt")),
        title=text,
    )
    return _make_chunks(text, payload)


def normalize_reel(doc: dict) -> list[RagChunk]:
    """Reel — only isPublic=True, skip isDeleted=True."""
    if doc.get("isDeleted") or not doc.get("isPublic", True):
        return []

    caption = doc.get("caption", "").strip()
    tags = " ".join(str(t) for t in doc.get("tags", []))
    text = f"{caption} {tags}".strip()
    if not text:
        return []

    payload = AclPayload(
        source_type="reel",
        source_id=_str_id(doc.get("_id")),
        visibility="public",
        owner_id=_str_id(doc.get("creator_id")),
        allowed_user_ids=[],
        updated_at=_str_date(doc.get("updated_at")),  # Reel uses snake_case timestamps
        title=(caption[:100] if caption else tags[:80]).strip(),
    )
    return _make_chunks(text, payload)


def normalize_user(doc: dict) -> list[RagChunk]:
    """User — only profileVisibility="public" on an active account."""
    privacy = doc.get("settings", {}).get("privacy", {})
    if privacy.get("profileVisibility") != "public":
        return []
    # Deactivated profiles are hidden everywhere else in the app, and a
    # deleted account's profile must not be indexed at all.
    if doc.get("accountStatus") in ("deactivated", "deleted"):
        return []

    name = doc.get("name", "").strip()
    about = doc.get("about", "").strip()
    status = (doc.get("customStatus") or {}).get("text", "").strip()
    text = "\n".join(p for p in [name, about, status] if p)
    if not text:
        return []

    payload = AclPayload(
        source_type="user",
        source_id=_str_id(doc.get("_id")),
        visibility="public",
        owner_id=_str_id(doc.get("_id")),
        allowed_user_ids=[],
        updated_at=_str_date(doc.get("updatedAt")),
        title=name,
    )
    return _make_chunks(text, payload)


def _msg_text(doc: dict) -> str:
    """Extract plain text from a Message document (content can be str or object)."""
    content = doc.get("content", "")
    if isinstance(content, dict):
        return (
            content.get("plainText", "") or content.get("formattedContent", "")
        ).strip()
    return str(content).strip()


def normalize_channel_message(
    doc: dict, server_visibility: str, server_member_ids: list[str]
) -> list[RagChunk]:
    """Channel message — skip deletedForEveryone; skip if deletedFor is non-empty."""
    if doc.get("deletedForEveryone"):
        return []
    if doc.get("deletedFor"):
        return []

    text = _msg_text(doc)
    if not text:
        return []

    is_public = server_visibility == "public"
    payload = AclPayload(
        source_type="message",
        source_id=_str_id(doc.get("_id")),
        visibility="public" if is_public else "private",
        server_id=_str_id(doc.get("server")),
        channel_id=_str_id(doc.get("channel")),
        owner_id=_str_id(doc.get("sender")),
        allowed_user_ids=[] if is_public else server_member_ids,
        updated_at=_str_date(doc.get("updatedAt")),
        title=text[:80],
    )
    return _make_chunks(text, payload)


def normalize_dm_message(doc: dict, participants: list[str]) -> list[RagChunk]:
    """DM message — allowed_user_ids = participants minus deletedFor."""
    if doc.get("deletedForEveryone"):
        return []

    text = _msg_text(doc)
    if not text:
        return []

    deleted_for = set(_str_ids(doc.get("deletedFor") or []))
    allowed = [uid for uid in participants if uid not in deleted_for]
    if not allowed:
        return []

    payload = AclPayload(
        source_type="message",
        source_id=_str_id(doc.get("_id")),
        visibility="private",
        conversation_id=_str_id(doc.get("conversationId")),
        owner_id=_str_id(doc.get("sender")),
        allowed_user_ids=allowed,
        updated_at=_str_date(doc.get("updatedAt")),
        title=text[:80],
    )
    return _make_chunks(text, payload)


def normalize_group_message(doc: dict, participants: list[str]) -> list[RagChunk]:
    """Group DM message — allowed_user_ids = participants minus deletedFor."""
    if doc.get("deletedForEveryone"):
        return []

    text = _msg_text(doc)
    if not text:
        return []

    deleted_for = set(_str_ids(doc.get("deletedFor") or []))
    allowed = [uid for uid in participants if uid not in deleted_for]
    if not allowed:
        return []

    payload = AclPayload(
        source_type="message",
        source_id=_str_id(doc.get("_id")),
        visibility="private",
        group_id=_str_id(doc.get("groupId")),
        owner_id=_str_id(doc.get("sender")),
        allowed_user_ids=allowed,
        updated_at=_str_date(doc.get("updatedAt")),
        title=text[:80],
    )
    return _make_chunks(text, payload)


def normalize_voice_session(
    doc: dict,
    server_visibility: str,
    server_member_ids: list[str],
    channel_name: str = "",
) -> list[RagChunk]:
    """Voice session summary — inherits visibility from its parent server,
    exactly like a channel message. `doc` is a VoiceSession Mongo document;
    summary/keyPoints/actionItems live directly on it (see
    backend/src/models/VoiceSession.ts), not a separate collection, since a
    session only ever has one summary. Sessions with no summary yet (still
    active/processing/failed) have nothing to embed and are skipped.
    `channel_name` is optional (callers may not always have it resolved) and
    only affects the display title, not the indexed text or ACL."""
    summary = (doc.get("summary") or "").strip()
    key_points = doc.get("keyPoints") or []
    action_items = doc.get("actionItems") or []
    parts = [summary] if summary else []
    if key_points:
        parts.append("Key points: " + "; ".join(key_points))
    if action_items:
        parts.append("Action items: " + "; ".join(action_items))
    text = "\n".join(parts).strip()
    if not text:
        return []

    is_public = server_visibility == "public"
    title = f"Voice call in #{channel_name}" if channel_name else "Voice session"
    payload = AclPayload(
        source_type="voice_session",
        source_id=_str_id(doc.get("_id")),
        visibility="public" if is_public else "private",
        server_id=_str_id(doc.get("server")),
        channel_id=_str_id(doc.get("channel")),
        allowed_user_ids=[] if is_public else server_member_ids,
        updated_at=_str_date(doc.get("endedAt")),
        title=title,
    )
    return _make_chunks(text, payload)


def normalize_notification(doc: dict) -> list[RagChunk]:
    """Notification — recipient-only (drives "summarize unread")."""
    title = doc.get("title", "").strip()
    message = doc.get("message", "").strip()
    text = f"{title}\n{message}".strip() if title and message else (title or message)
    if not text:
        return []

    recipient = _str_id(doc.get("recipient"))
    if not recipient:
        return []

    payload = AclPayload(
        source_type="notification",
        source_id=_str_id(doc.get("_id")),
        visibility="private",
        owner_id=recipient,
        allowed_user_ids=[recipient],
        updated_at=_str_date(doc.get("updatedAt")),
        title=title,
    )
    return _make_chunks(text, payload)

"""Look up a DM conversation with a *named* person — MongoDB only, no Qdrant.

Powers the CONVERSATION_WITH_USER intent ("my conversation with divyanshu").
Resolves the requested name against the people the caller actually has direct
messages with, then returns a deterministic insight card for that conversation.

No semantic search, no Qdrant, no search_all(), and no LLM in this module — the
result is built purely from MongoDB and rendered deterministically by the caller,
so it can never hallucinate a conversation that does not exist.

Access control: only conversations where the caller is a participant are ever
considered, and matching is done against those conversations' partners only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.core.mongo import get_db
from app.tools._common import oid_set, to_oid
from app.tools.conversation_insights import (
    _MAX_SCAN_MESSAGES,
    _format_activity,
    _msg_text,
    _summarize,
)

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.conversation_lookup")


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class ConversationLookupResult:
    # "found" | "ambiguous" | "no_conversation" | "no_person" | "invalid"
    status: str
    requested_name: str = ""
    match: dict | None = None             # insight card when status == "found"
    candidates: list[str] = field(default_factory=list)  # names for ambiguous/no_conversation


# ── Name matching ──────────────────────────────────────────────────────────────

def _match(requested: str, partners: list[tuple[Any, str]]) -> list[tuple[Any, str]]:
    """Match `requested` against (oid, name) partners by descending strictness.

    Tiers (first non-empty wins): exact full name → token/first-name match →
    substring contains. Returns all partners in the best matching tier.
    """
    req = requested.strip().lower()
    req_tokens = set(req.split())

    exact: list[tuple[Any, str]] = []
    token: list[tuple[Any, str]] = []
    substr: list[tuple[Any, str]] = []

    for oid, pname in partners:
        pl = (pname or "").strip().lower()
        if not pl:
            continue
        if pl == req:
            exact.append((oid, pname))
            continue
        p_tokens = set(pl.split())
        if req in p_tokens or (req_tokens and req_tokens <= p_tokens):
            token.append((oid, pname))
        elif req in pl or pl in req:
            substr.append((oid, pname))

    return exact or token or substr


async def _build_card(
    db: Any, user_oid: Any, conv: dict, partner_oid: Any, partner: dict
) -> dict:
    """Build the deterministic insight card for one resolved conversation."""
    conv_id = conv["_id"]
    msg_filter: dict[str, Any] = {
        "conversationId": conv_id,
        "deletedForEveryone": {"$ne": True},
        "deletedFor": {"$nin": [user_oid]},
    }

    visible_count = await db["messages"].count_documents(msg_filter)
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

    return {
        "conversation_id": str(conv_id),
        "friend_id": str(partner_oid),
        "friend_name": partner.get("name") or "Unknown",
        "friend_profile_pic": partner.get("profilePic") or "",
        "message_count": visible_count,
        "latest_message": latest_text,
        "latest_activity": _format_activity(conv.get("updatedAt")),
        "summary": _summarize(raw_msgs),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

async def lookup_conversation_with(
    user_id: str, name: str
) -> ConversationLookupResult:
    """Resolve the caller's DM conversation with the person called `name`.

    Never raises — any error degrades to status "invalid" (caller renders a
    friendly fallback). MongoDB only; no Qdrant, no semantic search, no LLM.
    """
    requested = (name or "").strip()
    if not requested:
        return ConversationLookupResult(status="invalid")

    me_oid = to_oid(user_id)
    if not me_oid:
        logger.warning("invalid user_id for conversation lookup: %r", user_id)
        return ConversationLookupResult(status="invalid", requested_name=requested)

    try:
        db = get_db()

        me = await db["users"].find_one(
            {"_id": me_oid}, {"friends": 1, "blockedUsers": 1}
        )
        my_blocked = oid_set(me.get("blockedUsers") if me else None)

        # All DM conversations the caller is in and has not deleted.
        convs = await db["conversations"].find(
            {"participants": me_oid, "deletedFor": {"$nin": [me_oid]}},
            {"participants": 1, "updatedAt": 1},
        ).sort("updatedAt", -1).to_list(length=500)

        # Map partner_oid → most-recent conversation (convs already sorted desc).
        partner_conv: dict[Any, dict] = {}
        for conv in convs:
            others = [p for p in conv.get("participants", []) if p != me_oid]
            if not others:
                continue
            partner = others[0]
            if partner in my_blocked:
                continue
            partner_conv.setdefault(partner, conv)  # first = most recent

        if partner_conv:
            partner_oids = list(partner_conv.keys())
            name_docs = await db["users"].find(
                {"_id": {"$in": partner_oids}},
                {"name": 1, "profilePic": 1, "blockedUsers": 1},
            ).to_list(length=len(partner_oids))
            profile_by_oid = {d["_id"]: d for d in name_docs}

            # Drop partners who blocked the caller.
            partners: list[tuple[Any, str]] = []
            for oid in partner_oids:
                prof = profile_by_oid.get(oid)
                if not prof:
                    continue
                if me_oid in oid_set(prof.get("blockedUsers")):
                    continue
                partners.append((oid, prof.get("name") or ""))

            matched = _match(requested, partners)
            distinct = {oid for oid, _ in matched}

            if len(distinct) == 1:
                oid = next(iter(distinct))
                conv = partner_conv[oid]
                card = await _build_card(
                    db, me_oid, conv, oid, profile_by_oid[oid]
                )
                return ConversationLookupResult(
                    status="found", requested_name=requested, match=card
                )
            if len(distinct) > 1:
                names = [nm for _, nm in matched if nm]
                return ConversationLookupResult(
                    status="ambiguous", requested_name=requested,
                    candidates=names[:5],
                )

        # No conversation matched. Does a friend by that name exist (no DM yet)?
        my_friends = oid_set(me.get("friends") if me else None)
        if my_friends:
            friend_docs = await db["users"].find(
                {"_id": {"$in": list(my_friends)}}, {"name": 1}
            ).to_list(length=len(my_friends))
            friend_pairs = [(d["_id"], d.get("name") or "") for d in friend_docs]
            friend_matches = _match(requested, friend_pairs)
            if friend_matches:
                return ConversationLookupResult(
                    status="no_conversation", requested_name=requested,
                    candidates=[nm for _, nm in friend_matches if nm][:5],
                )

        return ConversationLookupResult(status="no_person", requested_name=requested)

    except Exception:  # noqa: BLE001 — graceful degradation boundary
        logger.exception("conversation lookup failed for user %s", user_id)
        return ConversationLookupResult(status="invalid", requested_name=requested)

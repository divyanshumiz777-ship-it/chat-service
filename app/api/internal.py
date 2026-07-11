"""Internal (gateway-only) endpoints.

Reachable only by the Hono API gateway, which must present the shared
`X-Internal-Token`. The service publishes no host port (internal network only).

Identity contract: user identity is sourced exclusively from the `X-User-Id`
header forwarded by the gateway after JWT verification. Body-supplied user IDs
are never read or trusted.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from app.config import get_settings
from app.core.security import require_internal_token
from app.llm.intent_router import (
    Intent,
    extract_channel_name,
    extract_community_name,
    extract_conversation_target,
    extract_user_search_term,
    route_intent,
)
from app.llm.capabilities import (
    capabilities_payload,
    parse_command,
    render_help_menu,
    suggestions_for,
)
from app.llm.provider import (
    generate_answer,
    generate_channel_summary,
    generate_chat_response,
    generate_community_summary,
    generate_conversation_summary,
    generate_voice_session_summary,
)
from app.llm.prompts import build_citations
from app.rag.retriever import search_all
from app.tools.conversation_summary import build_conversation_summary
from app.tools import (
    get_channel_summary,
    get_community_summary,
    get_conversation_insights,
    get_followed_creators,
    get_my_communities,
    get_my_friends,
    get_social_overview,
    get_trending_creators,
    get_voice_session_recall,
    lookup_conversation_with,
    lookup_my_community,
    search_users,
    suggest_communities,
    suggest_creators,
    suggest_friends,
)
from app.rag.ingest_sync import ingest_all, ingest_voice_session
from app.rag.delete_sync import delete_by_channel, delete_by_server, delete_source

router = APIRouter(prefix="/internal/v1", tags=["internal"])
settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.api.internal")


# ── Request models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # reserved; persistence not yet implemented


class SearchRequest(BaseModel):
    query: str
    # Optional source-type filter. Accepts friendly names (e.g. "community",
    # "creator") and canonical types; resolved + validated in the retriever.
    # None/empty → search all sources.
    types: list[str] | None = None


class TranscriptTurn(BaseModel):
    sender: str
    text: str


class VoiceSessionSummarizeRequest(BaseModel):
    session_id: str
    channel_id: str
    server_id: str
    channel_name: str
    server_name: str
    # "public" | "private" — mirrors every other normalize_* function's ACL
    # inputs (chunker.py); the gateway already knows this from its own
    # DiscordServer document, so this service never needs to re-derive it.
    visibility: str
    allowed_user_ids: list[str] = []
    participants: list[str]
    segments: list[TranscriptTurn]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/ingest", dependencies=[Depends(require_internal_token)])
async def ingest() -> dict:
    """Trigger a full ingestion pass (Mongo → chunk → embed → Qdrant upsert).

    Gateway-only (internal token). Returns ingest_all()'s per-source summary
    verbatim — it already has the shape:
        {"status": "success", "results": [{"source", "processed", ...}, ...]}
    """
    logger.info("ingest triggered")
    try:
        return await ingest_all()
    except Exception as exc:
        logger.exception("ingestion failed")
        raise HTTPException(
            status_code=500,
            detail=f"ingestion failed: {exc}",
        )


@router.get("/ping", dependencies=[Depends(require_internal_token)])
async def ping() -> dict:
    """Authenticated reachability check for the gateway."""
    return {"status": "ok", "service": settings.service_name}


class DeleteContentRequest(BaseModel):
    scope: str  # "server" | "channel" | "source"
    id: str
    # Required only when scope == "source" — which content-type table the id
    # belongs to (e.g. "thread", "message"). Ignored for "server"/"channel"
    # scopes, which match by field rather than by a single source_id.
    source_type: str | None = None


@router.post("/delete", dependencies=[Depends(require_internal_token)])
async def delete_content(request: DeleteContentRequest) -> dict:
    """Remove indexed content for a deleted server/channel/single source
    document — the Qdrant-side half of a Mongo delete. Gateway-only, called
    right after the corresponding MongoDB delete succeeds (deleteServer /
    deleteChannel / etc. in the Node backend). Best-effort from the caller's
    perspective — the Mongo delete has already committed either way; this
    only controls whether the deleted content keeps surfacing through
    search/the assistant.
    """
    if request.scope == "server":
        await delete_by_server(request.id)
    elif request.scope == "channel":
        await delete_by_channel(request.id)
    elif request.scope == "source":
        if not request.source_type:
            raise HTTPException(
                status_code=400, detail="source_type is required when scope='source'"
            )
        await delete_source(request.source_type, request.id)
    else:
        raise HTTPException(status_code=400, detail=f"unknown scope: {request.scope!r}")
    return {"status": "ok"}


@router.get("/capabilities", dependencies=[Depends(require_internal_token)])
async def capabilities(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    """Return the command registry for the frontend command palette (gateway-only).

    The palette fetches this once and filters locally (Part 10 — no backend call
    per keystroke). It is the single source of truth shared by natural-language
    routing, slash commands, the help menu, and the palette UI.
    """
    payload = capabilities_payload()
    logger.info(
        "HOP-3 capabilities user=%s count=%d", x_user_id or "?", len(payload)
    )
    return {"capabilities": payload}


# Recommender intent → (tool, kind, header, empty-result message).
_RECOMMENDERS = {
    Intent.SUGGEST_FRIENDS: (
        suggest_friends, "friends", "👥 Suggested friends for you:",
        "I couldn't find any friend recommendations yet. As you join communities "
        "and connect with people, I'll be able to suggest relevant friends.",
    ),
    Intent.SUGGEST_COMMUNITIES: (
        suggest_communities, "communities", "🏘️ Suggested communities for you:",
        "I couldn't find any community recommendations yet. As more communities "
        "are created, I'll be able to suggest relevant ones.",
    ),
    Intent.SUGGEST_CREATORS: (
        suggest_creators, "creators", "🎬 Suggested creators for you:",
        "I couldn't find any creator recommendations yet. As more users and reels "
        "are added, I'll be able to suggest relevant creators.",
    ),
    Intent.TRENDING_CREATORS: (
        get_trending_creators, "creators", "🔥 Trending creators right now:",
        "I couldn't find any trending creators yet. Once people start following "
        "creators, I'll be able to show who's growing fastest.",
    ),
}


def _format_recommendations(header: str, items: list[dict]) -> str:
    """Render recommender output deterministically (no LLM).

    Because there is no generative step, the reply can NEVER contain chain-of-
    thought, system-prompt text, or invented creators/reasons — only the names
    and reasons the MongoDB tool produced (which already excludes the current
    user, blocked/followed creators, and deleted entities).
    """
    lines: list[str] = [header, ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.get('name', 'Unknown')}")
        lines.append("")
        lines.append("Reasons:")
        reasons = [str(r).strip() for r in (item.get("reasons") or []) if str(r).strip()]
        if reasons:
            for r in reasons:
                lines.append(f"• {r[0].upper()}{r[1:]}")
        else:
            lines.append("• Recommended for you")
        lines.append("")
    return "\n".join(lines).strip()


async def _recommend(user_id: str, intent: Intent) -> dict:
    """Run a MongoDB recommender tool and format the result deterministically.

    The reply is templated from the tool's structured output — the LLM is NOT
    used here, so chain-of-thought and invented recommendations cannot leak.
    No Qdrant, no search_all().
    """
    tool, kind, header, empty_msg = _RECOMMENDERS[intent]
    items = await tool(user_id)

    if not items:
        return {"answer": empty_msg, "citations": [], "intent": intent, "recommendations": []}

    return {
        "answer": _format_recommendations(header, items),
        "citations": [],
        "intent": intent,
        "recommendations": items,
    }


async def _summarize_conversation(
    user_id: str, conversation_id: str | None, intent: Intent
) -> dict:
    """Summarize the user's current/open conversation from MongoDB (no RAG)."""
    data = await build_conversation_summary(user_id, conversation_id)

    if data is None:
        return {
            "answer": (
                "I couldn't find a conversation to summarize. Open a direct "
                "message or group chat and try again."
            ),
            "citations": [],
            "intent": intent,
        }

    if data.message_count == 0:
        answer = "This conversation doesn't have any messages yet."
    else:
        answer = await generate_conversation_summary(user_id=user_id, data=data)

    return {
        "answer": answer,
        "citations": [],
        "intent": intent,
        "conversation": {
            "conversation_id": data.conversation_id,
            "kind": data.kind,
            "participants": data.participants,
            "message_count": data.message_count,
            "latest_activity": data.latest_activity,
            "topics": data.topics,
        },
    }


def _format_conversation_insights(items: list[dict]) -> str:
    """Render conversation insights deterministically (no LLM).

    The LLM is never involved so chain-of-thought and invented conversations
    cannot appear. Only data returned by get_conversation_insights() is used.
    """
    lines: list[str] = ["💬 Your recent conversations:", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.get('friend_name', 'Unknown')}")
        mc = item.get("message_count", 0)
        lines.append(f"Messages: {mc}")
        activity = item.get("latest_activity", "")
        if activity:
            lines.append(f"Latest activity: {activity}")
        latest = item.get("latest_message", "")
        if latest:
            lines.append(f"Last message: {latest}")
        bullets = item.get("summary") or []
        if bullets:
            lines.append("")
            lines.append("Summary:")
            for b in bullets:
                lines.append(f"• {b}")
        lines.append("")
    return "\n".join(lines).strip()


async def _conversation_insights(user_id: str, intent: Intent) -> dict:
    """Load DM conversation insights from MongoDB (no RAG, no LLM for data)."""
    items = await get_conversation_insights(user_id)
    if not items:
        return {
            "answer": (
                "I couldn't find any conversations. Start a direct message "
                "with someone and come back to see your conversation insights."
            ),
            "citations": [],
            "intent": intent,
            "conversations": [],
        }
    return {
        "answer": _format_conversation_insights(items),
        "citations": [],
        "intent": intent,
        "conversations": items,
    }


def _format_single_conversation(card: dict) -> str:
    """Render one resolved conversation deterministically (no LLM)."""
    lines: list[str] = [f"💬 Your conversation with {card.get('friend_name', 'Unknown')}:", ""]
    lines.append(f"Messages: {card.get('message_count', 0)}")
    activity = card.get("latest_activity", "")
    if activity:
        lines.append(f"Latest activity: {activity}")
    latest = card.get("latest_message", "")
    if latest:
        lines.append(f"Last message: {latest}")
    bullets = card.get("summary") or []
    if bullets:
        lines.append("")
        lines.append("Summary:")
        for b in bullets:
            lines.append(f"• {b}")
    return "\n".join(lines).strip()


async def _conversation_with_user(user_id: str, message: str, intent: Intent) -> dict:
    """Look up the user's DM with a named person from MongoDB (no RAG, no LLM).

    The person name is extracted from the message; the conversation_lookup tool
    resolves it against the conversations the caller actually participates in.
    The reply is rendered deterministically, so no conversation can be invented.
    """
    name = extract_conversation_target(message)
    if not name:
        return {
            "answer": (
                "Tell me who you'd like to see the conversation with — for "
                "example, \"my conversation with Alex\"."
            ),
            "citations": [],
            "intent": intent,
            "conversations": [],
        }

    result = await lookup_conversation_with(user_id, name)

    if result.status == "found" and result.match:
        return {
            "answer": _format_single_conversation(result.match),
            "citations": [],
            "intent": intent,
            "conversations": [result.match],
        }

    if result.status == "ambiguous":
        names = ", ".join(result.candidates)
        answer = (
            f"You have conversations with more than one person matching "
            f"“{name}”: {names}. Who did you mean?"
        )
    elif result.status == "no_conversation":
        who = result.candidates[0] if result.candidates else name
        answer = (
            f"You're friends with {who}, but you don't have any direct messages "
            f"yet. Say hi to start the conversation!"
        )
    else:  # no_person / invalid
        answer = (
            f"I couldn't find a conversation with “{name}”. Check the "
            f"spelling, or start a direct message with them first."
        )

    return {
        "answer": answer,
        "citations": [],
        "intent": intent,
        "conversations": [],
    }


# ── Social graph listing handlers (MongoDB only, no Qdrant, no LLM) ───────────

def _format_friends(items: list[dict]) -> str:
    lines: list[str] = ["👥 Your friends:", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.get('name', 'Unknown')}")
        mutual = item.get("mutual_servers", 0)
        if mutual:
            lines.append(f"   Mutual servers: {mutual}")
        last = item.get("last_activity")
        if last:
            lines.append(f"   Last chat: {last}")
        lines.append("")
    return "\n".join(lines).strip()


async def _list_friends(user_id: str, intent: Intent) -> dict:
    items = await get_my_friends(user_id)
    if not items:
        return {
            "answer": (
                "You haven't added any friends yet. Send a friend request to "
                "someone and they'll appear here once they accept!"
            ),
            "citations": [],
            "intent": intent,
            "friends": [],
        }
    return {
        "answer": _format_friends(items),
        "citations": [],
        "intent": intent,
        "friends": items,
    }


def _format_communities(items: list[dict], header: str = "🏘️ Communities you've joined:") -> str:
    lines: list[str] = [header, ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.get('name', 'Unknown')}")
        lines.append(f"   Role: {item.get('role', 'member').title()}")
        lines.append(f"   Members: {item.get('member_count', 0)}")
        vis = item.get("visibility", "public").replace("-", " ").title()
        lines.append(f"   Visibility: {vis}")
        if item.get("created_at"):
            lines.append(f"   Created: {item['created_at']}")
        lines.append("")
    return "\n".join(lines).strip()


async def _list_communities(user_id: str, intent: Intent) -> dict:
    items = await get_my_communities(user_id)
    if not items:
        return {
            "answer": (
                "You haven't joined any communities yet. Explore public communities "
                "and join one to get started!"
            ),
            "citations": [],
            "intent": intent,
            "communities": [],
        }
    return {
        "answer": _format_communities(items),
        "citations": [],
        "intent": intent,
        "communities": items,
    }


async def _community_lookup(user_id: str, intent: Intent) -> dict:
    items = await lookup_my_community(user_id)
    if not items:
        return {
            "answer": (
                "You're not a member of any community yet. Browse public communities "
                "and join one to get started!"
            ),
            "citations": [],
            "intent": intent,
            "communities": [],
        }
    return {
        "answer": _format_communities(items, header="🏘️ Communities you belong to:"),
        "citations": [],
        "intent": intent,
        "communities": items,
    }


def _format_creators(items: list[dict]) -> str:
    lines: list[str] = ["🎬 Creators you follow:", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item.get('name', 'Unknown')}")
        rc = item.get("reel_count", 0)
        if rc:
            lines.append(f"   Reels: {rc}")
        about = (item.get("about") or "").strip()
        if about:
            # Truncate long bios.
            lines.append(f"   {about[:80]}{'…' if len(about) > 80 else ''}")
        lines.append("")
    return "\n".join(lines).strip()


async def _list_creators(user_id: str, intent: Intent) -> dict:
    items = await get_followed_creators(user_id)
    if not items:
        return {
            "answer": (
                "You aren't following any creators yet. Explore the reels feed "
                "and follow creators whose content you enjoy!"
            ),
            "citations": [],
            "intent": intent,
            "creators": [],
        }
    return {
        "answer": _format_creators(items),
        "citations": [],
        "intent": intent,
        "creators": items,
    }


def _suggested_actions(
    data: dict, friend_recs: list[dict], community_recs: list[dict]
) -> list[str]:
    """Deterministic next-step prompts derived from gaps in the user's activity."""
    actions: list[str] = []
    if not data.get("friends_count"):
        actions.append('👥 Add your first friend — try "suggest friends"')
    elif friend_recs:
        name = friend_recs[0].get("name", "someone new")
        actions.append(f'👥 Connect with {name} — try "suggest friends"')
    if not data.get("communities_count"):
        actions.append('🏘️ Join a community — try "suggest communities"')
    elif community_recs:
        name = community_recs[0].get("name", "a new community")
        actions.append(f'🏘️ Explore {name} — try "suggest communities"')
    if not data.get("creators_count"):
        actions.append('🎬 Follow a creator — try "suggest creators"')
    if not data.get("messages_count"):
        actions.append('💬 Start a conversation — try "my friends"')
    return actions[:4]


def _format_dashboard(
    data: dict, friend_recs: list[dict], community_recs: list[dict]
) -> str:
    """Render the unified dashboard deterministically (no LLM). Cards + icons."""
    lines: list[str] = ["📊 Your TheSocial dashboard", ""]

    # ── Stats ───────────────────────────────────────────────────────────────────
    lines.append(f"👥 Friends: {data.get('friends_count', 0)}")
    lines.append(f"🏘️ Communities: {data.get('communities_count', 0)}")
    lines.append(f"🎬 Creators followed: {data.get('creators_count', 0)}")
    lines.append(f"💬 Messages (last 30 days): {data.get('messages_count', 0)}")
    lines.append("")

    # ── Most active ───────────────────────────────────────────────────────────────
    maf = data.get("most_active_friend")
    mac = data.get("most_active_community")
    mar = data.get("most_active_creator")
    if maf or mac or mar:
        lines.append("Most active:")
        if maf:
            lines.append(f"• 👤 Friend: {maf}")
        if mac:
            lines.append(f"• 🏘️ Community: {mac}")
        if mar:
            lines.append(f"• 🎬 Creator: {mar}")
        lines.append("")

    # ── Recent activity ───────────────────────────────────────────────────────────
    bullets = data.get("recent_bullets") or []
    if bullets:
        lines.append("Recent activity:")
        for b in bullets:
            lines.append(f"• {b}")
        lines.append("")

    # ── Recommendations (previews from the recommender tools) ──────────────────────
    rec_lines: list[str] = []
    for f in friend_recs[:3]:
        rec_lines.append(f"• 👤 {f.get('name', 'Unknown')} (friend)")
    for c in community_recs[:3]:
        rec_lines.append(f"• 🏘️ {c.get('name', 'Unknown')} (community)")
    if rec_lines:
        lines.append("Recommended for you:")
        lines.extend(rec_lines)
        lines.append("")

    # ── Suggested actions ─────────────────────────────────────────────────────────
    actions = _suggested_actions(data, friend_recs, community_recs)
    if actions:
        lines.append("Suggested actions:")
        for a in actions:
            lines.append(f"• {a}")

    return "\n".join(lines).strip()


async def _social_overview(user_id: str, intent: Intent) -> dict:
    """Unified dashboard: stats + most-active + recent activity + recommendations.

    Runs one MongoDB overview plus two recommender previews concurrently. All
    three degrade gracefully (overview→None, recs→[]), so a partial failure still
    renders a useful dashboard. No Qdrant, no LLM — fully deterministic.
    """
    data, friend_recs, community_recs = await asyncio.gather(
        get_social_overview(user_id),
        suggest_friends(user_id, limit=3),
        suggest_communities(user_id, limit=3),
    )
    if data is None:
        return {
            "answer": (
                "I couldn't load your dashboard right now. Please try again "
                "in a moment."
            ),
            "citations": [],
            "intent": intent,
        }
    return {
        "answer": _format_dashboard(data, friend_recs or [], community_recs or []),
        "citations": [],
        "intent": intent,
        "overview": data,
        "recommendations": {
            "friends": friend_recs or [],
            "communities": community_recs or [],
        },
    }


# ── Global people search (MongoDB only, no Qdrant, no LLM) ────────────────────

_FRIEND_STATUS_LABEL = {
    "friend": "✅ Friend",
    "pending": "⏳ Request pending",
    "none": "",
}


def _format_user_search(items: list[dict], term: str | None) -> str:
    """Render user-search results deterministically (no LLM). Cards + icons."""
    header = f'👤 People matching "{term}":' if term else "👤 People on TheSocial:"
    lines: list[str] = [header, ""]
    for i, item in enumerate(items, 1):
        name = item.get("name", "Unknown")
        badge = " ⭐ Creator" if item.get("is_creator") else ""
        lines.append(f"{i}. {name}{badge}")
        status_label = _FRIEND_STATUS_LABEL.get(item.get("friend_status", "none"), "")
        if status_label:
            lines.append(f"   {status_label}")
        mutual = item.get("mutual_communities", 0)
        if mutual:
            lines.append(f"   🏘️ {mutual} mutual communit{'ies' if mutual != 1 else 'y'}")
        cs = (item.get("custom_status") or "").strip()
        if cs:
            lines.append(f"   💬 {cs[:60]}")
        about = (item.get("about") or "").strip()
        if about:
            lines.append(f"   {about[:80]}{'…' if len(about) > 80 else ''}")
        url = item.get("profile_url")
        if url:
            lines.append(f"   🔗 {url}")
        lines.append("")
    return "\n".join(lines).strip()


async def _user_search(user_id: str, term: str | None, intent: Intent) -> dict:
    """Global ACL-safe people search. MongoDB only — no Qdrant, no LLM.

    `term` is the person name for "find <name>" / "search users <name>", or None
    for "users"/"list users" (recent users). Results are ACL-filtered by the tool.
    """
    items = await search_users(user_id, term=term)
    if not items:
        title = (
            f'No users matching "{term}" were found.' if term
            else "No matching users were found."
        )
        return {
            "answer": _empty_state(title, ["users", "search users", "my friends"]),
            "citations": [],
            "intent": intent,
            "users": [],
        }
    return {
        "answer": _format_user_search(items, term),
        "citations": [],
        "intent": intent,
        "users": items,
    }


# ── Community / channel summary handlers (MongoDB + LLM, no Qdrant) ───────────

async def _community_summary(user_id: str, message: str, intent: Intent) -> dict:
    """Summarise a community from MongoDB metadata and recent messages.

    Extracts the community name from the natural-language message (may be None
    for generic queries like "what is this community about"), queries MongoDB,
    then asks the LLM to write a structured overview.  Never returns
    "AI assistant is temporarily unavailable" for a query about a valid entity.
    """
    name = extract_community_name(message)
    data = await get_community_summary(user_id, name)

    if data is None:
        if name:
            return {
                "answer": (
                    f"I couldn't find a community called \"{name}\". "
                    "Check the spelling, or try \"my communities\" to see the ones "
                    "you've joined."
                ),
                "citations": [],
                "intent": intent,
            }
        return {
            "answer": (
                "You haven't joined any communities yet. "
                "Explore public communities and join one to get started! "
                "Try \"suggest communities\" to see what's available."
            ),
            "citations": [],
            "intent": intent,
        }

    # Use LLM when there's enough data; fall back to a structured card otherwise.
    if data.recent_messages or data.description or data.channel_names:
        answer = await generate_community_summary(user_id=user_id, data=data)
    else:
        ch = ", ".join(data.channel_names[:5]) or "no channels yet"
        vis = data.visibility.replace("-", " ").title()
        mc = data.member_count
        answer = (
            f"**{data.name}** is a {vis} community with "
            f"{mc} member{'s' if mc != 1 else ''}. "
            f"Channels: {ch}."
        )
        if data.description:
            answer += f"\n\n{data.description}"

    return {
        "answer": answer,
        "citations": [],
        "intent": intent,
        "community": {
            "server_id": data.server_id,
            "name": data.name,
            "member_count": data.member_count,
            "channel_count": len(data.channel_names),
            "visibility": data.visibility,
            "is_member": data.is_member,
        },
    }


async def _channel_summary(user_id: str, message: str, intent: Intent) -> dict:
    """Summarise a channel from its recent messages and pinned content.

    Extracts the channel name from the natural-language message, queries
    MongoDB, then asks the LLM to write a brief overview.  Quiet channels
    (zero messages) receive a graceful explanation rather than an error.
    """
    channel_name = extract_channel_name(message)

    if not channel_name:
        return {
            "answer": (
                "Which channel would you like to know about? "
                "For example: \"what happens in #general\" or \"summarize #notes\"."
            ),
            "citations": [],
            "intent": intent,
        }

    data = await get_channel_summary(user_id, channel_name)

    if data is None:
        return {
            "answer": (
                f"I couldn't find a channel called \"#{channel_name}\". "
                "Check the spelling, or try \"my communities\" to browse your "
                "servers and their channels."
            ),
            "citations": [],
            "intent": intent,
        }

    if data.is_quiet:
        answer = (
            f"The **#{data.name}** channel in {data.server_name} is currently "
            f"quiet — there are no messages yet. "
            f"It appears to be a space for "
            f"{'voice conversations' if 'voice' in data.name.lower() else 'text discussion'}."
        )
    else:
        answer = await generate_channel_summary(user_id=user_id, data=data)

    return {
        "answer": answer,
        "citations": [],
        "intent": intent,
        "channel": {
            "channel_id": data.channel_id,
            "name": data.name,
            "server_name": data.server_name,
            "message_count": data.message_count,
            "thread_count": data.thread_count,
        },
    }


# ── Voice session recall (MongoDB only, no Qdrant, no LLM) ────────────────────

async def _voice_session_recall(user_id: str, message: str, intent: Intent) -> dict:
    """Recall the most recent summarized voice session in a named channel.

    Extracts the channel name from the natural-language message, then renders
    the STORED summary/key points/action items deterministically (no LLM
    call) — that text was already generated once by
    generate_voice_session_summary() when the call ended, so re-running it
    through the LLM here would be redundant and risk paraphrasing an
    already-final recap.
    """
    channel_name = extract_channel_name(message)

    if not channel_name:
        return {
            "answer": (
                "Which channel's call would you like to recall? "
                "For example: \"what happened in the design call\"."
            ),
            "citations": [],
            "intent": intent,
        }

    data = await get_voice_session_recall(user_id, channel_name)

    if data is None:
        return {
            "answer": (
                f"I couldn't find a recent voice call in \"#{channel_name}\". "
                "Check the spelling, or that channel may not have had a "
                "summarized call yet."
            ),
            "citations": [],
            "intent": intent,
        }

    lines = [
        f"🎙️ Recap of the last call in **#{data.channel_name}** ({data.server_name}):",
        "",
        data.summary,
    ]
    if data.key_points:
        lines += ["", "Key points:"] + [f"• {p}" for p in data.key_points]
    if data.action_items:
        lines += ["", "Action items:"] + [f"• {a}" for a in data.action_items]

    return {
        "answer": "\n".join(lines).strip(),
        "citations": [],
        "intent": intent,
        "voice_session": {
            "session_id": data.session_id,
            "channel_id": data.channel_id,
            "channel_name": data.channel_name,
            "server_name": data.server_name,
            "ended_at": data.ended_at,
            "participant_count": data.participant_count,
        },
    }


# ── Help + cross-cutting response helpers ─────────────────────────────────────

async def _help(user_id: str, intent: Intent) -> dict:
    """Render the full command menu from the Capability Registry (no LLM).

    `capabilities` is included so a client can render a rich menu; the text
    `answer` is the same menu for plain chat surfaces. user_id is unused (the
    menu is identical for everyone) but kept for a uniform handler signature.
    """
    return {
        "answer": render_help_menu(),
        "citations": [],
        "intent": intent,
        "capabilities": capabilities_payload(),
    }


def _empty_state(title: str, suggestions: list[str]) -> str:
    """Render a helpful empty state that never dead-ends the conversation."""
    lines = [title, "", "Try"]
    lines += [f"• {s}" for s in suggestions]
    return "\n".join(lines)


def _attach_suggestions(response: dict, intent: Intent) -> dict:
    """Append related-command hints to a tool response (Part 5 smart suggestions).

    No-op for intents without registered suggestions (e.g. CHAT/QA). Adds a
    `suggestions` field plus a "You can also ask" footer to the answer text, so
    even empty states surface discoverable next steps.
    """
    sugg = suggestions_for(intent)
    if not sugg:
        return response
    response = dict(response)
    response["suggestions"] = sugg
    answer = response.get("answer")
    if answer:
        footer = "\n".join(["You can also ask:"] + [f"• {s}" for s in sugg])
        response["answer"] = f"{answer}\n\n{footer}"
    return response


# Intent → no-arg tool handler. All take (user_id, intent) and return a response
# dict. Adding a tool here + a Capability is the entire wiring for a new command.
_SIMPLE_TOOL_HANDLERS = {
    Intent.LIST_FRIENDS: _list_friends,
    Intent.LIST_COMMUNITIES: _list_communities,
    Intent.COMMUNITY_LOOKUP: _community_lookup,
    Intent.LIST_CREATORS: _list_creators,
    Intent.SOCIAL_OVERVIEW: _social_overview,
    Intent.CONVERSATION_INSIGHTS: _conversation_insights,
}

# Message-aware handlers: receive the raw user message for name extraction.
# Keyed separately so the dispatch code can forward the message parameter.
_MESSAGE_TOOL_HANDLERS = {
    Intent.COMMUNITY_SUMMARY: _community_summary,
    Intent.CHANNEL_SUMMARY: _channel_summary,
    Intent.VOICE_SESSION_RECALL: _voice_session_recall,
}


@router.post("/chat", dependencies=[Depends(require_internal_token)])
async def chat(
    request: ChatRequest,
    x_user_id: str = Header(alias="X-User-Id"),
) -> dict:
    """Dual-mode chat: general conversation or RAG, selected by intent routing.

    Flow:
      chat intent    → LLM only (Qdrant bypassed entirely)
      mixed intent   → RAG; falls back to LLM-only when Qdrant returns nothing
      all others     → RAG (search_all → generate_answer)

    ACL enforcement is mandatory inside search_all() and is never bypassed.
    User identity comes exclusively from X-User-Id (gateway-forwarded, JWT-verified).
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty",
        )

    # Slash command? Resolve it (and any args) from the Capability Registry first;
    # otherwise classify natural language via the router (registry alias → regex).
    parsed = parse_command(message)
    intent = parsed.intent if parsed is not None else route_intent(message)
    logger.info(
        "chat user=%s intent=%s slash=%s msg_len=%d",
        x_user_id, intent, parsed is not None, len(message),
    )

    # ── Tool intents — deterministic MongoDB responses (+ smart suggestions) ──────
    # The tool produces the data; replies are formatted deterministically (no LLM,
    # no Qdrant). _attach_suggestions adds related-command hints (Part 5).
    if intent in _RECOMMENDERS:
        return _attach_suggestions(await _recommend(x_user_id, intent), intent)

    handler = _SIMPLE_TOOL_HANDLERS.get(intent)
    if handler is not None:
        return _attach_suggestions(await handler(x_user_id, intent), intent)

    # Community / channel summary — require the raw message for name extraction.
    msg_handler = _MESSAGE_TOOL_HANDLERS.get(intent)
    if msg_handler is not None:
        return _attach_suggestions(
            await msg_handler(x_user_id, message, intent), intent
        )

    # Global people search — term comes from the slash command or the NL extractor.
    if intent == Intent.USER_SEARCH:
        term = parsed.term if parsed is not None else extract_user_search_term(message)
        return _attach_suggestions(await _user_search(x_user_id, term, intent), intent)

    # Conversation with a named person ("my conversation with divyanshu").
    if intent == Intent.CONVERSATION_WITH_USER:
        return _attach_suggestions(
            await _conversation_with_user(x_user_id, message, intent), intent
        )

    # Current-conversation summary: read MongoDB directly (no Qdrant/RAG), then
    # let the LLM format the prose. Uses the gateway-forwarded conversation_id
    # when present, else the user's most-recently-active conversation.
    if intent == Intent.SUMMARIZE_CONVERSATION:
        return _attach_suggestions(
            await _summarize_conversation(x_user_id, request.conversation_id, intent),
            intent,
        )

    # Help / command menu — rendered from the registry (already a full menu, so no
    # suggestion footer is appended).
    if intent == Intent.HELP:
        return await _help(x_user_id, intent)

    # General conversation: skip Qdrant, call the domain-locked LLM directly.
    if intent == Intent.CHAT:
        answer = await generate_chat_response(user_id=x_user_id, query=message)
        return {"answer": answer, "citations": [], "intent": intent}

    # ── Platform / RAG path ───────────────────────────────────────────────────────
    # A slash search may override the query and scope it to specific source types.
    query = parsed.query if (parsed is not None and parsed.query) else message
    types = parsed.types if parsed is not None else None
    results = await search_all(user_id=x_user_id, query=query, types=types)

    # Mixed intent with no RAG results: fall back to conversational LLM
    if intent == Intent.MIXED and not results:
        answer = await generate_chat_response(user_id=x_user_id, query=message)
        return {"answer": answer, "citations": [], "intent": intent}

    answer = await generate_answer(user_id=x_user_id, query=query, results=results)
    return {
        "answer": answer,
        "citations": build_citations(results),
        "intent": intent,
    }


@router.post("/search", dependencies=[Depends(require_internal_token)])
async def search(
    request: SearchRequest,
    x_user_id: str = Header(alias="X-User-Id"),
) -> dict:
    """Semantic search: retrieve → return raw results.

    ACL enforcement is mandatory inside search_all() and is never bypassed.
    User identity comes exclusively from X-User-Id (gateway-forwarded, JWT-verified).
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query must not be empty",
        )

    logger.info(
        "search user=%s query_len=%d types=%s",
        x_user_id, len(query), request.types,
    )

    results = await search_all(
        user_id=x_user_id,
        query=query,
        types=request.types,
    )
    return {"results": build_citations(results)}


# ── Voice session summarization + immediate RAG ingestion ────────────────────

def _parse_voice_summary(raw: str) -> tuple[str, list[str], list[str]]:
    """Parse the SUMMARY/KEY POINTS/ACTION ITEMS sections the system prompt
    requires. Falls back to treating the whole response as the summary (empty
    key points/action items) if the model didn't follow the format exactly —
    keeps a malformed-but-non-empty response usable rather than discarding it.
    """
    summary = ""
    key_points: list[str] = []
    action_items: list[str] = []

    section: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("SUMMARY:"):
            section = "summary"
            rest = stripped[len("SUMMARY:"):].strip()
            if rest:
                summary = rest
            continue
        if upper.startswith("KEY POINTS:"):
            section = "key_points"
            continue
        if upper.startswith("ACTION ITEMS:"):
            section = "action_items"
            continue
        if not stripped:
            continue

        bullet = stripped.lstrip("-•").strip()
        if section == "summary":
            summary = f"{summary} {stripped}".strip() if summary else stripped
        elif section == "key_points" and bullet and bullet.lower() != "none":
            key_points.append(bullet)
        elif section == "action_items" and bullet and bullet.lower() != "none":
            action_items.append(bullet)

    if not summary and not key_points and not action_items:
        # Model didn't follow the format at all — use the raw text as-is
        # rather than silently discarding a real (if unstructured) summary.
        summary = raw.strip()

    return summary, key_points, action_items


@router.post("/voice/summarize", dependencies=[Depends(require_internal_token)])
async def summarize_voice_session(request: VoiceSessionSummarizeRequest) -> dict:
    """Summarize a voice-session transcript and immediately index it into the
    same RAG pipeline as every other content type (chunker.normalize_voice_session
    → embedder → Qdrant, via ingest_voice_session) — searchable/recallable the
    moment the summary exists, not on the next batch/admin-triggered ingest.

    No X-User-Id here (unlike /chat and /search): this call summarizes a
    session on behalf of all its participants, not a single querying user —
    ACL is enforced later, at retrieval time, via `allowed_user_ids`.
    """
    raw = await generate_voice_session_summary(
        user_id=request.session_id,  # audit/log field only, no per-user ACL here
        channel_name=request.channel_name,
        server_name=request.server_name,
        participants=request.participants,
        transcript=[t.model_dump() for t in request.segments],
    )
    if raw is None:
        logger.error("voice session summarization failed for session %s", request.session_id)
        raise HTTPException(
            status_code=503,
            detail="AI summary generation is temporarily unavailable",
        )

    summary, key_points, action_items = _parse_voice_summary(raw)

    try:
        doc = {
            "_id": request.session_id,
            "server": request.server_id,
            "channel": request.channel_id,
            "summary": summary,
            "keyPoints": key_points,
            "actionItems": action_items,
            "endedAt": None,
        }
        ingest_result = await ingest_voice_session(
            doc, request.visibility, request.allowed_user_ids, request.channel_name
        )
        ingest_result.log_summary()
    except Exception:
        logger.exception(
            "voice session RAG ingestion failed for session %s", request.session_id
        )
        # Summary generation already succeeded — still return it below. A
        # failed ingest only means this session won't be semantically
        # searchable yet; it doesn't invalidate the summary the gateway (and
        # the user waiting on it) needs.

    return {"summary": summary, "keyPoints": key_points, "actionItems": action_items}

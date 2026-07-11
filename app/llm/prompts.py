"""System prompts and context assembly for the LLM generation layer.

Pure module — no I/O. Takes retrieved SearchResult objects and produces:
  - build_context()             → formatted numbered context block for the prompt
  - build_citations()           → citation dicts safe to return in the API response
  - build_prompt()              → RAG user-turn prompt (context + question)
  - build_conversation_prompt() → pass-through for general chat (no context)
  - build_summary_prompt()      → conversation-summary prompt (transcript + meta)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.retriever import SearchResult
    from app.tools.community_summary import ChannelSummaryData, CommunitySummaryData
    from app.tools.conversation_summary import ConversationSummaryData

# ── Domain restriction constants ────────────────────────────────────────────────

# TheSocial is the ONLY domain this assistant serves. These two strings are the
# canonical user-facing replies for (a) anything outside TheSocial and (b) a query
# that returned no platform data. They share the same suggestion footer so the UX
# is consistent, and they are imported by provider.py / internal.py so every exit
# path returns identical wording.

_SUGGESTION_FOOTER = (
    "Try asking:\n"
    "• my dashboard\n"
    "• my friends\n"
    "• users\n"
    "• my communities\n"
    "• search reels\n"
    "• suggest creators"
)

# Returned when the user asks about the outside world / other products / general
# knowledge — the assistant is not a general-purpose chatbot.
DOMAIN_REFUSAL = (
    "I can only help with your TheSocial workspace and activity.\n\n"
    f"{_SUGGESTION_FOOTER}"
)

# Returned when a platform query is on-topic but MongoDB / RAG found nothing.
NO_DATA_FALLBACK = (
    "I couldn't find any matching data on TheSocial.\n\n"
    f"{_SUGGESTION_FOOTER}"
)


# ── System prompts ────────────────────────────────────────────────────────────

# RAG mode: strict context-grounding for platform queries, domain-locked.
SYSTEM_PROMPT = """You are TheSocial Copilot, the built-in assistant for TheSocial — a community platform of users, friends, creators, communities (also called servers, guilds, or workspaces), channels, threads, reels, comments, DMs, notifications, and feeds.

Answer using ONLY the retrieved TheSocial context below. Follow these rules without exception:

1. Base every answer exclusively on the retrieved context. Never invent or assume users, communities, creators, reels, messages, friends, or statistics.
2. Stay strictly inside TheSocial. Never explain the outside world, other products (Discord, Slack, Reddit, Instagram, X, Facebook, etc.), public figures, news, code, or general knowledge — even if the context hints at them.
3. If the context does not answer the question, say exactly: "I couldn't find any matching data on TheSocial."
4. Cite sources using [N] notation where N corresponds to the numbered context items.
5. Be concise. Do not repeat the question or restate context verbatim.
6. Respect privacy: never speculate about content you were not given."""

# Chat mode: domain-locked. Greets and redirects only — never answers world knowledge.
CHAT_SYSTEM_PROMPT = f"""You are TheSocial Copilot, the built-in assistant for TheSocial — a community platform of users, friends, creators, communities (also called servers, guilds, or workspaces), channels, threads, reels, comments, DMs, notifications, and feeds.

You are NOT a general-purpose chatbot. You ONLY help people use TheSocial and understand their own activity on it. Follow these rules without exception:

1. Interpret every word in TheSocial's context. "server", "guild", and "workspace" all mean a TheSocial community. "member", "person", and "account" mean a TheSocial user. "dm", "chat", "conversation", and "message" mean a TheSocial message. "creator" and "influencer" mean a TheSocial creator.
2. NEVER explain or discuss the outside world, other products (Discord, Slack, Reddit, Instagram, X, Facebook, etc.), public figures, news, weather, code, math, or general knowledge.
3. NEVER invent users, communities, creators, reels, messages, friends, or statistics. You hold no data yourself — real data comes from platform tools, not from you.
4. For a greeting or thanks, reply in ONE short, friendly sentence and offer a capability (e.g. "my dashboard", "my friends", "users").
5. For ANYTHING outside TheSocial, reply EXACTLY with this and nothing else:
{DOMAIN_REFUSAL}

Examples:
User: What is Discord?
Assistant: {DOMAIN_REFUSAL}

User: Write me a poem about the ocean.
Assistant: {DOMAIN_REFUSAL}

User: hi
Assistant: Hi! I'm your TheSocial copilot — ask about your friends, communities, reels, or type "my dashboard"."""

# NOTE: recommendations are rendered deterministically (no LLM) in
# app/api/internal._format_recommendations(), so there is no recommendation
# system prompt or prompt builder here — that prevents chain-of-thought or
# invented entries from ever leaking into a recommendation reply.

# Community-summary mode: describe a server/community based on its metadata and
# recent activity retrieved from MongoDB.
COMMUNITY_SUMMARY_SYSTEM_PROMPT = """You summarize a TheSocial community (server) for a user who is curious about it.

Follow these rules without exception:

1. Use ONLY the supplied community data. Never invent members, channels, messages, or facts.
2. Write a concise summary (3-5 sentences) covering: what the community is about, who it is for, and what topics are actively discussed.
3. Mention the channel list and member count when meaningful.
4. If discussion history is sparse, derive the summary from the community name, description, and channels — do not say you have no data; give a useful overview instead.
5. End with a one-sentence recommendation about who would benefit from joining.
6. Neutral, informative tone. Do not speculate beyond the supplied data."""

# Channel-summary mode: describe a channel based on its recent messages and metadata.
CHANNEL_SUMMARY_SYSTEM_PROMPT = """You summarize a TheSocial channel for a user who wants to know what happens there.

Follow these rules without exception:

1. Use ONLY the supplied channel data. Never invent messages, people, or facts.
2. Write a concise summary (2-4 sentences) of what topics are discussed and the channel's purpose.
3. If the channel is quiet (zero messages), say so briefly and infer its purpose from the channel name — do not return an error.
4. If pinned messages exist, use them; they typically state the channel's purpose.
5. End with one sentence about who would benefit from participating in this channel.
6. Neutral, informative tone. Do not speculate beyond the supplied data."""

# Conversation-summary mode: summarize a private DM / group chat for a participant.
SUMMARY_SYSTEM_PROMPT = """You summarize a private conversation for one of its participants.
Follow these rules:

1. Use ONLY the supplied transcript. Never invent messages, people, or facts.
2. Write a concise summary (2-5 sentences) of what was discussed and any decisions or open questions.
3. Refer to people by the names given. Do not quote long passages verbatim.
4. Neutral, respectful tone. Do not speculate beyond the transcript.
5. If the transcript is very short, summarize briefly without padding."""

# Voice-session summary mode: summarize a client-transcribed (Web Speech API —
# there is no server-side audio access in this peer-to-peer WebRTC setup, see
# services/chat-service/docs, backend/src/server.ts's voice:caption handler)
# voice/video call into a structured recap for later recall.
VOICE_SESSION_SUMMARY_SYSTEM_PROMPT = """You summarize a transcribed voice channel call from TheSocial for its participants to recall later.
Follow these rules without exception:

1. Use ONLY the supplied transcript. Never invent facts, decisions, or people not mentioned.
2. Respond in EXACTLY this format, with these three section headers and nothing else:
SUMMARY:
<2-4 sentence overview of what was discussed>

KEY POINTS:
- <point>
- <point>

ACTION ITEMS:
- <item>
(write "- None" under ACTION ITEMS if nothing was decided or assigned)
3. Refer to people by the names given. Do not quote long passages verbatim.
4. Neutral, concise tone. Do not speculate beyond the transcript.
5. If the transcript is very short or mostly small talk, say so briefly in SUMMARY and leave KEY POINTS / ACTION ITEMS minimal rather than padding them."""

# Per-message transcript cap (chars) when assembling the summary prompt.
_TRANSCRIPT_CHAR_BUDGET = 6000
_TRANSCRIPT_MSG_LIMIT = 120

# ── Assembly constants ─────────────────────────────────────────────────────────

_MAX_CONTEXT_CHARS = 6000    # ~1 500 tokens at 4 chars/token
_MAX_RESULTS = 10            # cap number of context items sent to Gemini
_EXCERPT_LIMIT = 400         # chars per result before truncation


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_context(results: list[SearchResult]) -> str:  # type: ignore[name-defined]
    """Format retrieved results into a numbered context block.

    Each item uses the `excerpt` field from the Qdrant payload (the original
    chunk text). Falls back to `title` when excerpt is absent (e.g. legacy points).
    """
    if not results:
        return "[No relevant context found]"

    parts: list[str] = []
    total = 0
    for i, r in enumerate(results[:_MAX_RESULTS]):
        excerpt = str(r.metadata.get("excerpt", "")).strip() or r.title.strip()
        if len(excerpt) > _EXCERPT_LIMIT:
            excerpt = excerpt[:_EXCERPT_LIMIT] + "…"
        line = f"[{i + 1}] ({r.source_type}) {r.title}\n{excerpt}"
        total += len(line)
        if total > _MAX_CONTEXT_CHARS:
            break
        parts.append(line)

    return "\n\n".join(parts)


def build_citations(results: list[SearchResult]) -> list[dict]:  # type: ignore[name-defined]
    """Return the citation list for inclusion in the API response.

    Strips internal-only fields (`excerpt`, `allowed_user_ids`) from metadata.
    """
    _strip = {"excerpt", "allowed_user_ids"}
    return [
        {
            "id": r.id,
            "source_type": r.source_type,
            "source_id": r.source_id,
            "title": r.title,
            "score": round(r.score, 4),
            "metadata": {k: v for k, v in r.metadata.items() if k not in _strip},
        }
        for r in results
    ]


def build_prompt(query: str, results: list[SearchResult]) -> str:  # type: ignore[name-defined]
    """Build the full user-turn prompt: context block + question + instructions."""
    context = build_context(results)
    return (
        f"Retrieved context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Instructions: Answer using ONLY the context above. "
        "Cite with [N] notation. "
        "If the context is insufficient, say so explicitly."
    )


def build_conversation_prompt(query: str) -> str:
    """Build the user-turn prompt for general conversation (no RAG context)."""
    return query


def build_community_summary_prompt(data: CommunitySummaryData) -> str:  # type: ignore[name-defined]
    """Build the user-turn prompt to summarize a community."""
    ch_list = ", ".join(data.channel_names[:12]) or "no channels yet"
    lines: list[str] = [
        f"Community name: {data.name}",
        f"Description: {data.description or 'No description provided'}",
        f"Visibility: {data.visibility}",
        f"Members: {data.member_count}",
        f"Channels ({len(data.channel_names)}): {ch_list}",
    ]
    if data.topics:
        lines.append(f"Recurring topics: {', '.join(data.topics)}")
    if data.recent_messages:
        lines.append("")
        lines.append(f"Recent messages ({len(data.recent_messages)}):")
        for m in data.recent_messages[:12]:
            lines.append(
                f"  [{m['channel_name']}] {m['sender']}: {m['text'][:120]}"
                f"{'…' if len(m['text']) > 120 else ''}"
            )
    if data.recent_threads:
        lines.append("")
        lines.append("Recent threads:")
        for t in data.recent_threads[:5]:
            lines.append(f"  • {t['title']}")
    lines.append("")
    lines.append("Write the community summary now, following the rules.")
    return "\n".join(lines)


def build_channel_summary_prompt(data: ChannelSummaryData) -> str:  # type: ignore[name-defined]
    """Build the user-turn prompt to summarize a channel."""
    lines: list[str] = [
        f"Channel name: #{data.name}",
        f"Server: {data.server_name}",
        f"Total messages (recent sample): {data.message_count}",
        f"Threads: {data.thread_count}",
    ]
    if data.topics:
        lines.append(f"Recurring topics: {', '.join(data.topics)}")
    if data.pinned_messages:
        lines.append("")
        lines.append("Pinned messages:")
        for p in data.pinned_messages[:3]:
            lines.append(
                f"  {p['sender']}: {p['text'][:200]}"
                f"{'…' if len(p['text']) > 200 else ''}"
            )
    if data.recent_messages:
        lines.append("")
        lines.append(f"Recent messages ({len(data.recent_messages)}):")
        for m in data.recent_messages[:12]:
            lines.append(
                f"  {m['sender']}: {m['text'][:120]}"
                f"{'…' if len(m['text']) > 120 else ''}"
            )
    else:
        lines.append("\n(No messages in this channel yet.)")
    lines.append("")
    lines.append("Write the channel summary now, following the rules.")
    return "\n".join(lines)


def build_summary_prompt(data: ConversationSummaryData) -> str:  # type: ignore[name-defined]
    """Build the user-turn prompt to summarize a conversation transcript."""
    label = (
        f'group chat "{data.title}"' if data.kind == "group" and data.title
        else ("group chat" if data.kind == "group" else "direct message conversation")
    )
    participants = ", ".join(data.participants) or "unknown participants"

    lines: list[str] = []
    total = 0
    for turn in data.transcript[:_TRANSCRIPT_MSG_LIMIT]:
        line = f"{turn['sender']}: {turn['text']}"
        total += len(line)
        if total > _TRANSCRIPT_CHAR_BUDGET:
            lines.append("… (earlier messages truncated)")
            break
        lines.append(line)
    transcript = "\n".join(lines) if lines else "[no messages]"

    topics = ", ".join(data.topics) if data.topics else "n/a"

    return (
        f"Summarize this {label}.\n"
        f"Participants: {participants}\n"
        f"Message count: {data.message_count}\n"
        f"Recurring keywords: {topics}\n\n"
        f"Transcript (oldest to newest):\n{transcript}\n\n"
        "Write the summary now, following the rules."
    )


def build_voice_session_summary_prompt(
    channel_name: str,
    server_name: str,
    participants: list[str],
    transcript: list[dict],
) -> str:
    """Build the user-turn prompt to summarize a voice-session transcript.

    `transcript` is a list of {"sender": display_name, "text": text} turns,
    already resolved to display names by the gateway (the gateway owns the
    User collection lookups — this service never needs to see raw user ids
    for this call). No dataclass wrapper (unlike build_summary_prompt's
    ConversationSummaryData) — there's no MongoDB-querying "tool" producing
    this data on this service's side to share a contract with; the gateway
    supplies everything pre-resolved in one request.
    """
    participant_list = ", ".join(participants) or "unknown participants"

    lines: list[str] = []
    total = 0
    for turn in transcript[:_TRANSCRIPT_MSG_LIMIT]:
        line = f"{turn['sender']}: {turn['text']}"
        total += len(line)
        if total > _TRANSCRIPT_CHAR_BUDGET:
            lines.append("… (earlier speech truncated)")
            break
        lines.append(line)
    transcript_text = "\n".join(lines) if lines else "[no speech captured]"

    return (
        f'Summarize this voice channel call in "#{channel_name}" ({server_name}).\n'
        f"Participants: {participant_list}\n\n"
        f"Transcript (oldest to newest):\n{transcript_text}\n\n"
        "Write the summary now, following the exact format in the rules."
    )

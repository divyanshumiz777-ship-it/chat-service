"""Capability Registry — the single source of truth for assistant commands.

Every user-facing command (natural language, slash, or palette selection) is
declared ONCE here as a `Capability`. The registry powers:

  • slash commands        — parse_command("/friends") → Intent.LIST_FRIENDS
  • exact-alias routing    — route_intent("my friends") consults this first
  • the help / command menu — render_help_menu()
  • smart suggestions       — suggestions_for(intent)
  • the frontend palette    — capabilities_payload() (served by /capabilities)

Adding a new command is a ONE-OBJECT change: append a Capability below. The
regex router in intent_router.py remains as a fallback for the long tail of
phrasings, so nothing existing breaks — the registry is additive and primary.

Design notes
------------
* `intent` ties a capability to an existing Intent (and thus to the MongoDB tool
  wired in api/internal.py). `tool` names that tool for documentation only.
* Routing uses EXACT, whole-message alias matching (normalized) — never
  substrings — so "suggest friends" can never be swallowed by the "friends"
  capability. Anything not an exact alias falls through to the regex router.
* This module imports `Intent` from intent_router; intent_router imports back
  from here LAZILY (inside route_intent) to avoid an import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.intent_router import Intent


@dataclass(frozen=True)
class Capability:
    """One declarative command definition. Immutable, hashable."""

    name: str                                   # canonical id, e.g. "friends"
    intent: Intent                              # existing Intent it resolves to
    category: str                               # menu grouping
    icon: str                                   # emoji shown in menus
    label: str                                  # human label, e.g. "My Friends"
    command: str                                # canonical text the palette sends
    description: str
    tool: str = ""                              # tool function name (documentation)
    aliases: tuple[str, ...] = ()               # exact whole-message NL forms
    slash: tuple[str, ...] = ()                 # slash forms (without the leading /)
    examples: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()           # related command names (for preview)
    required_permissions: tuple[str, ...] = ()  # reserved; all are gateway-authed today
    hidden: bool = False                        # excluded from menus/palette if True
    long_description: str = ""                  # detailed paragraph for the preview panel
    argument_hint: str = ""                     # placeholder shown for parametric commands
    returns: tuple[str, ...] = ()               # what the command surfaces (for preview)


# Category display order for the help menu and palette.
CATEGORY_ORDER: tuple[str, ...] = (
    "Social", "Messages", "Content", "Search", "Recommendations", "Insights", "Help",
)


# ── The registry ────────────────────────────────────────────────────────────────
# One object per command. This is the ONLY place to edit when adding a command.
CAPABILITIES: tuple[Capability, ...] = (
    # ── Social ──────────────────────────────────────────────────────────────────
    Capability(
        name="friends", intent=Intent.LIST_FRIENDS, category="Social",
        icon="👥", label="My Friends", command="my friends",
        description="Your current friends, with mutual servers and recent activity.",
        long_description=(
            "Browse your complete friend list with online status, mutual servers, "
            "and recent activity. Use this as your social hub to quickly jump into "
            "conversations or check who is active right now."
        ),
        returns=("Friend cards", "Online / offline status", "Mutual servers", "Last seen"),
        tool="get_my_friends",
        aliases=("friends", "my friends", "friend list", "who are my friends",
                 "friend activity", "friend requests", "people i am friends with"),
        slash=("friends", "friend list", "friend activity"),
        examples=("my friends", "friend list"),
        suggestions=("suggest_friends", "users", "conversations"),
    ),
    Capability(
        name="users", intent=Intent.USER_SEARCH, category="Social",
        icon="👤", label="Users", command="users",
        description="Browse all registered users on the platform.",
        long_description=(
            "Browse all registered users on TheSocial. Use this as a directory to "
            "discover accounts before connecting. For a targeted search use "
            "Search Users with a name or username."
        ),
        returns=("User cards", "Username & display name", "Bio", "Follow status"),
        tool="search_users",
        aliases=("users", "list users", "show users", "search users",
                 "who is on the platform", "platform users", "members"),
        slash=("users", "list users"),
        examples=("users", "find divyanshu"),
        suggestions=("search_users", "friends", "suggest_friends"),
    ),
    Capability(
        name="communities", intent=Intent.LIST_COMMUNITIES, category="Social",
        icon="🏘️", label="Communities", command="my communities",
        description="Communities you've joined, with role and member count.",
        long_description=(
            "List every community (server) you have joined, including your role, "
            "member count, and active channels. Great for getting a bird's-eye view "
            "of your entire social graph on TheSocial."
        ),
        returns=("Community cards", "Your role & permissions", "Member count", "Active channels"),
        tool="get_my_communities",
        aliases=("communities", "my communities", "community", "servers",
                 "discord servers", "community members", "community owner",
                 "community activity", "community channels", "community list",
                 "server list"),
        slash=("communities", "community", "community members", "community owner",
               "community activity", "community channels"),
        examples=("my communities", "community members"),
        suggestions=("suggest_communities", "search_communities", "community_summary"),
    ),
    Capability(
        name="community_lookup", intent=Intent.COMMUNITY_LOOKUP, category="Social",
        icon="🏘️", label="My Membership", command="which communities am i in",
        description="Check which communities you belong to.",
        tool="lookup_my_community", hidden=True,
        aliases=("in which community am i", "am i in any community",
                 "which community do i belong"),
        suggestions=("communities", "community_summary"),
    ),
    Capability(
        name="community_summary", intent=Intent.COMMUNITY_SUMMARY, category="Social",
        icon="🏘️", label="Community Summary", command="tell me about this community",
        description=(
            "Get a summary of a community — purpose, active members, topics discussed."
        ),
        long_description=(
            "Ask the AI to summarize any community: its purpose, most-active members, "
            "popular discussion topics, and recent highlights. "
            "Works best when you name the specific community."
        ),
        returns=("Community overview", "Active members", "Popular topics", "Recent highlights"),
        tool="get_community_summary",
        aliases=(
            "what is this community about",
            "what topics are discussed",
            "tell me about this community",
            "community overview",
            "community summary",
            "summarize this community",
        ),
        slash=("community summary", "about community"),
        examples=("tell me about TheSocial", "summarize TheSocial community"),
        suggestions=("communities", "channel_summary", "suggest_communities"),
    ),
    Capability(
        name="channel_summary", intent=Intent.CHANNEL_SUMMARY, category="Social",
        icon="💬", label="Channel Summary", command="summarize #general",
        description=(
            "Get a summary of a channel — topics, pinned messages, key participants."
        ),
        long_description=(
            "Summarize what happens in a specific channel: what topics are pinned, "
            "what is recently discussed, and which members are most active. "
            "Mention the channel name (e.g. #general) for best results."
        ),
        returns=("Channel overview", "Key topics", "Pinned messages", "Active participants"),
        tool="get_channel_summary",
        aliases=(
            "what happens in #general",
            "summarize a channel",
            "channel overview",
        ),
        slash=("channel summary",),
        examples=("summarize #general", "what happens in #notes", "tell me about #social-zone"),
        suggestions=("communities", "community_summary"),
    ),
    # ── Messages ──────────────────────────────────────────────────────────────────
    Capability(
        name="conversations", intent=Intent.CONVERSATION_INSIGHTS, category="Messages",
        icon="💬", label="Recent Messages", command="my messages",
        description="Your recent direct-message conversations.",
        long_description=(
            "See a timeline of your recent direct messages and group chats. "
            "Quickly catch up on conversations you have not opened yet, "
            "with message previews and unread counts."
        ),
        returns=("Conversation list", "Last message preview", "Unread count", "Timestamp"),
        tool="get_conversation_insights",
        aliases=("messages", "my messages", "recent messages", "conversations",
                 "my conversations", "recent conversations", "my chats",
                 "show my chats", "recent chats", "who have i talked to"),
        slash=("messages", "recent messages", "conversations"),
        examples=("my messages", "recent messages"),
        suggestions=("summarize_chat", "friends", "dashboard"),
    ),
    Capability(
        name="summarize_chat", intent=Intent.SUMMARIZE_CONVERSATION, category="Messages",
        icon="📝", label="Summarize Chat", command="summarize this conversation",
        description="Summarize the current direct message or group chat.",
        long_description=(
            "Ask the AI to read and summarize the current DM or group chat. "
            "Useful when you are returning to a long thread and want a quick recap "
            "of decisions, action items, and key moments."
        ),
        returns=("Summary paragraph", "Key decisions", "Action items", "Participants"),
        tool="build_conversation_summary",
        aliases=("summarize chat", "summarize conversation",
                 "summarize this conversation", "recap this chat"),
        slash=("summarize", "summarize chat"),
        examples=("summarize this conversation",),
        suggestions=("conversations", "friends"),
    ),
    # ── Content ───────────────────────────────────────────────────────────────────
    Capability(
        name="creators", intent=Intent.LIST_CREATORS, category="Content",
        icon="🎬", label="Creators", command="my creators",
        description="Creators you follow.",
        long_description=(
            "See every creator account you follow, with their latest reel activity "
            "and engagement metrics. Use this to keep track of your favourite "
            "content makers on TheSocial."
        ),
        returns=("Creator cards", "Reel count", "Follower count", "Latest reel"),
        tool="get_followed_creators",
        aliases=("creators", "creator", "my creators", "followed creators",
                 "who do i follow", "who am i following", "creators i follow",
                 "my following"),
        slash=("creators", "followed creators"),
        examples=("my creators", "followed creators"),
        suggestions=("suggest_creators", "search_reels"),
    ),
    Capability(
        name="search_reels", intent=Intent.PLATFORM_SEARCH, category="Content",
        icon="🎬", label="Search Reels", command="search reels ",
        description='Search reels by topic (e.g. "search reels ai").',
        long_description=(
            "Search the entire reel library by topic, keyword, or creator name. "
            "Results are ranked by relevance and engagement score."
        ),
        argument_hint="topic or keyword  — e.g. ai, travel, gaming",
        returns=("Reel cards", "Creator info", "Engagement score", "Relevance rank"),
        tool="search_all",
        slash=("search reels", "reels"),
        examples=("search reels ai", "search reels travel", "search reels gaming"),
        suggestions=("creators", "suggest_creators"),
    ),
    # ── Search ────────────────────────────────────────────────────────────────────
    Capability(
        name="search_users", intent=Intent.USER_SEARCH, category="Search",
        icon="🔍", label="Search Users", command="search users ",
        description='Find a person by name (e.g. "search users divyanshu").',
        long_description=(
            "Search for a specific person by username or display name. "
            "Results respect your privacy settings and show mutual connections."
        ),
        argument_hint="username or display name",
        returns=("User cards", "Bio", "Mutual friends", "Follow status"),
        tool="search_users",
        slash=("search users",),
        examples=("search users divyanshu", "search users alex", "find divyanshu"),
        suggestions=("users", "friends", "suggest_friends"),
    ),
    Capability(
        name="search_communities", intent=Intent.PLATFORM_SEARCH, category="Search",
        icon="🔍", label="Search Communities", command="search communities ",
        description="Search communities by name or topic.",
        long_description=(
            "Search for public communities by name or topic. "
            "Results include member count and activity level to help you "
            "find the most relevant community to join."
        ),
        argument_hint="topic or community name  — e.g. gaming, AI, music",
        returns=("Community cards", "Member count", "Activity level", "Join button"),
        tool="search_all",
        slash=("search communities",),
        examples=("search communities gaming", "search communities AI", "search communities music"),
        suggestions=("communities", "suggest_communities"),
    ),
    Capability(
        name="search_messages", intent=Intent.PLATFORM_SEARCH, category="Search",
        icon="🔍", label="Search Messages", command="search messages ",
        description="Search your messages (access-scoped).",
        long_description=(
            "Search your own messages and conversations for a keyword or phrase. "
            "Results are scoped to conversations you have access to — "
            "your DMs, group chats, and community channels you are a member of."
        ),
        argument_hint="keyword or phrase  — e.g. meeting, deployment, react",
        returns=("Message snippets", "Conversation context", "Timestamp", "Sender"),
        tool="search_all",
        slash=("search messages",),
        examples=("search messages react", "search messages meeting", "search messages deployment"),
        suggestions=("conversations", "summarize_chat"),
    ),
    # ── Recommendations ───────────────────────────────────────────────────────────
    Capability(
        name="suggest_friends", intent=Intent.SUGGEST_FRIENDS, category="Recommendations",
        icon="✨", label="Suggest Friends", command="suggest friends",
        description="People you may want to connect with.",
        long_description=(
            "Get AI-powered friend suggestions based on mutual connections, "
            "shared communities, and interaction history. "
            "The more you use TheSocial, the better the suggestions become."
        ),
        returns=("Suggested profiles", "Mutual friends count", "Shared communities", "Connect button"),
        tool="suggest_friends",
        aliases=("suggest friends", "recommend friends", "friend suggestions",
                 "people i may know", "who should i connect with"),
        slash=("suggest friends",),
        examples=("suggest friends",),
        suggestions=("friends", "users"),
    ),
    Capability(
        name="suggest_communities", intent=Intent.SUGGEST_COMMUNITIES, category="Recommendations",
        icon="✨", label="Suggest Communities", command="suggest communities",
        description="Communities you may want to join.",
        long_description=(
            "Get community recommendations tailored to your interests, activity, "
            "and the communities your friends are already part of. "
            "Discover active spaces you might be missing."
        ),
        returns=("Community cards", "Why recommended", "Member count", "Join button"),
        tool="suggest_communities",
        aliases=("suggest communities", "recommend communities",
                 "communities to join", "which community should i join"),
        slash=("suggest communities",),
        examples=("suggest communities",),
        suggestions=("communities", "search_communities"),
    ),
    Capability(
        name="suggest_creators", intent=Intent.SUGGEST_CREATORS, category="Recommendations",
        icon="✨", label="Suggest Creators", command="suggest creators",
        description="Creators you may want to follow.",
        long_description=(
            "Discover new creators based on what you already follow, "
            "what is trending across the platform, "
            "and what your network is engaging with most."
        ),
        returns=("Creator cards", "Why recommended", "Follower count", "Follow button"),
        tool="suggest_creators",
        aliases=("suggest creators", "recommend creators", "who should i follow",
                 "people to follow", "creator suggestions"),
        slash=("suggest creators",),
        examples=("suggest creators",),
        suggestions=("creators", "search_reels"),
    ),
    # ── Insights ──────────────────────────────────────────────────────────────────
    Capability(
        name="dashboard", intent=Intent.SOCIAL_OVERVIEW, category="Insights",
        icon="📊", label="Dashboard", command="dashboard",
        description="Your social overview: friends, communities, activity.",
        long_description=(
            "Your personal social dashboard: friends online now, recent community "
            "updates, latest messages, and quick actions — all in one place. "
            "The best starting point for your day on TheSocial."
        ),
        returns=("Active friends", "Recent communities", "Latest messages", "Quick actions"),
        tool="get_social_overview",
        aliases=("dashboard", "my dashboard", "overview", "platform overview",
                 "my activity", "my stats", "my social stats", "social overview"),
        slash=("dashboard", "overview"),
        examples=("dashboard", "overview"),
        suggestions=("friends", "communities", "conversations"),
    ),
    # ── Help ──────────────────────────────────────────────────────────────────────
    Capability(
        name="help", intent=Intent.HELP, category="Help",
        icon="❓", label="Help", command="help",
        description="Show everything the copilot can do.",
        long_description=(
            "Display the full command menu, grouped by category, "
            "with icons, descriptions, and usage examples for every available command. "
            "Type @ to open the command palette at any time."
        ),
        returns=("Command list", "Category grouping", "Usage examples"),
        tool="-",
        aliases=("help", "commands", "what can you do", "assistant help",
                 "show commands", "show menu", "show shortcuts", "@"),
        slash=("help", "commands"),
        examples=("help", "what can you do"),
        suggestions=("dashboard", "friends", "users"),
    ),
)


# ── Normalization + indexes (built once at import) ────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text.strip(" \t\r\n?.!,")


def _build_index(attr: str) -> dict[str, Capability]:
    index: dict[str, Capability] = {}
    for cap in CAPABILITIES:
        for form in getattr(cap, attr):
            key = _norm(form)
            if key and key not in index:   # first registration wins
                index[key] = cap
    return index


_ALIAS_INDEX: dict[str, Capability] = _build_index("aliases")
_SLASH_INDEX: dict[str, Capability] = _build_index("slash")

# First capability per intent — used for suggestions lookup.
_BY_INTENT: dict[Intent, Capability] = {}
for _cap in CAPABILITIES:
    _BY_INTENT.setdefault(_cap.intent, _cap)

# Capabilities by name — used for related-command lookup.
_BY_NAME: dict[str, Capability] = {c.name: c for c in CAPABILITIES}


# ── Parsed slash command ──────────────────────────────────────────────────────────

@dataclass
class ParsedCommand:
    """Result of parsing a slash command into an intent + extracted args."""

    intent: Intent
    term: str | None = None              # USER_SEARCH person term
    query: str | None = None             # PLATFORM_SEARCH query (overrides message)
    types: list[str] | None = None       # PLATFORM_SEARCH source-type filter


# "/search <scope> <query>" scopes → (intent, source types).
_SEARCH_SCOPES: dict[str, tuple[Intent, list[str] | None]] = {
    "user": (Intent.USER_SEARCH, None),
    "users": (Intent.USER_SEARCH, None),
    "people": (Intent.USER_SEARCH, None),
    "member": (Intent.USER_SEARCH, None),
    "members": (Intent.USER_SEARCH, None),
    "reel": (Intent.PLATFORM_SEARCH, ["reel"]),
    "reels": (Intent.PLATFORM_SEARCH, ["reel"]),
    "community": (Intent.PLATFORM_SEARCH, ["server", "channel", "thread"]),
    "communities": (Intent.PLATFORM_SEARCH, ["server", "channel", "thread"]),
    "server": (Intent.PLATFORM_SEARCH, ["server", "channel", "thread"]),
    "servers": (Intent.PLATFORM_SEARCH, ["server", "channel", "thread"]),
    "message": (Intent.PLATFORM_SEARCH, ["message"]),
    "messages": (Intent.PLATFORM_SEARCH, ["message"]),
    "chat": (Intent.PLATFORM_SEARCH, ["message"]),
    "chats": (Intent.PLATFORM_SEARCH, ["message"]),
}


def parse_command(message: str) -> ParsedCommand | None:
    """Parse a slash command. Returns None when `message` is not a slash command.

    Handles "/search <scope> <query>" specially (extracting the scope + query) and
    resolves all other "/x" forms against the registry's slash + alias indexes.
    Pure function, no I/O.
    """
    raw = (message or "").strip()
    if not raw.startswith("/"):
        return None
    body = raw[1:].strip()
    if not body:
        return None

    tokens = body.split()
    verb = tokens[0].lower()

    # "/search <scope> [query...]" and "/find <scope> [query...]"
    if verb in ("search", "find") and len(tokens) >= 2:
        scope = tokens[1].lower()
        query = " ".join(tokens[2:]).strip()
        mapped = _SEARCH_SCOPES.get(scope)
        if mapped:
            intent, types = mapped
            if intent is Intent.USER_SEARCH:
                return ParsedCommand(intent, term=query or None)
            return ParsedCommand(intent, query=query or scope, types=types)
        # Unknown scope → treat the whole remainder as a general platform search.
        return ParsedCommand(Intent.PLATFORM_SEARCH, query=body[len(verb):].strip() or None)

    # Exact slash form, longest first ("community members" before "community").
    low = _norm(body)
    cap = _SLASH_INDEX.get(low)
    if cap is None:
        cap = _ALIAS_INDEX.get(low)
    if cap is not None:
        return ParsedCommand(cap.intent)
    return None


# ── Public lookups (used by the router and API) ───────────────────────────────────

def lookup_capability_by_alias(text: str) -> Capability | None:
    """Exact whole-message alias match (normalized). None if no exact match."""
    return _ALIAS_INDEX.get(_norm(text))


def lookup_capability_by_intent(intent: Intent) -> Capability | None:
    return _BY_INTENT.get(intent)


def lookup_capability_by_name(name: str) -> Capability | None:
    return _BY_NAME.get(name)


def suggestions_for(intent: Intent) -> list[str]:
    """Smart-suggestion command strings for a resolved intent (may be empty)."""
    cap = _BY_INTENT.get(intent)
    return list(cap.suggestions) if cap else []


# ── Rendering ─────────────────────────────────────────────────────────────────────

def render_help_menu() -> str:
    """Render the full command menu, grouped by category, with icons (no LLM)."""
    lines: list[str] = [
        "🤖 TheSocial Copilot",
        "Type @ for the command palette, / for slash commands, or just ask.",
        "",
    ]
    for category in CATEGORY_ORDER:
        if category == "Help":
            continue  # the menu itself is the help; don't list "help" inside it
        caps = [c for c in CAPABILITIES if c.category == category and not c.hidden]
        if not caps:
            continue
        lines.append(category)
        for cap in caps:
            lines.append(f"{cap.icon} {cap.label} — {cap.description}")
        lines.append("")
    return "\n".join(lines).strip()


def capabilities_payload() -> list[dict]:
    """Serialize visible capabilities for the frontend command palette.

    The palette filters this list locally (never calls the backend per keystroke);
    it is fetched once and cached. This keeps the registry the single source of
    truth while leaving autocomplete fully client-side.

    New fields (long_description, argument_hint, returns, suggestions) power the
    rich preview panel — all derived from capability metadata, no extra API calls.
    """
    return [
        {
            "name": c.name,
            "label": c.label,
            "icon": c.icon,
            "category": c.category,
            "command": c.command,
            "description": c.description,
            "aliases": list(c.aliases),
            "slash": list(c.slash),
            "examples": list(c.examples),
            "suggestions": list(c.suggestions),
            "longDescription": c.long_description,
            "argumentHint": c.argument_hint,
            "returns": list(c.returns),
        }
        for c in CAPABILITIES
        if not c.hidden
    ]

"""Lightweight intent router for the AI chat endpoint.

Classifies user messages into one of nineteen intents using regex/keyword matching.
No LLM call, no embeddings — O(1) per message.

Intents:
  chat                     — greetings only; domain-locked (no world knowledge)
  platform_search          — find/search for platform content (communities, reels)
  recommendation           — suggest/recommend content (generic, e.g. reels)
  suggest_friends          — recommend NEW people to connect with (MongoDB tool)
  suggest_communities      — recommend NEW communities to join (MongoDB tool)
  suggest_creators         — recommend NEW creators to follow (MongoDB tool)
  trending_creators        — most popular / fastest-growing creators platform-wide (MongoDB tool)
  list_friends             — list EXISTING friends (MongoDB tool, not recommendations)
  list_communities         — list JOINED communities (MongoDB tool, not recommendations)
  community_lookup         — membership check / specific community search (MongoDB tool)
  list_creators            — list FOLLOWED creators (MongoDB tool, not recommendations)
  user_search              — global people search: "users", "find <name>" (MongoDB tool)
  social_overview          — dashboard / aggregate social stats (MongoDB tool)
  conversation_insights    — list recent DM conversations with insights (MongoDB tool)
  conversation_with_user   — a DM with one named person, e.g. "chat with divyanshu" (MongoDB tool)
  summarize                — summarize unread / community activity
  summarize_conversation   — summarize the current/open DM or group chat
  community_summary        — summarize a community (server) — metadata + recent activity (MongoDB tool)
  channel_summary          — summarize a channel — messages + pinned + threads (MongoDB tool)
  voice_session_recall     — recap the last voice/call session in a channel (MongoDB tool)
  mixed                    — platform query fused with general explanation request
  qa                       — default: platform-scoped question answered via RAG
"""
from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    CHAT = "chat"
    PLATFORM_SEARCH = "platform_search"
    RECOMMENDATION = "recommendation"
    SUGGEST_FRIENDS = "suggest_friends"
    SUGGEST_COMMUNITIES = "suggest_communities"
    SUGGEST_CREATORS = "suggest_creators"
    TRENDING_CREATORS = "trending_creators"
    LIST_FRIENDS = "list_friends"
    LIST_COMMUNITIES = "list_communities"
    COMMUNITY_LOOKUP = "community_lookup"
    LIST_CREATORS = "list_creators"
    USER_SEARCH = "user_search"
    SOCIAL_OVERVIEW = "social_overview"
    CONVERSATION_INSIGHTS = "conversation_insights"
    CONVERSATION_WITH_USER = "conversation_with_user"
    SUMMARIZE = "summarize"
    # Concrete resolution of SUMMARIZE ("what did I miss", "summarize unread").
    # Never produced by the router directly — internal._resolve_umbrella_intent
    # maps SUMMARIZE onto it so the phrasing reaches the digest tool that
    # already backs the Digest tab and the proactive modal.
    CATCH_ME_UP = "catch_me_up"
    SUMMARIZE_CONVERSATION = "summarize_conversation"
    COMMUNITY_SUMMARY = "community_summary"
    CHANNEL_SUMMARY = "channel_summary"
    VOICE_SESSION_RECALL = "voice_session_recall"
    HELP = "help"
    # "How does TheSocial work?" — answered from the hand-written help corpus
    # (content/help/*.md). Distinct from HELP, which lists available commands.
    PRODUCT_HELP = "product_help"
    QA = "qa"
    MIXED = "mixed"


# Words whose presence signals the message is about platform content.
# Absence of all of these → treat the message as general conversation.
_PLATFORM_WORDS = frozenset({
    "community", "communities", "channel", "channels",
    "reel", "reels", "creator", "creators",
    "friend", "friends", "message", "messages",
    "notification", "notifications", "server", "servers",
    "unread", "user", "users", "member", "members",
    "profile", "dm", "group", "post", "posts",
    "the social",
    # Added: without these, natural phrasings degraded to Intent.CHAT (the
    # domain-locked "greetings only" path) and produced a refusal. "Search
    # people" and "Explain this platform" — both shipped as UI buttons —
    # were failing on exactly this.
    "people", "person", "platform", "profiles", "thesocial",
})

# Summarize the *current* conversation (the open DM / group chat). Checked
# before everything else because phrases like "summarize this chat" carry no
# platform keyword and would otherwise fall through to general conversation.
# Two shapes:
#   1. summarize-verb + demonstrative ("this"/"current"/"our"/"the") + any chat noun
#   2. summarize-verb + a strong conversation noun (conversation/chat/dm/discussion)
# "summarize unread messages" deliberately does NOT match (no demonstrative,
# "messages" is not a strong noun) so it stays routed to the unread summarizer.
# Tool-based recommenders. Detected before the platform gate (some phrases, e.g.
# "people I may know" / "who should I follow", carry no platform keyword) and
# routed to the dedicated MongoDB recommender tools — never RAG. Deliberately
# require "suggest"/"recommend" (or an explicit "who should I…/people I may
# know" phrase) so plain searches like "find AI communities" stay PLATFORM_SEARCH.
# "How does TheSocial work?" — product/how-to questions, answered from the help
# corpus. Requires an explicit how-to shape ("how do I", "what is a", "explain",
# "where do I") so it can't swallow data questions like "what did I miss" or
# "who are my friends", which are checked either before this or by their own
# more specific rules.
_PRODUCT_HELP_RE = re.compile(
    r"^\s*(how (do|can|does|would) (i|you|it|we)\b"
    r"|how (do i|to)\b"
    # "what is a/an <noun>" is a product question ("what is a thread"), but
    # "what is THE <noun>" is usually not ("what is the weather") — that must
    # stay on the domain-refusal path, so `the` is deliberately excluded here
    # and only allowed in the explicit "difference between" shape below.
    r"|what (is|are) (a|an)\b"
    r"|what'?s (a|an)\b"
    r"|what(?:'s| is) the difference between\b"
    r"|where (do|can) i\b"
    r"|explain\b"
    r"|tell me about (this|the) (platform|app|product|site)\b"
    r"|how does (this|the) (platform|app|product|site)\b)",
    re.IGNORECASE,
)

# "While you were away" phrasings. These are the digest question however the
# user words it, including when they name a surface ("...in my communities"),
# which is why this is checked ahead of every list/summary rule.
_CATCH_ME_UP_RE = re.compile(
    r"\bwhat('?s| is| did| have)?\s*(i|we)?\s*(miss|missed|been missing)\b"
    r"|\bcatch me up\b"
    r"|\bcatch up on\b"
    r"|\bwhat('s| is| has)?\s+(new|happened)\b.{0,30}\b(while|since)\b"
    r"|\bwhile i (was|were) (away|gone|offline)\b"
    r"|\bsummari[sz]e\s+(my\s+)?unread\b"
    r"|\banything new\b",
    re.IGNORECASE,
)

_SUGGEST_FRIENDS_RE = re.compile(
    r"\b(suggest|recommend)\b.*\b(friends?|people|connections?)\b"
    r"|\bpeople i (may|might) know\b"
    r"|\bwho (should|can|could|do) i (connect|add|friend|know)\b"
)
_SUGGEST_COMMUNITIES_RE = re.compile(
    r"\b(suggest|recommend)\b.*\b(communit(y|ies)|servers?)\b"
    r"|\b(what|which) communit(y|ies) (should|can|could|do) i join\b"
    r"|\bcommunit(y|ies) (i should|to) join\b"
)
_SUGGEST_CREATORS_RE = re.compile(
    r"\b(suggest|recommend)\b.*\b(creators?)\b"
    # "who should/can/could I follow" — future-tense recommendation intent only.
    # "who do I follow" is a listing query and routes to LIST_CREATORS instead.
    r"|\bwho (should|can|could) i follow\b"
    r"|\bpeople to follow\b",
    re.IGNORECASE,
)
# Platform-wide, non-personalized ranking — distinct from SUGGEST_CREATORS
# (personalized to the asker's history) and LIST_CREATORS (who THEY follow).
_TRENDING_CREATORS_RE = re.compile(
    r"\b(trending|hot|top|popular)\b.{0,15}\bcreators?\b"
    r"|\bcreators?\b.{0,15}\b(trending|popular)\b"
    r"|\bmost\s+popular\s+creators?\b",
    re.IGNORECASE,
)

# ── Social graph listing — ACTUAL data, not recommendations ─────────────────
# These are checked AFTER the explicit recommenders so "suggest friends" stays
# routed to SUGGEST_FRIENDS, and AFTER conversation intents so "my friends I
# chatted with" still routes to CONVERSATION_INSIGHTS.

_LIST_FRIENDS_RE = re.compile(
    r"\bmy\s+friends?\b"
    r"|\b(show|list|view|see|who\s+are|display)\b.{0,15}\bfriends?\b"
    r"|\bfriend\s+list\b"
    r"|\bfriends?\s+(activity|requests?)\b"   # "friend activity", "friend requests"
    r"|\bpeople\s+i\s+am\s+friends\s+with\b"
    r"|\bfriends?\s*$",          # bare "friend(s)" at end of input
    re.IGNORECASE,
)

_LIST_COMMUNITIES_RE = re.compile(
    r"\bmy\s+(communities|community|servers?)\b"
    r"|\b(show|list|view|see|display)\b.{0,20}\b(communities|community|servers?)\b"
    r"|\bwhich\s+(communities|community|servers?)\s+am\s+i\s+(in|part\s+of|member\s+of)\b"
    r"|\b(communities|community|servers?)\s+(i\s+)?(joined|belong\s+to|am\s+in|am\s+a\s+member)\b"
    r"|\b(community|server)\s+list\b"
    r"|^\s*(communit(y|ies)|servers?)\s*$"          # bare "community" / "servers"
    r"|^\s*discord\s+servers?\s*$"                    # bare "discord server(s)"
    # community/server detail phrases — the tool returns role, members & owner
    r"|\b(community|server)\s+(members?|owner|stats?|channels?|activity|info)\b",
    re.IGNORECASE,
)

# "in which community i am" / "am I in any community" — more of a membership
# check phrasing than a list request; kept as a separate intent for routing
# granularity even though the underlying tool returns the same data.
_COMMUNITY_LOOKUP_RE = re.compile(
    r"\bin\s+which\s+(community|server|communities)\s+(am\s+i|i\s+am)\b"
    r"|\bam\s+i\s+(in|a\s+member\s+of)\s+(any|a)\s+(community|server)\b"
    r"|\bwhich\s+(community|server)\s+do\s+i\s+belong\b",
    re.IGNORECASE,
)

_LIST_CREATORS_RE = re.compile(
    r"\bmy\s+(creators?|followed\s+creators?)\b"
    r"|\b(show|list|view|see|display)\b.{0,20}\b(creators?)\b"
    r"|\bwho\s+do\s+i\s+follow\b"
    r"|\bwho\s+am\s+i\s+following\b"
    r"|\bmy\s+following\b"
    r"|\bcreators?\s+i\s+follow\b"
    r"|\bfollowed\s+creators?\b"          # "followed creators" (no "my")
    r"|^\s*creators?\s*$",                # bare "creator(s)"
    re.IGNORECASE,
)

# Global people search — "users", "list users", "who is on the platform", and
# "find/search <name>". MongoDB tool, ACL-safe. Two shapes share one intent:
#   • list mode  — no name; returns recent users
#   • find mode  — a person name follows "find/search"
# Checked before the platform gate so "users" (a platform word) routes to the
# MongoDB people tool, not RAG (Part 3: MongoDB tools take priority).
_USER_SEARCH_LIST_RE = re.compile(
    r"^\s*users?\s*$"
    r"|^\s*members?\s*$"
    r"|\b(list|show|view|see|display|all|search|find|browse)\s+(the\s+|all\s+)?users?\b"
    r"|\b(list|show|view|see|display|all)\s+(the\s+|all\s+)?members?\b"
    r"|\bwho\s+(is|are)\s+(on|using|in)\s+(the\s+)?(platform|thesocial|the\s+social)\b"
    r"|\bwho(?:'?s| is)\s+on\s+(thesocial|the\s+social)\b"
    r"|\bplatform\s+users?\b"
    r"|\bwho\s+are\s+(?:the\s+)?users?\b",
    re.IGNORECASE,
)
# "find <name>" / "search for <name>" — the trailing capture is a person name.
_FIND_PERSON_RE = re.compile(
    r"\b(?:find|search\s+for|look\s+for|search)\s+(?P<name>[a-z0-9][a-z0-9 ._'-]{0,38})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_SOCIAL_OVERVIEW_RE = re.compile(
    # Dashboard / overview (Part 12) — same aggregate data, rendered as a dashboard.
    r"\bdashboard\b"
    r"|\boverview\b"
    r"|\bsocial\s+(overview|summary|stats?|activity|status)\b"
    r"|\bplatform\s+(overview|summary|stats?)\b"
    r"|\bmy\s+(social\s+)?(activity|stats?|summary)\b"
    r"|\bsummar[iy]z[e]?\s+my\s+social\b"
    r"|\bwhat\s+am\s+i\s+(doing|up\s+to)\s+(on|in)\s+(the\s+)?platform\b"
    r"|\bmy\s+platform\s+(activity|stats?|overview)\b",
    re.IGNORECASE,
)

# Conversation WITH a specific named person, e.g. "my conversation with divyanshu",
# "chat with alex", "messages with john doe". The trailing capture is the person
# name. Generic targets ("friends", "people", "someone") are NOT person names and
# are excluded (those route to conversation insights instead). Checked before the
# platform gate — these phrases carry no platform keyword.
_CONVERSATION_WITH_RE = re.compile(
    r"\b(?:conversation|conversations|chat|chats|chatted|messages?|dms?|talk|"
    r"talked|talking|discussion|texts?|texting)\s+with\s+(?P<name>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
# Words that follow "with" but denote *all* contacts, not one specific person.
_GENERIC_TARGETS = frozenset({
    "friend", "friends", "people", "someone", "anyone", "everyone",
    "somebody", "anybody", "others", "them", "all", "everybody",
    # Pronouns: not a specific person name.
    "me", "myself", "you", "yourself", "him", "her", "us", "we", "they",
})

# Conversation insights: list the user's DM conversations with activity details.
# Checked before the platform gate — many phrases ("show my chats", "my recent
# conversations", "my conversation") contain no platform keyword and would
# otherwise fall through to CHAT.
_CONVERSATION_INSIGHTS_RE = re.compile(
    # "show/list/view/see my chats/dms/messages/conversations"
    r"\b(show|list|view|see|display)\s+(my\s+)?(messages?|chats?|dms?|conversations?)\b"
    # bare possessive — "my conversation(s)/chat(s)/dm(s)/messages"
    r"|\bmy\s+(messages?|conversations?|chats?|dms?)\b"
    # "my recent/latest/active/past messages…"
    r"|\bmy\s+(recent|latest|active|old|past|previous)\s+(messages?|chats?|dms?|conversations?|friends?)\b"
    # Hinglish — "mere messages" means "my messages" in Hindi
    r"|\bmere\s+(messages?|chats?)\b"
    # "who have I talked/chatted/spoken to/with"
    r"|\bwho\s+have\s+i\s+(talked?|chatted?|spoken?)\s*(to|with)?\b"
    # "who am I chatting/talking/messaging with"
    r"|\bwho\s+am\s+i\s+(chatting|talking|messaging)\s+with\b"
    # "friends I chatted/talked/messaged with"
    r"|\bfriends?\s+(i\s+)?(chatted?|talked?|messaged?|spoke)\s*(with|to)?\b"
    # "show active conversations"
    r"|\bshow\s+active\s+conversations?\b"
    # bare "message(s)"
    r"|^\s*messages?\s*$"
    # "recent/latest/active/new/old/past chats/dms/conversations/messages"
    r"|\b(recent|latest|active|new|old|past)\s+(chats?|dms?|conversations?|messages?)\b",
    re.IGNORECASE,
)

_SUMMARIZE_VERBS = r"(summari[sz]e|summary|recap|tl;?dr|catch me up|brief me)"
_SUMMARIZE_CONVERSATION_RE = re.compile(
    rf"\b{_SUMMARIZE_VERBS}\b.*\b(this|current|our|the)\s+"
    r"(conversation|chat|dm|discussion|thread|messages?)\b"
    rf"|\b{_SUMMARIZE_VERBS}\b.*\b(conversation|chat|dm|discussion)\b"
)

# Voice-session recall — recap of a PAST voice/call session, distinct from
# CHANNEL_SUMMARY's present-tense "what happens in X channel" (ongoing text
# activity). Requires recall framing (past tense "what happened"/"what was
# discussed", or recap/summarize) PLUS an explicit call/voice/meeting/session
# marker, so a plain "summarize #general" still routes to CHANNEL_SUMMARY.
# Checked BEFORE CHANNEL_SUMMARY — "recap the voice session" would otherwise
# match CHANNEL_SUMMARY's looser "recap ... channel" pattern once a channel
# name is mentioned nearby.
_VOICE_SESSION_RECALL_RE = re.compile(
    r"\bwhat\s+(?:happened|was\s+(?:discussed|said|talked\s+about)|did\s+we\s+(?:talk|discuss)\s+about)\b"
    r".{0,40}\b(?:call|voice(?:\s+(?:call|chat|session))?|meeting|session)\b"
    r"|\b(?:recap|summar[iy]ze?)\b.{0,30}\b(?:call|voice\s+(?:call|chat|session)|meeting|voice\s+session)\b"
    r"|\b(?:call|voice\s+(?:call|chat)|meeting|voice\s+session)\b.{0,20}\b(?:recap|summary)\b",
    re.IGNORECASE,
)

# ── Community / channel summary (MongoDB tools — never RAG) ────────────────────
# Checked before the platform gate because many channel names (#notes, #general)
# carry no _PLATFORM_WORDS token and would otherwise fall through to CHAT.
# CHANNEL_SUMMARY is checked before COMMUNITY_SUMMARY because # is a stronger
# channel signal than the looser community patterns.

# Channel summary — any query that contains a #tag OR asks "what happens in X
# channel" / "summarize [the] X channel".
_CHANNEL_SUMMARY_RE = re.compile(
    # Explicit #channelname (strongest signal)
    r"#[\w][\w-]{0,40}"
    # "summarize/recap/tell me about [the] <name> channel"
    r"|\b(?:summarize?|recap|tell\s+me\s+about)\b.{0,30}\bchannel\b"
    # "what [happens/is discussed/goes on] in [the] #<name>"
    r"|\bwhat\s+(?:happens?|is\s+discussed|goes\s+on)\s+in\s+(?:the\s+)?#?[\w][\w-]{0,40}\b"
    # "who should join [the] <name> channel"
    r"|\bwho\s+should\s+(?:join|use|participate\s+in)\b.{0,20}\bchannel\b"
    # "what is #<name>" / "what is the <name> channel"
    r"|\bwhat\s+is\s+(?:the\s+)?#[\w][\w-]{0,40}\b",
    re.IGNORECASE,
)

# Community summary — queries about a named community/server or generic
# "what is this community about" phrasing.
_COMMUNITY_SUMMARY_RE = re.compile(
    # "summarize/tell me about/describe/overview of … community/server"
    r"\b(?:summarize?|tell\s+me\s+about|describe|overview\s+of)\b.{0,50}\b(?:community|server|workspace)\b"
    # … OR the word comes before the verb: "community overview/summary/about"
    r"|\b(?:community|server|workspace)\b.{0,30}\b(?:overview|summary|summarize?|about)\b"
    # Generic phrasing with no specific name
    r"|\bwhat\s+(?:is|are)\s+(?:this|the)\s+community\s+about\b"
    r"|\bwhat\s+topics?\s+(?:are|is)\s+discussed(?:\s+in\s+(?:this|the)\s+community)?\b"
    r"|\bshould\s+i\s+join\s+\w[\w\s.-]{2,40}\b"
    # "what is <name> community" / "about <name> community"
    r"|\bwhat\s+is\s+\w[\w\s.-]{2,30}\s+(?:community|server)\b"
    r"|\babout\s+\w[\w\s.-]{2,30}\s+(?:community|server)\b",
    re.IGNORECASE,
)

# ── Name-extraction regexes (used by extract_channel_name / extract_community_name) ──

_EXTRACT_CHANNEL_HASH_RE = re.compile(r"#([\w][\w-]{0,40})", re.IGNORECASE)
_EXTRACT_CHANNEL_WORD_RE = re.compile(
    r"\bthe\s+([\w-]{2,40})\s+channel\b"
    r"|\b(?:summarize?|recap)\s+(?:the\s+)?([\w-]{2,40})\s+channel\b"
    r"|\bin\s+(?:the\s+)?([\w-]{2,40})\s+channel\b"
    # "what happens/happened/is discussed/was discussed/was said/was talked
    # about in [the] <name>" — the "happened"/"was …" forms feed
    # VOICE_SESSION_RECALL's past-tense phrasing (extract_channel_name is
    # reused for both intents; a channel name is a channel name either way).
    r"|\bwhat\s+(?:happens?|happened|is\s+discussed|was\s+discussed|was\s+said|was\s+talked\s+about)\s+in\s+(?:the\s+)?([\w-]{2,40})\b"
    r"|\bwho\s+should\s+(?:join|use)\s+(?:the\s+)?([\w-]{2,40})\b"
    # Trailing "in [the] <name>" with nothing after it (e.g. "recap the voice
    # session in general") — a last-resort fallback, so it's listed last;
    # only reached when none of the more specific forms above matched.
    r"|\bin\s+(?:the\s+)?([\w-]{2,40})\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_EXTRACT_COMMUNITY_RE = re.compile(
    r"\b(?:summarize?|tell\s+me\s+about|describe|overview\s+of)\s+(?:the\s+)?([A-Za-z][\w\s.-]{1,40}?)(?:\s+community|\s+server|\s*[?!.]|$)"
    r"|\bshould\s+i\s+join\s+([A-Za-z][\w\s.-]{2,40})(?:\s*[?!.]|$)"
    r"|\bwhat\s+is\s+([A-Za-z][\w\s.-]{2,30})\s+(?:community|server)\b",
    re.IGNORECASE,
)
_SKIP_COMMUNITY_WORDS = frozenset({
    "the", "this", "that", "a", "an", "any", "some", "my", "our", "your",
    "community", "server", "workspace", "channel", "platform", "it", "about",
    "what", "which", "that", "there",
})

_SUMMARIZE_RE = re.compile(
    r"\b(summarize?|summary|unread|catch me up|what did i miss|recap)\b"
)
_RECOMMEND_RE = re.compile(
    r"\b(recommend|suggest|suggestion|suggestions|discover|explore)\b"
)
_SEARCH_RE = re.compile(
    r"\b(find|search|look for|show me|list|who|which|any)\b"
)
# Mixed: explicit personal preference + platform action, or explanation of platform entity
_MIXED_RE = re.compile(
    r"(i (like|love|enjoy|prefer)|i'?m interested in).+\b(suggest|recommend|find|show|discover)\b"
    r"|"
    r"\b(suggest|recommend|find).+(like|similar|based on|because)\b"
    r"|"
    r"\bexplain\b.+\b(creator|community|channel|reel|server)\b"
    r"|"
    r"\b(recommend|suggest).+\bwhy\b"
)


def extract_channel_name(message: str) -> str | None:
    """Extract the channel name from a channel-summary or voice-recall message.

    Tries `#channelname` notation first (strongest signal), then falls back to
    "the <name> channel" / "in <name> channel" / "what happened in <name>"
    word patterns.  Returns the lowercased channel name, or None when no
    specific name is found.
    """
    if not message:
        return None
    m = _EXTRACT_CHANNEL_HASH_RE.search(message)
    if m:
        return m.group(1).lower()
    m = _EXTRACT_CHANNEL_WORD_RE.search(message)
    if m:
        for grp in m.groups():
            if grp and grp.lower() not in {"the", "a", "an", "this", "that"}:
                return grp.lower()
    return None


def extract_community_name(message: str) -> str | None:
    """Extract the community name from a community-summary message.

    Matches explicit names after "summarize", "tell me about", "should I join",
    etc.  Returns None when the message is a generic community query with no
    specific name (e.g. "what is this community about").
    """
    if not message:
        return None
    m = _EXTRACT_COMMUNITY_RE.search(message)
    if m:
        for grp in m.groups():
            if grp:
                name = grp.strip(" ?.!,\"'").strip()
                if name.lower() not in _SKIP_COMMUNITY_WORDS and len(name) >= 2:
                    return name
    return None


def _has_platform_context(text: str) -> bool:
    words = frozenset(re.findall(r"\b\w+\b", text))
    return bool(words & _PLATFORM_WORDS)


def _target_match(text: str) -> "re.Match[str] | None":
    """Return the 'conversation with <name>' match iff <name> is a real person.

    Filters out generic targets ("friends", "people", …) and over-long captures
    (likely full sentences, not names) so only specific people qualify.
    """
    m = _CONVERSATION_WITH_RE.search(text)
    if not m:
        return None
    name = m.group("name").strip(" ?.!,'\"").strip()
    if not name or name.lower() in _GENERIC_TARGETS:
        return None
    if len(name) > 40 or len(name.split()) > 4:
        return None
    return m


def extract_conversation_target(message: str) -> str | None:
    """Extract the person name from a 'conversation with <name>' message.

    Returns the original-cased name (for display), or None when the message is
    not a person-specific conversation lookup. Pure function, no I/O.
    """
    if not message:
        return None
    m = _target_match(message.strip())
    if not m:
        return None
    return m.group("name").strip(" ?.!,'\"").strip()


def _find_person_match(text: str) -> str | None:
    """Return the person name from 'find/search <name>' iff it's a real name.

    Rejects captures that contain a platform keyword ("find AI communities",
    "search reels") so those stay PLATFORM_SEARCH, and generic targets
    ("find someone"). Returns the cleaned name, else None.
    """
    m = _FIND_PERSON_RE.search(text)
    if not m:
        return None
    name = m.group("name").strip(" ?.!,'\"-_").strip()
    if not name or name.lower() in _GENERIC_TARGETS:
        return None
    # A capture containing platform vocabulary is a content search, not a person.
    if frozenset(re.findall(r"\b\w+\b", name.lower())) & _PLATFORM_WORDS:
        return None
    return name


def is_user_search(message: str) -> bool:
    """True when the message is a global people-search (list or find mode)."""
    if not message:
        return False
    text = message.lower().strip()
    return bool(_USER_SEARCH_LIST_RE.search(text)) or _find_person_match(text) is not None


def extract_user_search_term(message: str) -> str | None:
    """Return the person name for a 'find <name>' query, or None for list mode.

    None means "list users" (the tool returns recent users); a string narrows the
    search to that name. Pure function, no I/O.
    """
    if not message:
        return None
    return _find_person_match(message.strip())


def route_intent(message: str) -> Intent:
    """Return the best-match Intent for *message*. Pure function, no I/O.

    Resolution order:
      1. Capability Registry — exact, whole-message alias match (authoritative,
         single source of truth). Lazy-imported to avoid an import cycle.
      2. The regex rules below — fallback for the long tail of phrasings the
         registry does not enumerate. Preserves all pre-registry behavior.
    """
    text = message.lower().strip()

    # 1. Registry exact-alias fast path. Whole-message match only, so a phrase
    #    like "suggest friends" maps to its own capability and is never swallowed
    #    by the "friends" capability. New commands register here, not as regexes.
    from app.llm.capabilities import lookup_capability_by_alias  # lazy: breaks cycle

    cap = lookup_capability_by_alias(text)
    if cap is not None:
        return cap.intent

    # 2. Regex fallback ─────────────────────────────────────────────────────────
    # "What did I miss" / "catch me up" — checked FIRST among the regexes.
    # These phrasings carry no platform noun of their own, so they used to fail
    # the platform gate and land on the domain-locked chat prompt (a refusal);
    # and the moment a noun was added ("...in my communities") they matched
    # _LIST_COMMUNITIES_RE instead and merely listed communities. Both are the
    # digest question, so answer it with the digest.
    if _CATCH_ME_UP_RE.search(text):
        return Intent.CATCH_ME_UP

    # Product/how-to questions, answered from the help corpus. Checked after
    # catch-me-up (so "what's new" stays the digest) but before the platform
    # gate, because a question like "how do I create a community?" is exactly
    # the kind that previously fell through to a domain refusal.
    if _PRODUCT_HELP_RE.search(text):
        return Intent.PRODUCT_HELP

    # Explicit recommenders — "suggest/recommend …" phrasing. Checked before the
    # platform gate, routed to the MongoDB recommender tools (never RAG).
    if _SUGGEST_FRIENDS_RE.search(text):
        return Intent.SUGGEST_FRIENDS
    if _SUGGEST_COMMUNITIES_RE.search(text):
        return Intent.SUGGEST_COMMUNITIES
    if _SUGGEST_CREATORS_RE.search(text):
        return Intent.SUGGEST_CREATORS
    # Checked before LIST_CREATORS: "trending creators" must not fall through
    # to the bare "creators" catch-all in _LIST_CREATORS_RE.
    if _TRENDING_CREATORS_RE.search(text):
        return Intent.TRENDING_CREATORS

    # Conversation WITH a named person ("my conversation with divyanshu"). Must
    # be checked before insights — the phrase also contains "my conversation".
    # Routed to the conversation_lookup MongoDB tool, never RAG.
    if _target_match(text):
        return Intent.CONVERSATION_WITH_USER

    # Conversation insights — list DM conversations with activity summaries
    # ("show my chats", "my conversation", "who have I talked to"). Many of these
    # carry no platform keyword, so they are checked before the platform gate.
    if _CONVERSATION_INSIGHTS_RE.search(text):
        return Intent.CONVERSATION_INSIGHTS

    # Social graph listing — actual friends/communities/creators the user already
    # has, NOT recommendations. Checked AFTER conversation intents so "my friends
    # I chatted with" still routes to CONVERSATION_INSIGHTS. The order below is:
    #  1. LIST_FRIENDS before COMMUNITY_LOOKUP (no overlap risk; different words)
    #  2. COMMUNITY_LOOKUP before LIST_COMMUNITIES (more specific phrasing first)
    #  3. LIST_CREATORS before SOCIAL_OVERVIEW (no overlap; different words)
    if _LIST_FRIENDS_RE.search(text):
        return Intent.LIST_FRIENDS
    if _COMMUNITY_LOOKUP_RE.search(text):
        return Intent.COMMUNITY_LOOKUP
    if _LIST_COMMUNITIES_RE.search(text):
        return Intent.LIST_COMMUNITIES
    if _LIST_CREATORS_RE.search(text):
        return Intent.LIST_CREATORS

    # Global people search — "users", "list users", "find <name>". Checked before
    # the platform gate so "users" (a platform word) goes to the MongoDB people
    # tool, not RAG. List mode first, then the "find <name>" person match.
    if _USER_SEARCH_LIST_RE.search(text) or _find_person_match(text):
        return Intent.USER_SEARCH

    # Dashboard / aggregate social stats ("dashboard", "overview", "my activity").
    if _SOCIAL_OVERVIEW_RE.search(text):
        return Intent.SOCIAL_OVERVIEW

    # Current-conversation summary — checked first (these phrases may carry no
    # platform keyword) and routed to the dedicated Mongo summarizer, not RAG.
    if _SUMMARIZE_CONVERSATION_RE.search(text):
        return Intent.SUMMARIZE_CONVERSATION

    # Voice-session recall — checked BEFORE CHANNEL_SUMMARY: "recap the voice
    # session" or "what happened in the design call" would otherwise match
    # CHANNEL_SUMMARY's looser "recap/what happens ... channel" patterns.
    # Requires explicit call/voice/meeting/session framing (see the regex's
    # own comment), so a plain "summarize #general" is unaffected.
    if _VOICE_SESSION_RECALL_RE.search(text):
        return Intent.VOICE_SESSION_RECALL

    # Community / channel summary — checked BEFORE the platform gate because
    # channel names (#notes, #general) often carry no _PLATFORM_WORDS token and
    # would otherwise fall through to CHAT, and BEFORE SUMMARIZE so that
    # "summarize #general" routes here rather than to the unread summarizer.
    # CHANNEL_SUMMARY first (# is a strong unambiguous signal).
    if _CHANNEL_SUMMARY_RE.search(text):
        return Intent.CHANNEL_SUMMARY
    if _COMMUNITY_SUMMARY_RE.search(text):
        return Intent.COMMUNITY_SUMMARY

    # No platform vocabulary → pure general conversation, skip Qdrant entirely
    if not _has_platform_context(text):
        return Intent.CHAT

    # Platform context present — classify by action type
    if _MIXED_RE.search(text):
        return Intent.MIXED
    if _SUMMARIZE_RE.search(text):
        return Intent.SUMMARIZE
    if _RECOMMEND_RE.search(text):
        return Intent.RECOMMENDATION
    if _SEARCH_RE.search(text):
        return Intent.PLATFORM_SEARCH

    return Intent.QA

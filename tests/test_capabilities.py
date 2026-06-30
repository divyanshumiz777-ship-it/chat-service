"""Tests for the Capability Registry, slash commands, help, and smart suggestions.

These cover Part 12 of the Copilot spec and guard Part 13 ("do not break"): every
pre-existing natural-language route must still resolve identically after the
registry was layered on top of the regex router.

Pure functions only — no MongoDB, no Qdrant, no network. Run with `pytest`.
"""

from __future__ import annotations

import pytest

from app.llm.capabilities import (
    CAPABILITIES,
    capabilities_payload,
    lookup_capability_by_alias,
    parse_command,
    render_help_menu,
    suggestions_for,
)
from app.llm.intent_router import Intent, route_intent


# ── Part 2: slash commands ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,intent", [
    ("/friends", Intent.LIST_FRIENDS),
    ("/users", Intent.USER_SEARCH),
    ("/dashboard", Intent.SOCIAL_OVERVIEW),
    ("/overview", Intent.SOCIAL_OVERVIEW),
    ("/messages", Intent.CONVERSATION_INSIGHTS),
    ("/community members", Intent.LIST_COMMUNITIES),
    ("/community owner", Intent.LIST_COMMUNITIES),
    ("/help", Intent.HELP),
    ("/commands", Intent.HELP),
    ("/suggest friends", Intent.SUGGEST_FRIENDS),
])
def test_slash_commands_resolve(text, intent):
    parsed = parse_command(text)
    assert parsed is not None and parsed.intent is intent


def test_slash_search_extracts_args():
    p = parse_command("/search users divyanshu")
    assert p.intent is Intent.USER_SEARCH and p.term == "divyanshu"

    p = parse_command("/search reels ai")
    assert p.intent is Intent.PLATFORM_SEARCH and p.query == "ai" and p.types == ["reel"]

    p = parse_command("/search communities")
    assert p.intent is Intent.PLATFORM_SEARCH and p.types == ["server", "channel", "thread"]


def test_non_slash_and_unknown_slash_return_none():
    assert parse_command("my friends") is None        # not a slash command
    assert parse_command("/totally-unknown-xyz") is None
    assert parse_command("/") is None


# ── Part 2: natural language == slash ──────────────────────────────────────────

@pytest.mark.parametrize("nl,slash", [
    ("my friends", "/friends"),
    ("users", "/users"),
    ("dashboard", "/dashboard"),
    ("my messages", "/messages"),
])
def test_nl_matches_slash(nl, slash):
    assert route_intent(nl) is parse_command(slash).intent


# ── Part 4: help system ────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "help", "commands", "what can you do", "assistant help",
    "show commands", "show menu", "show shortcuts", "@",
])
def test_help_queries_route_to_help(q):
    assert route_intent(q) is Intent.HELP


def test_help_menu_renders_all_categories():
    menu = render_help_menu()
    for token in ("TheSocial Copilot", "Social", "Messages", "Content",
                  "Search", "Recommendations", "Insights"):
        assert token in menu
    assert "👥" in menu and "📊" in menu  # icons present


# ── Part 5: smart suggestions ───────────────────────────────────────────────────

def test_suggestions_present_for_tools_absent_for_chat():
    assert suggestions_for(Intent.LIST_FRIENDS)
    assert suggestions_for(Intent.SOCIAL_OVERVIEW)
    assert suggestions_for(Intent.CHAT) == []


# ── Part 6: user search ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "users", "find users", "search users", "list users",
    "who is on the platform", "find divyanshu",
])
def test_user_search_routes(q):
    assert route_intent(q) is Intent.USER_SEARCH


# ── Capability registry payload (frontend palette source of truth) ──────────────

def test_payload_shape_and_hidden_excluded():
    payload = capabilities_payload()
    assert len(payload) >= 12
    assert all({"name", "label", "icon", "category", "command"} <= set(c) for c in payload)
    # hidden capability (community_lookup) is not surfaced to the palette
    assert not any(c["name"] == "community_lookup" for c in payload)


def test_aliases_are_unique_across_registry():
    seen: dict[str, str] = {}
    for cap in CAPABILITIES:
        for alias in cap.aliases:
            assert alias not in seen, f"duplicate alias {alias!r} ({cap.name} vs {seen[alias]})"
            seen[alias] = cap.name


# ── Part 13: existing NL routes must not regress ────────────────────────────────

@pytest.mark.parametrize("msg,intent", [
    ("community", Intent.LIST_COMMUNITIES),
    ("community members", Intent.LIST_COMMUNITIES),
    ("servers", Intent.LIST_COMMUNITIES),
    ("messages", Intent.CONVERSATION_INSIGHTS),
    ("recent messages", Intent.CONVERSATION_INSIGHTS),
    ("friends", Intent.LIST_FRIENDS),
    ("friend activity", Intent.LIST_FRIENDS),
    ("creator", Intent.LIST_CREATORS),
    ("followed creators", Intent.LIST_CREATORS),
    ("suggest friends", Intent.SUGGEST_FRIENDS),
    ("suggest communities", Intent.SUGGEST_COMMUNITIES),
    ("suggest creators", Intent.SUGGEST_CREATORS),
    ("who should i follow", Intent.SUGGEST_CREATORS),
    ("my conversation with divyanshu", Intent.CONVERSATION_WITH_USER),
    ("my friends i chatted with", Intent.CONVERSATION_INSIGHTS),
    ("find AI communities", Intent.PLATFORM_SEARCH),
    ("find reels about machine learning", Intent.PLATFORM_SEARCH),
    ("what is discord", Intent.CHAT),       # off-topic → CHAT → domain refusal
    ("hello", Intent.CHAT),
])
def test_existing_routes_preserved(msg, intent):
    assert route_intent(msg) is intent

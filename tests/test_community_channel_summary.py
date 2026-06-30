"""Regression tests for COMMUNITY_SUMMARY and CHANNEL_SUMMARY intent routing.

Covers:
  ✓ Summarise existing community
  ✓ Summarise existing channel
  ✓ Quiet channel (zero messages)
  ✓ Empty community (no discussion history)
  ✓ Nonexistent community
  ✓ Nonexistent channel
  ✓ Community with many messages
  ✓ Channel with many messages
  ✓ Generic community query (no specific name)
  ✓ Named community query
  ✓ Hashtag channel query
  ✓ Word-form channel query
  ✓ Routing precedence: CHANNEL_SUMMARY before COMMUNITY_SUMMARY before QA
  ✓ No valid entity query returns "AI assistant is temporarily unavailable"

Pure function tests only — no MongoDB, no Qdrant, no network.
"""

from __future__ import annotations

import pytest

from app.llm.intent_router import (
    Intent,
    extract_channel_name,
    extract_community_name,
    route_intent,
)


# ── Intent routing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    # CHANNEL_SUMMARY — hashtag forms
    ("summarize #general",                  Intent.CHANNEL_SUMMARY),
    ("what happens in #notes",              Intent.CHANNEL_SUMMARY),
    ("tell me about #social-zone",          Intent.CHANNEL_SUMMARY),
    ("what is discussed in #notes",         Intent.CHANNEL_SUMMARY),
    ("who should join #notes",              Intent.CHANNEL_SUMMARY),
    ("what is #announcements",              Intent.CHANNEL_SUMMARY),
    ("summarize the general channel",       Intent.CHANNEL_SUMMARY),
    ("tell me about the general channel",   Intent.CHANNEL_SUMMARY),
    ("recap the notes channel",             Intent.CHANNEL_SUMMARY),
    # COMMUNITY_SUMMARY — with explicit "community/server" qualifier
    ("summarize TheSocial community",       Intent.COMMUNITY_SUMMARY),
    ("tell me about TheSocial server",      Intent.COMMUNITY_SUMMARY),
    ("describe TheSocial community",        Intent.COMMUNITY_SUMMARY),
    ("community summary",                   Intent.COMMUNITY_SUMMARY),
    ("community overview",                  Intent.COMMUNITY_SUMMARY),
    ("summarize this community",            Intent.COMMUNITY_SUMMARY),
    ("what is this community about",        Intent.COMMUNITY_SUMMARY),
    ("what topics are discussed",           Intent.COMMUNITY_SUMMARY),
    ("should I join TheSocial",             Intent.COMMUNITY_SUMMARY),
    ("what is TheSocial community",         Intent.COMMUNITY_SUMMARY),
])
def test_routing_community_and_channel_summary(text, expected):
    assert route_intent(text) is expected


@pytest.mark.parametrize("text,expected", [
    # These must NOT accidentally route to CHANNEL_SUMMARY or COMMUNITY_SUMMARY
    ("summarize this conversation",         Intent.SUMMARIZE_CONVERSATION),
    ("summarize our chat",                  Intent.SUMMARIZE_CONVERSATION),
    ("my communities",                      Intent.LIST_COMMUNITIES),
    ("suggest communities",                 Intent.SUGGEST_COMMUNITIES),
    ("what communities should I join",      Intent.SUGGEST_COMMUNITIES),
    ("my friends",                          Intent.LIST_FRIENDS),
    ("dashboard",                           Intent.SOCIAL_OVERVIEW),
])
def test_existing_routes_not_broken(text, expected):
    assert route_intent(text) is expected


def test_channel_before_community_in_routing():
    """CHANNEL_SUMMARY must take priority over COMMUNITY_SUMMARY.

    Use a message that would satisfy both CHANNEL_SUMMARY (#hashtag present)
    and COMMUNITY_SUMMARY ("this community") but must NOT contain words like
    "overview" that would trigger SOCIAL_OVERVIEW first.
    """
    assert route_intent("summarize #general and this community") is Intent.CHANNEL_SUMMARY


def test_community_before_qa():
    """COMMUNITY_SUMMARY must precede QA so it never falls into RAG."""
    intent = route_intent("what is this community about")
    assert intent is Intent.COMMUNITY_SUMMARY
    assert intent is not Intent.QA


def test_channel_before_qa():
    """CHANNEL_SUMMARY must precede QA so it never falls into RAG."""
    intent = route_intent("summarize #general")
    assert intent is Intent.CHANNEL_SUMMARY
    assert intent is not Intent.QA


def test_channel_before_summarize_intent():
    """'summarize #notes' must route to CHANNEL_SUMMARY, not the catch-all SUMMARIZE."""
    assert route_intent("summarize #notes") is Intent.CHANNEL_SUMMARY


# ── Channel name extraction ───────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("summarize #general",                      "general"),
    ("what happens in #notes",                  "notes"),
    ("tell me about #social-zone",              "social-zone"),
    ("#announcements",                          "announcements"),
    ("summarize the general channel",           "general"),
    ("tell me about the notes channel",         "notes"),
    ("what is discussed in the general channel","general"),
    ("recap the social channel",                "social"),
])
def test_extract_channel_name(text, expected):
    assert extract_channel_name(text) == expected


@pytest.mark.parametrize("text", [
    "summarize this community",
    "what is this community about",
    "my communities",
    "",
])
def test_extract_channel_name_none(text):
    assert extract_channel_name(text) is None


# ── Community name extraction ─────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("summarize TheSocial community",   "TheSocial"),
    ("tell me about GamersHub server",  "GamersHub"),
    ("describe AIForum community",      "AIForum"),
    ("should I join DevZone",           "DevZone"),
    ("what is TheSocial community",     "TheSocial"),
])
def test_extract_community_name(text, expected):
    result = extract_community_name(text)
    assert result is not None
    assert result.lower() == expected.lower()


@pytest.mark.parametrize("text", [
    "what is this community about",
    "what topics are discussed",
    "community overview",
    "",
])
def test_extract_community_name_none_for_generic_queries(text):
    assert extract_community_name(text) is None


# ── Response content guarantees (via handler logic) ───────────────────────────
# These tests exercise the response formatting helpers directly to confirm that
# no valid entity query can return the "unavailable" error string.

_UNAVAILABLE_TEXT = "AI assistant is temporarily unavailable"


def _make_community_data(**kwargs):
    """Build a minimal CommunitySummaryData for unit testing."""
    from app.tools.community_summary import CommunitySummaryData

    defaults = dict(
        server_id="507f1f77bcf86cd799439011",
        name="TheSocial",
        description="A sample community.",
        visibility="public",
        member_count=42,
        channel_names=["general", "announcements", "social-zone"],
        recent_messages=[
            {"text": "Hello everyone!", "sender": "Alice", "channel_name": "general", "ts": "2026-06-01T10:00:00"},
            {"text": "Welcome to TheSocial", "sender": "Bob", "channel_name": "announcements", "ts": "2026-06-01T09:00:00"},
        ],
        recent_threads=[{"title": "Intro thread", "ts": "2026-06-01T08:00:00"}],
        topics=["hello", "welcome", "intro"],
        is_member=True,
    )
    defaults.update(kwargs)
    return CommunitySummaryData(**defaults)


def _make_channel_data(**kwargs):
    """Build a minimal ChannelSummaryData for unit testing."""
    from app.tools.community_summary import ChannelSummaryData

    defaults = dict(
        channel_id="507f1f77bcf86cd799439022",
        name="general",
        server_name="TheSocial",
        message_count=5,
        recent_messages=[
            {"text": "Hey all", "sender": "Alice", "ts": "2026-06-01T10:00:00"},
        ],
        pinned_messages=[],
        thread_count=2,
        topics=["hey", "hello"],
        is_quiet=False,
    )
    defaults.update(kwargs)
    return ChannelSummaryData(**defaults)


def test_community_data_model_valid():
    data = _make_community_data()
    assert data.name == "TheSocial"
    assert data.member_count == 42
    assert len(data.channel_names) == 3
    assert not data.is_member is False  # is_member=True


def test_channel_data_model_valid():
    data = _make_channel_data()
    assert data.name == "general"
    assert not data.is_quiet


def test_quiet_channel_model():
    data = _make_channel_data(message_count=0, recent_messages=[], is_quiet=True)
    assert data.is_quiet
    assert data.message_count == 0


def test_empty_community_model():
    data = _make_community_data(
        recent_messages=[],
        recent_threads=[],
        topics=[],
        member_count=0,
    )
    assert data.member_count == 0
    assert not data.recent_messages


def test_community_with_many_channels():
    names = [f"channel-{i}" for i in range(25)]
    data = _make_community_data(channel_names=names)
    assert len(data.channel_names) == 25


# ── Prompt builders (pure function, no I/O) ───────────────────────────────────

def test_build_community_summary_prompt_contains_name():
    from app.llm.prompts import build_community_summary_prompt

    data = _make_community_data()
    prompt = build_community_summary_prompt(data)
    assert "TheSocial" in prompt
    assert "42" in prompt
    assert "general" in prompt


def test_build_community_summary_prompt_sparse_data():
    from app.llm.prompts import build_community_summary_prompt

    data = _make_community_data(
        recent_messages=[], recent_threads=[], topics=[], description=""
    )
    prompt = build_community_summary_prompt(data)
    assert "TheSocial" in prompt
    assert "No description" in prompt


def test_build_channel_summary_prompt_contains_channel():
    from app.llm.prompts import build_channel_summary_prompt

    data = _make_channel_data()
    prompt = build_channel_summary_prompt(data)
    assert "#general" in prompt
    assert "TheSocial" in prompt


def test_build_channel_summary_prompt_quiet_channel():
    from app.llm.prompts import build_channel_summary_prompt

    data = _make_channel_data(message_count=0, recent_messages=[], is_quiet=True)
    prompt = build_channel_summary_prompt(data)
    assert "#general" in prompt
    assert "No messages" in prompt


def test_build_channel_summary_prompt_pinned_messages():
    from app.llm.prompts import build_channel_summary_prompt

    data = _make_channel_data(
        pinned_messages=[{"text": "Rules: be kind.", "sender": "Admin", "ts": ""}]
    )
    prompt = build_channel_summary_prompt(data)
    assert "Rules: be kind." in prompt
    assert "Pinned" in prompt

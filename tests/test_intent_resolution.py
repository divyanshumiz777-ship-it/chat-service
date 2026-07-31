"""Routing guarantees for the phrasings the UI actually sends.

These exist because the assistant shipped with three of its five "Get started"
buttons wired to prose that missed its own tools — "Summarize unread" listed
communities, "Search people" produced a generic chat reply — and two intents
(RECOMMENDATION, SUMMARIZE) had no handler at all, so they fell through to
unscoped vector search. Nothing in the suite caught it, because nothing
asserted what a given sentence routes to.

Any string here that a UI surface sends must keep resolving to a real handler.
"""

from __future__ import annotations

import pytest

from app.api.internal import (
    _MESSAGE_TOOL_HANDLERS,
    _RECOMMENDERS,
    _SIMPLE_TOOL_HANDLERS,
    _resolve_umbrella_intent,
)
from app.llm.capabilities import parse_command
from app.llm.intent_router import Intent, route_intent


def resolve(message: str) -> Intent:
    """Exactly what handle_chat does to pick a handler."""
    parsed = parse_command(message)
    intent = parsed.intent if parsed is not None else route_intent(message)
    return _resolve_umbrella_intent(intent, message)


# Intents that reach a concrete tool rather than the generic RAG fallthrough.
HANDLED = (
    set(_RECOMMENDERS)
    | set(_SIMPLE_TOOL_HANDLERS)
    | set(_MESSAGE_TOOL_HANDLERS)
    | {
        Intent.USER_SEARCH,
        Intent.CONVERSATION_WITH_USER,
        Intent.SUMMARIZE_CONVERSATION,
        Intent.CATCH_ME_UP,
        Intent.HELP,
    }
)


class TestUmbrellaIntentsAreResolved:
    """RECOMMENDATION and SUMMARIZE must never reach dispatch unresolved."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("discover communities", Intent.SUGGEST_COMMUNITIES),
            ("explore servers to join", Intent.SUGGEST_COMMUNITIES),
            ("recommend some creators", Intent.SUGGEST_CREATORS),
            ("suggest reels for me to watch", Intent.SUGGEST_CREATORS),
            # NOTE: "people to follow" deliberately lands on creators, not
            # friends — following is the creator mechanic in this product,
            # while friendship is a separate mutual relationship.
            ("discover people to follow", Intent.SUGGEST_CREATORS),
        ],
    )
    def test_recommendation_resolves_by_object_noun(self, message, expected):
        assert resolve(message) is expected

    @pytest.mark.parametrize(
        "noun,expected",
        [
            ("communities", Intent.SUGGEST_COMMUNITIES),
            ("friends", Intent.SUGGEST_FRIENDS),
            ("creators", Intent.SUGGEST_CREATORS),
            ("reels", Intent.SUGGEST_CREATORS),
        ],
    )
    def test_resolver_maps_each_object_noun(self, noun, expected):
        """The resolver itself, isolated from whatever the router did first."""
        assert _resolve_umbrella_intent(Intent.RECOMMENDATION, f"discover {noun}") is expected

    def test_bare_recommendation_falls_back_to_overview_not_rag(self):
        assert _resolve_umbrella_intent(Intent.RECOMMENDATION, "give me suggestions") is (
            Intent.SOCIAL_OVERVIEW
        )

    def test_summarize_becomes_the_digest(self):
        assert _resolve_umbrella_intent(Intent.SUMMARIZE, "what did I miss") is (
            Intent.CATCH_ME_UP
        )

    @pytest.mark.parametrize(
        "message",
        [
            "what did I miss",
            "catch me up",
            "what did i miss in my communities",
            "summarize unread",
            "anything new?",
            "what happened while I was away",
        ],
    )
    def test_away_phrasings_reach_the_digest(self, message):
        """Live probing found two failure modes here: bare "catch me up" carried
        no platform noun and fell to the domain-locked CHAT refusal, while
        "what did I miss in my communities" matched the list-communities rule
        and merely listed them."""
        assert resolve(message) is Intent.CATCH_ME_UP

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("my communities", Intent.LIST_COMMUNITIES),
            ("summarize this conversation", Intent.SUMMARIZE_CONVERSATION),
            ("suggest communities", Intent.SUGGEST_COMMUNITIES),
        ],
    )
    def test_catch_me_up_rule_does_not_swallow_neighbours(self, message, expected):
        assert resolve(message) is expected

    def test_resolver_is_identity_for_concrete_intents(self):
        for intent in (Intent.LIST_FRIENDS, Intent.HELP, Intent.CHAT, Intent.PLATFORM_SEARCH):
            assert _resolve_umbrella_intent(intent, "anything") is intent

    def test_no_umbrella_intent_survives_resolution(self):
        for message in ("discover communities", "suggest creators", "what did I miss"):
            assert resolve(message) not in (Intent.RECOMMENDATION, Intent.SUMMARIZE)


class TestSlashCommandsFromTheUI:
    """The UI now sends slash commands so a button cannot silently mis-route."""

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("/suggest communities", Intent.SUGGEST_COMMUNITIES),
            ("/suggest creators", Intent.SUGGEST_CREATORS),
            ("/suggest friends", Intent.SUGGEST_FRIENDS),
            ("/trending creators", Intent.TRENDING_CREATORS),
            ("/dashboard", Intent.SOCIAL_OVERVIEW),
            ("/my friends", Intent.LIST_FRIENDS),
            ("/my communities", Intent.LIST_COMMUNITIES),
            ("/my messages", Intent.CONVERSATION_INSIGHTS),
            ("/help", Intent.HELP),
        ],
    )
    def test_slash_command_reaches_its_tool(self, command, expected):
        assert resolve(command) is expected

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("/suggest communities", Intent.SUGGEST_COMMUNITIES),
            ("/dashboard", Intent.SOCIAL_OVERVIEW),
            ("/my friends", Intent.LIST_FRIENDS),
        ],
    )
    def test_every_ui_command_has_a_handler(self, command, expected):
        assert resolve(command) in HANDLED


class TestPlatformVocabulary:
    """Words the UI puts in front of users must count as platform context.

    Without these, route_intent bails to Intent.CHAT — the domain-locked
    "greetings only" path — and the user gets a refusal for a question about
    the product they are standing in.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "Find people on this platform with interesting profiles",
            "show me people to follow",
            "who are the users on this platform",
        ],
    )
    def test_people_phrasings_are_not_treated_as_small_talk(self, message):
        assert route_intent(message) is not Intent.CHAT

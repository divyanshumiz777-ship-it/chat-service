"""Product ("how does TheSocial work?") questions must reach the help corpus.

The assistant previously could not explain the product it lives inside: the
chat prompt forbids explaining anything, so "how do I create a community?" got
the same canned refusal as a question about the weather — while the UI listed
"Explain the platform" as a capability.

Two properties matter and pull against each other:
  1. product questions reach PRODUCT_HELP, and
  2. outside-world questions still hit the domain refusal.
A regex that satisfies only the first is worse than none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.intent_router import Intent, route_intent
from app.rag import chunker
from app.rag.ingest_sync import _HELP_DIR


class TestProductQuestionsRoute:
    @pytest.mark.parametrize(
        "message",
        [
            "how do I create a community",
            "How do I create a community?",
            "how can I turn on two factor authentication",
            "what is a thread",
            "whats a reel",
            "what is the difference between friends and follows",
            "where do I find my saved items",
            "explain this platform",
            "how does this platform work",
            "how to join a community",
        ],
    )
    def test_reaches_product_help(self, message):
        assert route_intent(message) is Intent.PRODUCT_HELP


class TestOutsideWorldStillRefused:
    """The domain lock must survive the new rule."""

    @pytest.mark.parametrize(
        "message",
        [
            "what is the weather",
            "what is the capital of France",
            "tell me a joke",
            "write me a poem about the ocean",
            "hello",
        ],
    )
    def test_stays_on_the_chat_refusal_path(self, message):
        assert route_intent(message) is Intent.CHAT


class TestDoesNotSwallowDataQuestions:
    """Questions about the user's OWN data must keep reaching their tools —
    those answers come from MongoDB, not from documentation."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("what did I miss", Intent.CATCH_ME_UP),
            ("my communities", Intent.LIST_COMMUNITIES),
            ("my friends", Intent.LIST_FRIENDS),
            ("suggest communities", Intent.SUGGEST_COMMUNITIES),
            ("discover communities", Intent.RECOMMENDATION),
            ("summarize this conversation", Intent.SUMMARIZE_CONVERSATION),
        ],
    )
    def test_data_questions_are_untouched(self, message, expected):
        assert route_intent(message) is expected

    def test_imperative_does_it_but_how_to_explains_it(self):
        """A deliberate split: "discover communities" runs the recommender,
        "how do I discover communities?" explains where Discover lives."""
        assert route_intent("discover communities") is Intent.RECOMMENDATION
        assert route_intent("How do I discover communities?") is Intent.PRODUCT_HELP


class TestHelpCorpus:
    def test_corpus_exists_and_is_non_trivial(self):
        topics = sorted(_HELP_DIR.glob("*.md"))
        assert len(topics) >= 10, "help corpus is suspiciously small"

    def test_every_topic_has_a_title_and_body(self):
        for path in sorted(_HELP_DIR.glob("*.md")):
            raw = path.read_text(encoding="utf-8").strip()
            assert raw.startswith("# "), f"{path.name} has no '# Title' heading"
            body = "\n".join(raw.splitlines()[1:]).strip()
            assert len(body) > 100, f"{path.name} has almost no content"

    def test_every_topic_chunks_as_public_ownerless_docs(self):
        """Help is identical for every user, so it must never carry ACL or
        attribution that would let it land in the private collection."""
        for path in sorted(_HELP_DIR.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            lines = raw.strip().splitlines()
            chunks = chunker.normalize_doc(
                path.stem, lines[0][2:].strip(), "\n".join(lines[1:])
            )
            assert chunks, f"{path.name} produced no chunks"
            for c in chunks:
                assert c.payload.source_type == "docs"
                assert c.payload.visibility == "public"
                assert c.payload.allowed_user_ids == []
                assert not c.payload.owner_id

    def test_docs_is_a_searchable_source_type(self):
        from app.rag.retriever import _resolve_source_types

        assert _resolve_source_types(["docs"]) == {"docs"}
        assert _resolve_source_types(["help"]) == {"docs"}

    def test_empty_topic_produces_no_chunks(self):
        assert chunker.normalize_doc("blank", "", "") == []


class TestSplitTextTerminates:
    """_split_text could loop forever, which a real help topic triggered.

    `start` advances by `end - _OVERLAP_CHARS`. When the chosen sentence
    boundary lands within the overlap window of `start`, that expression is
    <= start, so the loop restarts at or before the same position and appends
    the same chunk until the process dies with MemoryError. Any ingest
    containing such a passage would hang the service.
    """

    def test_separator_near_window_start_still_terminates(self):
        from app.rag.chunker import _MAX_CHARS, _split_text

        evil = "A." + ("x" * (_MAX_CHARS * 3))
        chunks = _split_text(evil)
        assert 0 < len(chunks) < 50

    def test_no_separators_at_all_terminates(self):
        from app.rag.chunker import _MAX_CHARS, _split_text

        assert 0 < len(_split_text("y" * (_MAX_CHARS * 4))) < 50

    def test_ordinary_prose_is_unaffected(self):
        from app.rag.chunker import _MAX_CHARS, _split_text

        text = "Hello world. " * 200
        chunks = _split_text(text)
        assert chunks
        assert all(len(c) <= _MAX_CHARS for c in chunks)

    def test_short_text_is_a_single_chunk(self):
        from app.rag.chunker import _split_text

        assert _split_text("Just one line.") == ["Just one line."]


class TestRecommendationsAreClickable:
    """A recommendation with no link is a dead end.

    Every recommender already returned an id (community_id / creator_id /
    user_id); the reply template simply never used it, so the assistant would
    name a community and give the user no way to open it.
    """

    def test_community_links_to_the_community_route(self):
        from app.api.internal import _recommendation_href

        assert _recommendation_href({"community_id": "abc"}) == "/community/abc"

    def test_people_link_to_the_profile_route(self):
        from app.api.internal import _recommendation_href

        assert _recommendation_href({"creator_id": "c1"}) == "/profile/c1"
        assert _recommendation_href({"user_id": "u1"}) == "/profile/u1"

    def test_missing_id_yields_no_link(self):
        from app.api.internal import _recommendation_href

        assert _recommendation_href({"name": "x"}) is None

    def test_rendered_output_is_a_markdown_link(self):
        from app.api.internal import _format_recommendations

        out = _format_recommendations(
            "Suggested:", [{"community_id": "s1", "name": "Design", "reasons": ["active"]}]
        )
        assert "[Design](/community/s1)" in out

    def test_unlinkable_item_still_renders_its_name(self):
        from app.api.internal import _format_recommendations

        out = _format_recommendations("Suggested:", [{"name": "Orphan", "reasons": []}])
        assert "Orphan" in out
        assert "](" not in out

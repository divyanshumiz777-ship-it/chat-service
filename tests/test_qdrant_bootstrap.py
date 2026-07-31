"""Qdrant payload indexes must exist for every filtered key.

Found live, and it had disabled semantic search entirely: the collections were
created with vector params but no payload indexes, so Qdrant rejected every
filtered search with

    400 Bad request: Index required but not found for "visibility"

_query_collection catches that and returns [], so search_all quietly produced
zero results for every user, every query, forever — which in turn made the
domain-locked prompt answer real questions with "I couldn't find any matching
data on TheSocial."

Two properties are load-bearing:
  1. an index is created for every key _build_acl_filter filters on, and
  2. indexes are ensured for collections that ALREADY EXIST — the original
     code hit `continue` on those, which is precisely why the deployed
     collections never got them.
"""

from __future__ import annotations

import pytest

from app.rag import qdrant_client as qc


class FakeClient:
    def __init__(self, existing: set[str]):
        self._existing = existing
        self.created_collections: list[str] = []
        self.created_indexes: list[tuple[str, str]] = []

    async def collection_exists(self, name):
        return name in self._existing

    async def create_collection(self, collection_name, vectors_config):
        self.created_collections.append(collection_name)

    async def create_payload_index(self, collection_name, field_name, **_kwargs):
        self.created_indexes.append((collection_name, field_name))


@pytest.fixture
def fake(monkeypatch):
    def _install(existing=frozenset()):
        client = FakeClient(set(existing))
        monkeypatch.setattr(qc, "get_client", lambda: client)
        return client

    return _install


@pytest.mark.asyncio
class TestBootstrapPayloadIndexes:
    async def test_indexes_every_filtered_key(self, fake):
        client = fake()
        await qc.bootstrap_collections()

        for collection in (
            qc.settings.qdrant_collection_public,
            qc.settings.qdrant_collection_private,
        ):
            for key in ("visibility", "allowed_user_ids", "source_type"):
                assert (collection, key) in client.created_indexes

    async def test_backfills_indexes_on_preexisting_collections(self, fake):
        """The regression that broke production: collections already existed,
        so the loop short-circuited before ever creating an index."""
        both = {
            qc.settings.qdrant_collection_public,
            qc.settings.qdrant_collection_private,
        }
        client = fake(existing=both)
        await qc.bootstrap_collections()

        assert client.created_collections == []  # nothing re-created
        assert client.created_indexes  # but indexes still ensured
        assert len(client.created_indexes) == len(both) * len(qc._FILTERED_PAYLOAD_KEYS)

    async def test_index_failure_does_not_break_startup(self, fake, monkeypatch):
        """A service that can't index is still useful for every non-RAG tool."""
        client = fake()

        async def boom(**_kwargs):
            raise RuntimeError("qdrant unavailable")

        monkeypatch.setattr(client, "create_payload_index", boom)
        await qc.bootstrap_collections()  # must not raise

    async def test_filtered_keys_match_the_acl_filter(self):
        """Guard against the filter growing a key nobody indexed."""
        import inspect

        from app.rag import retriever

        src = inspect.getsource(retriever._build_acl_filter)
        for key in qc._FILTERED_PAYLOAD_KEYS:
            assert f'key="{key}"' in src, f"{key} indexed but no longer filtered"
        for key in ("visibility", "allowed_user_ids", "source_type"):
            assert key in qc._FILTERED_PAYLOAD_KEYS, f"{key} filtered but not indexed"

"""ingest_all(rebuild=True) re-embeds everything, ignoring stale rag_index_state."""

import asyncio

import app.rag.ingest_sync as ingest_sync


def test_rebuild_ignores_recorded_hashes_and_resets_afterwards(monkeypatch):
    async def recorded(source_type, ids):
        return {(i, 0): "hash" for i in ids}  # "already embedded" per the old cluster

    monkeypatch.setattr(ingest_sync, "get_existing_hashes", recorded)

    seen = {}

    async def fake_source(since=None):
        seen["during"] = await ingest_sync._existing_hashes("server", ["a", "b"])
        return ingest_sync.IngestResult(source_type="server")

    sources = [
        "ingest_servers", "ingest_channels", "ingest_threads", "ingest_reels",
        "ingest_users", "ingest_messages", "ingest_notifications", "ingest_voice_sessions",
    ]
    for name in sources:
        monkeypatch.setattr(ingest_sync, name, fake_source)

    asyncio.run(ingest_sync.ingest_all(rebuild=True))
    assert seen["during"] == {}  # nothing skipped while rebuilding

    after = asyncio.run(ingest_sync._existing_hashes("server", ["a"]))
    assert after == {("a", 0): "hash"}  # normal incremental behaviour restored

    asyncio.run(ingest_sync.ingest_all())
    assert seen["during"] == {("a", 0): "hash", ("b", 0): "hash"}

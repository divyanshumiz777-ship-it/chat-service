"""Cold start: the service must answer while its warm-up is still running."""

import asyncio
import time

from fastapi.testclient import TestClient

import app.main as main


def test_answers_immediately_while_warm_up_is_still_running(monkeypatch):
    calls = []

    async def slow_bootstrap():
        calls.append("bootstrap")
        await asyncio.sleep(30)  # a slow Qdrant during a cold start

    monkeypatch.setattr(main.settings, "startup_warmup", True)
    monkeypatch.setattr(main, "bootstrap_collections", slow_bootstrap)

    started = time.monotonic()
    with TestClient(main.app) as client:
        res = client.get("/health")
        elapsed = time.monotonic() - started
    assert res.status_code == 200
    assert elapsed < 5, f"startup blocked for {elapsed:.1f}s"
    assert calls == ["bootstrap"]  # the warm-up did start, in the background


def test_warm_up_steps_never_raise(monkeypatch):
    async def boom():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "bootstrap_collections", boom)

    import app.rag.embedder as embedder
    import app.rag.ingest_sync as ingest_sync

    monkeypatch.setattr(embedder, "get_model", lambda: (_ for _ in ()).throw(RuntimeError("no model")))

    async def no_docs():
        raise RuntimeError("no docs")

    monkeypatch.setattr(ingest_sync, "ingest_docs", no_docs)
    asyncio.run(main._warm_up())  # logs, doesn't raise


def test_model_loads_once_under_concurrency(monkeypatch):
    import threading

    import app.rag.embedder as embedder

    loads = []

    class FakeModel:
        def __init__(self, name):
            loads.append(name)
            time.sleep(0.2)  # slow load, so the threads overlap

    import sys
    import types

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    embedder.reset_model()
    try:
        threads = [threading.Thread(target=embedder.get_model) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(loads) == 1
    finally:
        embedder.reset_model()

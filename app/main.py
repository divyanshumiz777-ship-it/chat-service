"""FastAPI application entrypoint for the AI Chat Service.

Batch 14A is a runnable skeleton: health probes, the internal-token security
layer, and a metadata root. RAG (Qdrant + SentenceTransformers), Gemini,
ingestion, and the gateway/search endpoints are introduced in later batches.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.internal import router as internal_router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(settings.service_name)


from app.rag.qdrant_client import bootstrap_collections


async def _warm_up() -> None:
    """Everything the service used to do BEFORE accepting requests.

    It now runs in the background right after the server starts listening, so
    a cold start (Render free tier) is answering within seconds instead of
    holding every request until setup finished. Each step is independent and
    never raises: a request arriving mid-warm-up just does that work itself
    (the embedding model loads once — see embedder.get_model's lock).
    """
    try:
        await bootstrap_collections()
        logger.info("Qdrant collections initialized")
    except Exception as exc:
        logger.error("Failed to initialize Qdrant: %s", exc)

    # Load the embedding model now, off the event loop, so the first real
    # search doesn't pay for it (torch import + model load is the slow part
    # of a cold start).
    try:
        from app.rag.embedder import get_model

        await asyncio.to_thread(get_model)
        logger.info("embedding model ready")
    except Exception as exc:
        logger.error("Failed to load the embedding model: %s", exc)

    # Product help is a small, static, version-controlled corpus, so index it
    # on every start rather than relying on someone triggering an ingest.
    # Embeddings are content-hash cached, so repeat startups are nearly free.
    try:
        from app.rag.ingest_sync import ingest_docs

        result = await ingest_docs()
        logger.info(
            "product help ready: %d topic(s), %d chunk(s)", result.docs, result.chunks
        )
    except Exception as exc:
        # Never fatal: every non-help capability still works without it.
        logger.error("Failed to index product help: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "starting %s (env=%s)",
        settings.service_name,
        settings.environment,
    )
    warm_up = asyncio.create_task(_warm_up()) if settings.startup_warmup else None
    yield
    if warm_up is not None and not warm_up.done():
        warm_up.cancel()
    logger.info("shutting down %s", settings.service_name)

app = FastAPI(
    title="The Social — AI Chat Service",
    version="0.1.0",
    description=(
        "RAG assistant for The Social. Internal service — reachable only via the "
        "Hono API gateway, never directly from the frontend."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.include_router(health_router)
app.include_router(internal_router)


@app.get("/", tags=["meta"])
async def root() -> dict:
    """Service metadata."""
    return {
        "service": settings.service_name,
        "version": app.version,
        "status": "ok",
        "docs": app.docs_url,
    }

"""FastAPI application entrypoint for the AI Chat Service.

Batch 14A is a runnable skeleton: health probes, the internal-token security
layer, and a metadata root. RAG (Qdrant + SentenceTransformers), Gemini,
ingestion, and the gateway/search endpoints are introduced in later batches.
"""

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

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "starting %s (env=%s)",
        settings.service_name,
        settings.environment,
    )
    try:
        await bootstrap_collections()
        logger.info("Qdrant collections initialized")
    except Exception as exc:
        logger.error("Failed to initialize Qdrant: %s", exc)

    # Product help is a small, static, version-controlled corpus, so index it on
    # every start rather than relying on someone triggering an ingest. Without
    # this, "how do I create a community?" depends on an operational step that
    # nothing enforces — and the whole reason product help exists is that the
    # assistant used to refuse those questions outright. Embeddings are
    # content-hash cached, so repeat startups are nearly free.
    try:
        from app.rag.ingest_sync import ingest_docs

        result = await ingest_docs()
        logger.info(
            "product help ready: %d topic(s), %d chunk(s)", result.docs, result.chunks
        )
    except Exception as exc:
        # Never block startup: every non-help capability still works without it.
        logger.error("Failed to index product help: %s", exc)

    yield
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

"""Health and readiness probes.

`/health`        — liveness: the process is up and serving HTTP (no I/O).
`/health/ready`  — readiness: pings MongoDB and Redis (both required) and reports
                   Qdrant status (optional).

Readiness returns HTTP 503 only when a REQUIRED dependency (Mongo or Redis) is
unreachable. Qdrant is optional for now: its status is reported as
`healthy | unhealthy | not_wired` but never affects readiness. The container
liveness healthcheck uses `/health`, so it is unaffected by dependency outages.
"""

from fastapi import APIRouter, Response, status

from app.config import get_settings
from app.core import mongo
from app.core import redis as redis_client
from app.rag import qdrant_client

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health() -> dict:
    """Liveness probe — process is up and serving."""
    return {"status": "ok", "service": settings.service_name}


@router.get("/health/ready")
async def readiness(response: Response) -> dict:
    """Readiness probe — Mongo + Redis required; Qdrant optional."""
    mongo_ok = await mongo.check_health()
    redis_ok = await redis_client.check_health()
    qdrant_status = await qdrant_client.get_qdrant_status()

    ready = mongo_ok and redis_ok  # Qdrant does not gate readiness yet
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if ready else "not_ready",
        "service": settings.service_name,
        "checks": {
            "app": "ok",
            "mongo": "ok" if mongo_ok else "down",
            "redis": "ok" if redis_ok else "down",
            "qdrant": qdrant_status,  # healthy | unhealthy | not_wired
        },
    }

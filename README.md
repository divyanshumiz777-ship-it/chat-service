# AI Chat Service

RAG assistant for **The Social** — searches and answers questions about the
platform (communities, creators, users, channels, messages, notifications).
Built with FastAPI.

> **Internal service.** Reachable only through the Hono API gateway on the
> internal network (port `8002`, not published) — never directly from the
> frontend. All access is gated by the `X-Internal-Token` guard.

## Status

**Batch 14A — skeleton.** Runnable FastAPI app with health probes, typed config,
the internal-token security layer, and the folder structure. **No Gemini, no
Qdrant, no ingestion, no gateway integration yet** — those arrive in later
batches.

## Endpoints

| Method | Path                  | Auth             | Purpose                       |
|--------|-----------------------|------------------|-------------------------------|
| GET    | `/`                   | none             | Service metadata              |
| GET    | `/health`             | none             | Liveness probe                |
| GET    | `/health/ready`       | none             | Readiness (deps stubbed)      |
| GET    | `/internal/v1/ping`   | X-Internal-Token | Authenticated reachability    |
| GET    | `/docs`               | none             | Swagger UI (non-production)   |

## Run locally

```bash
cd services/chat-service
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8002
# → http://127.0.0.1:8002/health
```

## Layout

```
app/
├── main.py          # FastAPI app + lifespan
├── config.py        # pydantic-settings
├── core/
│   └── security.py  # X-Internal-Token guard
└── api/
    ├── health.py    # liveness + readiness
    └── internal.py  # gateway-only endpoints (guarded)
```

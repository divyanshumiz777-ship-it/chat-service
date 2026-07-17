"""Groq chat-completion client (OpenAI-compatible) using the official `groq` SDK.

Lazy singleton — no API call at import or startup. The `groq` package is imported
inside get_client() so this module loads without the package installed (health
checks stay lightweight). Uses the native async client (AsyncGroq) so the FastAPI
event loop is never blocked.

Provider implementation detail: callers go through app.llm.provider, which keeps
the rest of the application unaware of the concrete LLM backend.

NO tool calling. NO conversation memory. NO assistant endpoints. NO summarization.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.groq")


# ── Lazy singleton ─────────────────────────────────────────────────────────────

_client: Any | None = None


def get_client() -> Any:
    """Return the cached AsyncGroq client, initialising on first call.

    Timeout and retry budgets are baked into the client so every request inherits
    them. Raises RuntimeError if GROQ_API_KEY is unset, ImportError if the `groq`
    package is missing — both are handled gracefully by the provider facade.
    """
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured — set it in .env or the environment"
            )
        from groq import AsyncGroq

        _client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=settings.groq_timeout_seconds,
            max_retries=settings.groq_max_retries,
        )
        logger.info(
            "AsyncGroq client initialised (models: %s)",
            ", ".join(settings.groq_model_chain),
        )
    return _client


def reset_client() -> None:
    """Drop the cached client (frees reference; reloads on next call)."""
    global _client
    _client = None


# ── Generation ───────────────────────────────────────────────────────────────

async def complete(system_prompt: str, user_prompt: str) -> str:
    """Run a chat completion, walking the configured model fallback chain.

    Tries the primary model first, then each fallback in order. The SDK applies
    `groq_max_retries` for transient errors within a single model; model fallback
    handles a model being unavailable/decommissioned. Raises the last error only
    if every model in the chain fails — the provider facade maps that to a safe
    user-facing message (this function never returns a stack trace).
    """
    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for model in settings.groq_model_chain:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_output_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            logger.warning("groq model %s returned empty content; trying next", model)
        except Exception as exc:  # noqa: BLE001 — try the next model in the chain
            last_error = exc
            logger.warning("groq model %s failed: %s; trying next", model, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError("no Groq model produced a response")


# ── Audio transcription ────────────────────────────────────────────────────────

async def transcribe(audio_bytes: bytes, filename: str, language: str = "en") -> str:
    """Transcribe a short audio clip via Groq's Whisper endpoint.

    A distinct Groq API surface from chat completions (`audio.transcriptions`,
    not `chat.completions`) — no model-fallback chain here, just the one
    configured Whisper model. Raises on failure; the caller (the internal
    voice/transcribe-chunk route) maps that to a clean HTTP error rather than
    ever surfacing a raw exception.
    """
    client = get_client()
    response = await client.audio.transcriptions.create(
        model=settings.groq_whisper_model,
        file=(filename, audio_bytes),
        language=language,
    )
    return (response.text or "").strip()

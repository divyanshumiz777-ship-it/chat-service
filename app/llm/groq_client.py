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

# Whisper's decoder falls back on these boilerplate video-sign-off phrases
# (learned from its training data, which is heavy on YouTube/subtitle
# content) when fed silence or near-silence — and it does so CONFIDENTLY:
# direct testing with pure digital silence produced "Thank you." with
# no_speech_prob=0 and a perfectly normal avg_logprob, so the usual
# confidence-based heuristics for detecting hallucination do NOT catch this
# case. Filtering by exact match against known filler phrases is the
# pragmatic mitigation most production Whisper-wrapper tools use for exactly
# this reason — there's no clean confidence signal to threshold on instead.
_SILENCE_HALLUCINATIONS = frozenset({
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching!",
    "thank you for watching!",
    "bye",
    "bye-bye",
    "bye bye",
    "see you next time",
    "see you in the next video",
    "you",
    "",
})


def _looks_like_silence_hallucination(text: str) -> bool:
    return text.strip().lower().rstrip(".!?") in _SILENCE_HALLUCINATIONS


# Whisper's `prompt` field primes the decoder's initial context — giving it
# a Hinglish-style example (Hindi words spelled in Roman script) nudges it to
# CONTINUE transliterating that way rather than switching to Devanagari or
# translating outright, on top of `language="en"` below (a documented
# community trick: forcing the English-token decoder while it's actually
# hearing Hindi phonemes produces a phonetic Roman-script approximation of
# the speech instead of a literal English rendering or native-script output).
HINGLISH_PRIMING_PROMPT = (
    "Yaar, aaj ka din bohot accha tha, sab kuch theek chal raha hai."
)


async def transcribe(
    audio_bytes: bytes,
    filename: str,
    language: str = "en",
    prompt: str | None = None,
) -> str:
    """Transcribe a short audio clip via Groq's Whisper endpoint. Returns an
    EMPTY STRING — never None — for both "no speech at all" and "this looks
    like a known silence hallucination": both are legitimate, SUCCESSFUL
    outcomes (a 4-second window with nobody talking is completely normal and
    will happen constantly), not errors. Raises on a genuine failure
    (network/API error) so the caller can tell "processed fine, nothing to
    say" apart from "the call itself failed" — conflating the two would make
    ordinary silence between sentences look identical to the transcription
    service being down.

    A distinct Groq API surface from chat completions (`audio.transcriptions`,
    not `chat.completions`) — no model-fallback chain here, just the one
    configured Whisper model.
    """
    client = get_client()
    response = await client.audio.transcriptions.create(
        model=settings.groq_whisper_model,
        file=(filename, audio_bytes),
        language=language,
        prompt=prompt,
        response_format="verbose_json",
    )
    text = (response.text or "").strip()
    if not text or _looks_like_silence_hallucination(text):
        return ""
    return text

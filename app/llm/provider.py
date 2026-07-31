"""Provider-agnostic LLM facade.

This is the ONLY generation entry point the rest of the application imports. It
hides the concrete backend (currently Groq) behind a stable contract so call
sites never change when the provider does. Selection is driven by
`settings.llm_provider`.

Both public functions NEVER raise: any backend failure degrades gracefully to a
safe, user-facing message. Stack traces are logged internally, never returned.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.llm.prompts import (
    CATCH_ME_UP_SYSTEM_PROMPT,
    CHANNEL_SUMMARY_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    COMMUNITY_SUMMARY_SYSTEM_PROMPT,
    MODERATION_TRIAGE_SYSTEM_PROMPT,
    NO_DATA_FALLBACK,
    SUMMARY_SYSTEM_PROMPT,
    PRODUCT_HELP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VOICE_SESSION_SUMMARY_SYSTEM_PROMPT,
    build_catch_me_up_prompt,
    build_channel_summary_prompt,
    build_community_summary_prompt,
    build_moderation_triage_prompt,
    build_docs_prompt,
    build_prompt,
    build_summary_prompt,
    build_voice_session_summary_prompt,
)

if TYPE_CHECKING:
    from app.rag.retriever import SearchResult
    from app.tools.catch_me_up import CatchMeUpData
    from app.tools.community_summary import ChannelSummaryData, CommunitySummaryData
    from app.tools.conversation_summary import ConversationSummaryData

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.provider")

# Safe, user-facing fallback messages (kept stable across provider swaps so the
# frontend/UX sees identical degradation behaviour). Empty-context and error
# paths reuse the domain-aware NO_DATA_FALLBACK so the assistant always redirects
# to TheSocial capabilities instead of dead-ending.
_UNAVAILABLE = "The AI assistant is not available right now. Please try again later."
_GENERIC_ERROR = "I encountered an error while generating a response. Please try again."
_EMPTY = NO_DATA_FALLBACK


async def generate_product_help(
    user_id: str,
    query: str,
    results: list[SearchResult],  # type: ignore[name-defined]
) -> str:
    """Answer a "how does TheSocial work?" question from the help corpus.

    Same retrieval plumbing as generate_answer, but a different system prompt
    and a different empty-state: falling back to NO_DATA_FALLBACK here would
    tell a user asking a perfectly reasonable product question that there is
    "no matching data", which reads as a bug rather than a documentation gap.
    """
    if not results:
        return (
            "I don't have documentation on that yet.\n\n"
            "I can explain communities, channels, direct messages, reels, "
            "friends and follows, voice channels, tips, privacy, notifications, "
            "and account security."
        )

    prompt = build_docs_prompt(query=query, results=results)

    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=PRODUCT_HELP_SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 - graceful degradation boundary
        logger.error("generate_product_help failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_answer(
    user_id: str,
    query: str,
    results: list[SearchResult],  # type: ignore[name-defined]
) -> str:
    """Generate a grounded answer from retrieved context. Never raises.

    Args:
        user_id:  Caller identity (gateway-forwarded; used for audit/log).
        query:    The original user question.
        results:  Retrieved SearchResult objects from the retriever.
    """
    if not results:
        return NO_DATA_FALLBACK

    prompt = build_prompt(query=query, results=results)

    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_answer failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_chat_response(user_id: str, query: str) -> str:
    """Generate a conversational response without RAG context. Never raises.

    Used for general conversation intents (greetings, jokes, general knowledge)
    that should bypass Qdrant entirely.

    Args:
        user_id:  Caller identity (used for audit/log only).
        query:    The raw user message.
    """
    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=CHAT_SYSTEM_PROMPT, user_prompt=query
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_chat_response failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_conversation_summary(
    user_id: str,
    data: ConversationSummaryData,  # type: ignore[name-defined]
) -> str:
    """Format a prose summary of a conversation transcript. Never raises.

    The transcript is read from MongoDB by the conversation_summary tool (no RAG,
    no Qdrant); this function only turns it into readable text.

    Args:
        user_id:  Caller identity (used for audit/log only).
        data:     Structured summary data (participants, transcript, topics, …).
    """
    prompt = build_summary_prompt(data)

    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=SUMMARY_SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_conversation_summary failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_community_summary(
    user_id: str,
    data: CommunitySummaryData,  # type: ignore[name-defined]
) -> str:
    """Generate a prose summary of a community from its MongoDB data.  Never raises."""
    prompt = build_community_summary_prompt(data)
    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=COMMUNITY_SUMMARY_SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_community_summary failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_channel_summary(
    user_id: str,
    data: ChannelSummaryData,  # type: ignore[name-defined]
) -> str:
    """Generate a prose summary of a channel from its MongoDB data.  Never raises."""
    prompt = build_channel_summary_prompt(data)
    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=CHANNEL_SUMMARY_SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_channel_summary failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


async def generate_catch_me_up_digest(
    user_id: str,
    data: CatchMeUpData,  # type: ignore[name-defined]
) -> str:
    """Generate a prose "while you were away" digest from MongoDB activity data.

    Never raises. Skips the LLM entirely when there's nothing to report at
    all (data.servers is empty) — a deterministic message is more reliable
    than round-tripping to the model for a fact it can't get wrong.
    """
    if not data.servers:
        return "You're all caught up — no new activity since you were last online."

    prompt = build_catch_me_up_prompt(data)
    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            answer = await groq_client.complete(
                system_prompt=CATCH_ME_UP_SYSTEM_PROMPT, user_prompt=prompt
            )
            return answer or _EMPTY

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return _UNAVAILABLE
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return _UNAVAILABLE
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return _UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_catch_me_up_digest failed for user %s: %s", user_id, exc)
        return _GENERIC_ERROR


def _parse_moderation_triage(raw: str) -> dict | None:
    """Parse the SEVERITY/ACTION/RATIONALE format. Returns None if the model
    didn't follow the format at all — the caller treats that as "triage
    unavailable" rather than guessing at a severity."""
    severity = None
    action = None
    rationale_parts: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SEVERITY:"):
            severity = stripped.split(":", 1)[1].strip().lower()
        elif stripped.upper().startswith("ACTION:"):
            action = stripped.split(":", 1)[1].strip().lower()
        elif stripped.upper().startswith("RATIONALE:"):
            rationale_parts.append(stripped.split(":", 1)[1].strip())
        elif rationale_parts:
            rationale_parts.append(stripped)

    if severity not in ("low", "medium", "high"):
        return None
    if action not in ("none", "warn", "mute", "ban", "escalate"):
        action = "none"
    rationale = " ".join(p for p in rationale_parts if p).strip() or "No rationale provided."
    return {"severity": severity, "suggestedAction": action, "rationale": rationale[:300]}


async def generate_moderation_triage(
    reason: str,
    details: str | None,
    content_type: str | None,
    snippet: str | None,
) -> dict | None:
    """Triage a filed UserReport for the admin queue. Returns None (never
    raises) on any failure — triage is a nice-to-have prioritization signal,
    never a gate on the report itself existing or being reviewable."""
    prompt = build_moderation_triage_prompt(reason, details, content_type, snippet)
    try:
        if settings.llm_provider == "groq":
            from app.llm import groq_client

            raw = await groq_client.complete(
                system_prompt=MODERATION_TRIAGE_SYSTEM_PROMPT, user_prompt=prompt
            )
            if not raw:
                return None
            return _parse_moderation_triage(raw)

        logger.error("unknown llm_provider configured: %s", settings.llm_provider)
        return None
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_moderation_triage failed: %s", exc)
        return None


async def generate_voice_session_summary(
    user_id: str,
    channel_name: str,
    server_name: str,
    participants: list[str],
    transcript: list[dict],
) -> str | None:
    """Generate a structured (SUMMARY/KEY POINTS/ACTION ITEMS) voice-session
    recap. Never raises — but unlike every other generate_* function in this
    module, returns None on any failure instead of a safe fallback STRING.
    The other functions' callers only ever display the returned text as a
    transient chat reply, where showing "AI assistant is not available" is
    an acceptable (if unfortunate) message. This function's caller
    (app.api.internal's /voice/summarize) persists the result to MongoDB AND
    semantically indexes it into Qdrant — silently saving/embedding a
    degraded placeholder string as if it were a real meeting summary would
    corrupt both the record and future search/recall results, so the caller
    needs to be able to tell "failed" apart from "here is real text" and
    surface a genuine retry instead.
    """
    prompt = build_voice_session_summary_prompt(
        channel_name=channel_name,
        server_name=server_name,
        participants=participants,
        transcript=transcript,
    )

    try:
        if settings.llm_provider != "groq":
            logger.error("unknown llm_provider configured: %s", settings.llm_provider)
            return None

        from app.llm import groq_client

        answer = await groq_client.complete(
            system_prompt=VOICE_SESSION_SUMMARY_SYSTEM_PROMPT, user_prompt=prompt
        )
        return answer.strip() or None
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return None
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return None
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_voice_session_summary failed for user %s: %s", user_id, exc)
        return None


async def transcribe_audio_chunk(
    audio_bytes: bytes,
    filename: str,
    language: str = "en",
    prompt: str | None = None,
) -> str | None:
    """Transcribe one short live-caption audio chunk. Never raises.

    Returns None ONLY on a genuine failure (misconfiguration, network/API
    error) — the HTTP layer maps that to a 503. Returns an empty string for
    a chunk that was processed successfully but had nothing worth
    transcribing (silence, or a filtered silence-hallucination like
    Whisper's infamous "Thank you." on dead air) — that's a normal, frequent,
    SUCCESSFUL outcome, not an error, and must stay distinguishable from a
    real failure or ordinary silence between sentences would look identical
    to the transcription service being down.
    """
    try:
        if settings.llm_provider != "groq":
            logger.error("unknown llm_provider configured: %s", settings.llm_provider)
            return None

        from app.llm import groq_client

        return await groq_client.transcribe(audio_bytes, filename, language=language, prompt=prompt)
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return None
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return None
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("transcribe_audio_chunk failed: %s", exc)
        return None


# Groq's Whisper endpoint (like OpenAI's, which it mirrors) caps upload size
# at 25MB — reels are allowed up to 100MB client-side (see
# frontend/components/ReelUploadModal.tsx), so a real fraction of uploads
# will legitimately exceed what transcription can accept. That's treated as
# "captions unavailable" for this reel, not a retryable error.
_MAX_TRANSCRIBE_BYTES = 25 * 1024 * 1024
_VIDEO_DOWNLOAD_TIMEOUT_S = 30.0


def _vtt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _build_webvtt(segments: list[dict]) -> str:
    """Format Whisper segments as a WebVTT track. Pure formatting, no I/O."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_vtt_timestamp(seg['start'])} --> {_vtt_timestamp(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


async def generate_reel_captions(video_url: str) -> str | None:
    """Download a reel's video and transcribe it into a WebVTT caption track.

    Never raises. Returns None when captions genuinely can't be produced
    (download failure, file too large for Whisper, no speech detected, LLM
    provider down) — the caller (internal.py) treats that as "unavailable,"
    never as a reason to fail reel creation, which has already completed by
    the time this runs (fire-and-forget from reelController.ts).

    Whisper accepts common video containers (mp4, webm, mov) directly and
    extracts the audio itself server-side — no local ffmpeg/audio-extraction
    step is needed here.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_VIDEO_DOWNLOAD_TIMEOUT_S) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            video_bytes = resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_reel_captions: failed to download %s: %s", video_url, exc)
        return None

    if not video_bytes:
        return None
    if len(video_bytes) > _MAX_TRANSCRIBE_BYTES:
        logger.info(
            "generate_reel_captions: %s is %d bytes, over the %d Whisper cap — skipping",
            video_url, len(video_bytes), _MAX_TRANSCRIBE_BYTES,
        )
        return None

    filename = video_url.rsplit("/", 1)[-1] or "reel.mp4"

    try:
        if settings.llm_provider != "groq":
            logger.error("unknown llm_provider configured: %s", settings.llm_provider)
            return None

        from app.llm import groq_client

        segments = await groq_client.transcribe_with_segments(video_bytes, filename)
    except RuntimeError as exc:
        logger.error("llm provider not configured: %s", exc)
        return None
    except ImportError:
        logger.error("llm provider SDK is not installed")
        return None
    except Exception as exc:  # noqa: BLE001 — graceful degradation boundary
        logger.error("generate_reel_captions transcription failed for %s: %s", video_url, exc)
        return None

    if not segments:
        return None
    return _build_webvtt(segments)


# NOTE: recommendations are NOT formatted by the LLM. They are rendered
# deterministically in app/api/internal._format_recommendations() so that
# chain-of-thought / invented entries can never leak. There is intentionally no
# generate_recommendation_response() here.

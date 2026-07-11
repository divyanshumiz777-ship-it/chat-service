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
    CHANNEL_SUMMARY_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    COMMUNITY_SUMMARY_SYSTEM_PROMPT,
    NO_DATA_FALLBACK,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VOICE_SESSION_SUMMARY_SYSTEM_PROMPT,
    build_channel_summary_prompt,
    build_community_summary_prompt,
    build_prompt,
    build_summary_prompt,
    build_voice_session_summary_prompt,
)

if TYPE_CHECKING:
    from app.rag.retriever import SearchResult
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


# NOTE: recommendations are NOT formatted by the LLM. They are rendered
# deterministically in app/api/internal._format_recommendations() so that
# chain-of-thought / invented entries can never leak. There is intentionally no
# generate_recommendation_response() here.

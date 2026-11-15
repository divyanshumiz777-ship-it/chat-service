"""Qdrant deletion/tombstone path.

Every ingest_* function in ingest_sync.py upserts content into Qdrant, but
until this module, nothing ever removed a point once its source document was
deleted in MongoDB — a deleted server/channel's messages and voice-session
recaps stayed searchable/recallable through the assistant forever. This
closes that gap.

Deletes by FIELD FILTER (server_id / channel_id / source_type+source_id),
not by enumerating individual point ids. Every chunk's AclPayload already
carries server_id and channel_id (see chunker.py's normalize_* functions),
and both are self-referential on their own "server"/"channel" chunk too —
so one filtered delete per collection removes the entity's own chunk AND
everything nested under it in a single call, without needing to know the
full list of child message/thread/voice_session ids up front.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.rag.qdrant_client import delete_by_fields, scroll_source_ids
from app.rag.rag_document_index import delete_index_entries, delete_index_entries_many

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.delete_sync")


def _collections() -> tuple[str, str]:
    return (settings.qdrant_collection_public, settings.qdrant_collection_private)


async def delete_by_server(server_id: str) -> None:
    """Remove every indexed chunk belonging to `server_id` — the server's own
    chunk plus every channel/thread/message/voice_session nested under it
    (all carry server_id in their payload). Called when a server is deleted.
    """
    for collection in _collections():
        try:
            await delete_by_fields(collection, {"server_id": server_id})
        except Exception:
            logger.exception(
                "qdrant delete_by_server failed for server=%s in %s", server_id, collection
            )


async def delete_by_channel(channel_id: str) -> None:
    """Remove every indexed chunk belonging to `channel_id` — the channel's
    own chunk plus its messages and voice sessions. Called when a channel is
    deleted (independent of its parent server, e.g. a channel deleted on its
    own rather than via a whole-server delete).
    """
    for collection in _collections():
        try:
            await delete_by_fields(collection, {"channel_id": channel_id})
        except Exception:
            logger.exception(
                "qdrant delete_by_channel failed for channel=%s in %s", channel_id, collection
            )


async def delete_by_conversation(conversation_id: str) -> None:
    """Remove every indexed chunk belonging to a DM conversation_id — its
    messages (see chunker.py's normalize_dm_message, which stamps this field).
    Called when a whole DM conversation is deleted (e.g. once every
    participant has left), so its content doesn't linger in Qdrant even
    though delete_source above already handles a single message delete.
    """
    for collection in _collections():
        try:
            await delete_by_fields(collection, {"conversation_id": conversation_id})
        except Exception:
            logger.exception(
                "qdrant delete_by_conversation failed for conversation=%s in %s",
                conversation_id,
                collection,
            )


async def delete_by_group(group_id: str) -> None:
    """Remove every indexed chunk belonging to a group-DM group_id — its
    messages (see chunker.py's normalize_group_message, which stamps this
    field). Called when a whole group DM is deleted.
    """
    for collection in _collections():
        try:
            await delete_by_fields(collection, {"group_id": group_id})
        except Exception:
            logger.exception(
                "qdrant delete_by_group failed for group=%s in %s", group_id, collection
            )


async def delete_by_user(user_id: str) -> None:
    """Remove everything indexed that is PERSONAL to `user_id` — called on
    account deactivate and delete:

      * the user's own profile point(s) (source_type "user"), so a deleted or
        deactivated account stops surfacing in search/the assistant, and
      * every notification point addressed to them (notifications are erased
        from MongoDB on account deletion; their vectors must go too),

    plus the matching rag_index_state rows. Content the user posted in shared
    spaces (messages, reels, comments) is intentionally NOT touched here — it
    stays, attributed to "Deleted user", and is removed by its own delete paths.
    """
    if not user_id:
        raise ValueError("delete_by_user requires a user id")

    notification_ids: set[str] = set()
    for collection in _collections():
        try:
            notification_ids.update(
                await scroll_source_ids(
                    collection, {"source_type": "notification", "allowed_user_ids": user_id}
                )
            )
        except Exception:
            logger.exception(
                "qdrant notification scan failed for user=%s in %s", user_id, collection
            )
        for conditions in (
            {"source_type": "user", "source_id": user_id},
            {"source_type": "notification", "allowed_user_ids": user_id},
        ):
            try:
                await delete_by_fields(collection, conditions)
            except Exception:
                logger.exception(
                    "qdrant delete_by_user failed for user=%s (%s) in %s",
                    user_id, conditions["source_type"], collection,
                )

    try:
        await delete_index_entries("user", user_id)
        await delete_index_entries_many("notification", sorted(notification_ids))
    except Exception:
        logger.exception("rag_index_state cleanup failed for user=%s", user_id)


async def delete_source(source_type: str, source_id: str) -> None:
    """Remove ONE specific source document's chunk(s) — for a single-entity
    delete (a thread, a standalone message) rather than a server/channel-wide
    cascade. Also cleans up its rag_index_state tracking row(s), which the
    bulk cascades above deliberately skip (see delete_index_entries' own
    comment for why that's fine there).
    """
    for collection in _collections():
        try:
            await delete_by_fields(
                collection, {"source_type": source_type, "source_id": source_id}
            )
        except Exception:
            logger.exception(
                "qdrant delete_source failed for %s/%s in %s", source_type, source_id, collection
            )
    try:
        await delete_index_entries(source_type, source_id)
    except Exception:
        logger.exception("rag_index_state cleanup failed for %s/%s", source_type, source_id)

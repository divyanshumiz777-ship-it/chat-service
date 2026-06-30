"""Shared helpers for the MongoDB-backed recommender tools.

Pure utility functions: ObjectId coercion, set building, recency checks, and
top-N ranking. NO Qdrant, NO semantic search — recommenders read MongoDB only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


def to_oid(value: Any) -> Any | None:
    """Coerce a value to a bson ObjectId, or None if it isn't a valid id."""
    from bson import ObjectId
    from bson.errors import InvalidId

    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except (InvalidId, TypeError):
        return None


def oid_set(values: Iterable[Any] | None) -> set:
    """Build a set of ObjectIds from a heterogeneous iterable, dropping invalids."""
    out: set = set()
    for v in values or []:
        o = to_oid(v)
        if o is not None:
            out.add(o)
    return out


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def is_recent(value: Any, days: int) -> bool:
    """True if `value` is a datetime within the last `days` days."""
    if not isinstance(value, datetime):
        return False
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return (now_utc() - dt) <= timedelta(days=days)


def plural(n: int, singular: str, suffix: str = "s") -> str:
    """'1 friend' / '2 friends' — small helper for human-readable reasons."""
    return f"{n} {singular}" if n == 1 else f"{n} {singular}{suffix}"


def top_n(scored: list[dict], limit: int) -> list[dict]:
    """Return the highest-scoring items (descending), capped at `limit`."""
    return sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[: max(0, limit)]

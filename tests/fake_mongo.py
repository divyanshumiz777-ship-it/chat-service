"""Minimal in-memory stand-in for the motor collections the tools/retriever use.

Supports exactly the query shapes this codebase issues: equality (including
"array contains" for list fields), dotted paths, $in, $nin, $ne, $exists,
$regex (+ $options "i"), and cursors with sort / limit / to_list.
"""

from __future__ import annotations

import re
from typing import Any

_MISSING = object()


def _get(doc: dict, path: str) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _eq(value: Any, expected: Any) -> bool:
    if isinstance(value, list) and not isinstance(expected, list):
        return expected in value
    return value == expected


def _match_cond(value: Any, cond: Any) -> bool:
    if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
        for op, arg in cond.items():
            if op == "$in":
                if value is _MISSING or not any(_eq(value, a) for a in arg):
                    return False
            elif op == "$nin":
                if value is not _MISSING and any(_eq(value, a) for a in arg):
                    return False
            elif op == "$ne":
                if value is not _MISSING and _eq(value, arg):
                    return False
            elif op == "$exists":
                if (value is not _MISSING) != bool(arg):
                    return False
            elif op == "$regex":
                flags = re.IGNORECASE if "i" in cond.get("$options", "") else 0
                if value is _MISSING or not re.search(arg, str(value), flags):
                    return False
            elif op == "$options":
                continue
            else:  # pragma: no cover — unsupported operator in a test
                raise NotImplementedError(op)
        return True
    if value is _MISSING:
        return cond is None
    return _eq(value, cond)


def matches(doc: dict, query: dict) -> bool:
    return all(_match_cond(_get(doc, k), v) for k, v in (query or {}).items())


class FakeCursor:
    def __init__(self, docs: list[dict]):
        self._docs = list(docs)

    def sort(self, key, direction=None):
        if isinstance(key, list):
            key, direction = key[0]
        self._docs.sort(
            key=lambda d: (_get(d, key) is _MISSING, _get(d, key) if _get(d, key) is not _MISSING else 0),
            reverse=(direction == -1),
        )
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length=None):
        return self._docs if length is None else self._docs[:length]

    def __aiter__(self):
        self._it = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class FakeCollection:
    def __init__(self, docs: list[dict] | None = None, fail: bool = False):
        self.docs = list(docs or [])
        self.fail = fail
        self.queries: list[dict] = []

    def find(self, query=None, projection=None):
        if self.fail:
            raise RuntimeError("simulated mongo failure")
        self.queries.append(query or {})
        return FakeCursor([d for d in self.docs if matches(d, query or {})])

    async def find_one(self, query=None, projection=None, sort=None):
        if self.fail:
            raise RuntimeError("simulated mongo failure")
        self.queries.append(query or {})
        cur = FakeCursor([d for d in self.docs if matches(d, query or {})])
        if sort:
            cur.sort(sort)
        docs = await cur.to_list()
        return docs[0] if docs else None


class FakeDb:
    def __init__(self, **collections: FakeCollection):
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())

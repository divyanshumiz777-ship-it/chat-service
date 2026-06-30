"""Integration tests: GET /internal/v1/capabilities.

Verifies the complete HTTP layer — status codes, auth enforcement,
response shape, and exact payload match — without requiring MongoDB,
Qdrant, or Redis. The capabilities endpoint is pure Python; only the
FastAPI + auth middleware stack needs to run.

Run:
    pytest services/chat-service/tests/test_internal_capabilities.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import require_internal_token
from app.llm.capabilities import CAPABILITIES, capabilities_payload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def anon_client() -> TestClient:
    """TestClient with no auth override — real require_internal_token enforced."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def authed_client() -> TestClient:
    """TestClient with internal-token auth bypassed via dependency_overrides.

    This isolates endpoint logic from token configuration so the test does not
    depend on INTERNAL_SERVICE_TOKEN being set in the environment.
    """
    async def _always_ok() -> None:
        return

    app.dependency_overrides[require_internal_token] = _always_ok
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        app.dependency_overrides.pop(require_internal_token, None)


# ---------------------------------------------------------------------------
# Authentication tests (real token guard)
# ---------------------------------------------------------------------------


def test_missing_token_returns_401_or_503(anon_client: TestClient) -> None:
    """GET without X-Internal-Token must be rejected (401) or return 503 when
    no token is configured in settings.
    """
    resp = anon_client.get("/internal/v1/capabilities")
    assert resp.status_code in (401, 503), (
        f"Expected 401 or 503 (no token), got {resp.status_code}: {resp.text}"
    )


def test_wrong_token_returns_401_or_503(anon_client: TestClient) -> None:
    """GET with a wrong X-Internal-Token must be rejected."""
    resp = anon_client.get(
        "/internal/v1/capabilities",
        headers={"X-Internal-Token": "definitely-wrong-token-xyz"},
    )
    assert resp.status_code in (401, 503), (
        f"Expected 401 or 503 (wrong token), got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# Endpoint shape / payload tests (auth bypassed)
# ---------------------------------------------------------------------------


def test_returns_200_ok(authed_client: TestClient) -> None:
    """Authenticated GET → 200 OK."""
    resp = authed_client.get("/internal/v1/capabilities")
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"


def test_response_has_capabilities_key(authed_client: TestClient) -> None:
    """Response body must have a top-level 'capabilities' key."""
    resp = authed_client.get("/internal/v1/capabilities")
    data = resp.json()
    assert "capabilities" in data, f"Missing 'capabilities' in: {list(data.keys())}"


def test_capabilities_is_non_empty_list(authed_client: TestClient) -> None:
    """capabilities must be a non-empty list."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    assert isinstance(caps, list), f"Expected list, got {type(caps)}"
    assert len(caps) > 0, "capabilities list must not be empty"


def test_capability_required_fields(authed_client: TestClient) -> None:
    """Every capability must have the fields consumed by the frontend palette."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    required = {"name", "label", "icon", "category", "command", "description",
                "aliases", "slash", "examples"}
    for cap in caps:
        missing = required - set(cap)
        assert not missing, (
            f"Capability {cap.get('name')!r} missing fields: {sorted(missing)}"
        )


def test_at_least_15_visible_capabilities(authed_client: TestClient) -> None:
    """Registry has 16 capabilities (1 hidden = community_lookup) → at least 15 visible."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    assert len(caps) >= 15, f"Expected ≥15, got {len(caps)}"


def test_hidden_capabilities_excluded(authed_client: TestClient) -> None:
    """hidden=True capabilities must not appear in the palette payload."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    names = {c["name"] for c in caps}
    hidden_names = {c.name for c in CAPABILITIES if c.hidden}
    overlap = hidden_names & names
    assert not overlap, f"Hidden capabilities leaked into response: {overlap}"


def test_payload_matches_capabilities_payload(authed_client: TestClient) -> None:
    """HTTP response must be byte-for-byte identical to capabilities_payload()."""
    http_caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    assert http_caps == capabilities_payload(), (
        "HTTP response diverged from capabilities_payload() — "
        "endpoint is not calling capabilities_payload() directly"
    )


def test_all_categories_present(authed_client: TestClient) -> None:
    """Every expected category must appear in the payload."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    categories = {c["category"] for c in caps}
    expected = {"Social", "Messages", "Content", "Search", "Recommendations", "Insights"}
    missing = expected - categories
    assert not missing, f"Missing categories in payload: {missing}"


def test_fields_are_correct_types(authed_client: TestClient) -> None:
    """aliases, slash, and examples must be lists (not tuples or other types)."""
    caps = authed_client.get("/internal/v1/capabilities").json()["capabilities"]
    for cap in caps:
        for list_field in ("aliases", "slash", "examples"):
            val = cap[list_field]
            assert isinstance(val, list), (
                f"{cap['name']}.{list_field} should be list, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Route registration verification
# ---------------------------------------------------------------------------


def test_route_is_registered_in_app() -> None:
    """GET /internal/v1/capabilities must be registered in the FastAPI app.

    Verifies items 1 and 2 from the spec:
      1. The endpoint is defined in internal.py
      2. The router is included in main.py
    """
    routes = {(r.path, m) for r in app.routes for m in getattr(r, "methods", [])}
    assert ("/internal/v1/capabilities", "GET") in routes, (
        "Route GET /internal/v1/capabilities is not registered. "
        "Check that internal_router is included in main.py with prefix='/internal/v1'."
    )

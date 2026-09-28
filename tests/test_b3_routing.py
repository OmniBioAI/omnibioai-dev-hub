"""B3 deterministic routing and no-fallback contract tests."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.discovery import router
from integrations.discovery_routing import DiscoveryRoutingError, route_query


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_explicit_modes_select_only_their_authority():
    assert route_query("models").selected_source == "model-registry"
    assert route_query("service").selected_source == "docs-api-catalog"
    assert route_query("api").selected_source == "docs-api-catalog"
    assert route_query("tools").selected_source == "tes-discovery"
    assert route_query("documentation").authority_class == "DOCUMENTATION"


def test_unknown_intent_is_not_semantically_guessed():
    try:
        route_query(None)
    except DiscoveryRoutingError as exc:
        assert exc.code == "UNKNOWN_STRUCTURED_INTENT"
    else:
        raise AssertionError("expected unknown structured intent")


def test_workflow_and_plugin_are_not_routable_yet():
    for entity_type, code in (("workflow", "WORKFLOW_NOT_ROUTABLE_YET"), ("plugin", "PLUGIN_NOT_ROUTABLE_YET")):
        try:
            route_query(entity_type)
        except DiscoveryRoutingError as exc:
            assert exc.code == code
            assert exc.status_code == 501
        else:
            raise AssertionError("expected parallel adapter to remain unroutable")


def test_documentation_route_is_explicit_and_does_not_execute_rag():
    with patch("api.routes.discovery.require_auth", return_value="actor"):
        response = client().get("/api/discovery/query", params={"entity_type": "documentation", "q": "Explain Dev Hub"})
    assert response.status_code == 200
    assert response.json()["routing"]["authority_class"] == "DOCUMENTATION"
    assert response.json()["next_route"] == "/rag/query"


def test_structured_source_failure_is_not_fallback():
    with patch("api.routes.discovery._model_call", side_effect=Exception("should not be swallowed")):
        # Direct router selection remains deterministic; adapter exceptions are not converted to RAG.
        assert route_query("model").selected_source == "model-registry"


def test_auth_is_required_for_routing_endpoint(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")
    monkeypatch.setenv("JWT_ISSUER", "iss")
    response = client().get("/api/discovery/query", params={"entity_type": "model"})
    assert response.status_code == 401

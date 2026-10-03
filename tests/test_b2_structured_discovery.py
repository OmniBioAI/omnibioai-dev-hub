"""B2 model and service/API structured-discovery contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.discovery import router
from integrations.model_discovery import ModelDiscoveryClient, ModelDiscoveryError
from integrations.service_api_discovery import (
    ServiceAPIDiscoveryClient,
    ServiceAPIDiscoveryError,
)


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def model_meta(**extra):
    result = {"task": "classification", "model_name": "rnaseq", "version": "1.2.0", "description": "safe metadata", "promotion": "staging", "aliases": ["latest"], "package_dir": "/should/not/escape"}
    result.update(extra)
    return result


def test_model_list_emits_b1_envelope_and_preserves_model_identity(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_URL", "http://model-registry.test")
    with patch("integrations.model_discovery.requests.get", return_value=Response([model_meta()])) as get:
        result = ModelDiscoveryClient.from_environment().list_models(authorization="Bearer token")
    item = result[0]
    assert item["contract_version"] == "1.0"
    assert item["entity_type"] == "model"
    assert item["entity_id"]["source_local_id"] == {"task": "classification", "model_name": "rnaseq", "version": "1.2.0"}
    assert item["canonical_key"].startswith("model:model-registry:")
    assert item["freshness"]["state"] == "LIVE"
    assert item["visibility"]["metadata"] == "MODEL_READ"
    assert "package_dir" not in item["payload"]
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer token"}


def test_model_show_uses_read_only_source_endpoint_and_alias(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_URL", "http://model-registry.test")
    with patch("integrations.model_discovery.requests.get", return_value=Response({"ok": True, "meta": model_meta()})) as get:
        result = ModelDiscoveryClient.from_environment().get_model("classification", "rnaseq", alias="latest", authorization="Bearer token")
    assert result["payload"]["aliases"] == ["latest"]
    assert get.call_args.kwargs["params"] == {"task": "classification", "ref": "latest", "verify": "false"}


def test_model_403_is_not_converted_to_empty_or_rag(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_URL", "http://model-registry.test")
    with patch("integrations.model_discovery.requests.get", return_value=Response({}, 403)):
        try:
            ModelDiscoveryClient.from_environment().list_models(authorization="Bearer token")
        except ModelDiscoveryError as exc:
            assert exc.code == "MODEL_REGISTRY_FORBIDDEN"
            assert exc.status_code == 403
        else:
            raise AssertionError("expected authorization failure")


def test_service_catalog_emits_snapshot_envelope_and_source_provenance(tmp_path: Path, monkeypatch):
    catalog = {
        "_meta": {"generated_at": "2026-09-19T02:32:40Z", "generator": "generate-api-catalog.mjs", "verification_state": "IMPLEMENTED_IN_SOURCE"},
        "services": [{"service_id": "auth", "repository": "omnibioai-auth", "source_revision": "abc123", "default_visibility": "REVIEW_REQUIRED"}],
        "routes": [{"route_id": "auth:GET:/v1/auth", "service_id": "auth", "method": "GET", "declared_path": "/v1/auth", "visibility": "AUTHENTICATED_PUBLIC", "auth_status": "EXPLICIT", "source": {"source_path": "app/routes/auth.py", "source_revision": "abc123"}}],
    }
    path = tmp_path / "api-catalog.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setenv("SERVICE_API_CATALOG_PATH", str(path))
    service = ServiceAPIDiscoveryClient.from_environment().get_service("auth")
    api = ServiceAPIDiscoveryClient.from_environment().list_apis(service_id="auth")[0]
    assert service["entity_type"] == "service"
    assert api["entity_id"]["source_local_id"] == "auth:GET:/v1/auth"
    assert api["provenance"]["source_revision"] == "abc123"
    assert api["freshness"]["state"] == "SNAPSHOT"
    assert api["lifecycle"]["deployed"] == "UNKNOWN"
    assert api["visibility"]["discovery"] == "AUTHENTICATED_PUBLIC"


def test_service_catalog_malformed_and_unavailable_are_typed(tmp_path: Path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("SERVICE_API_CATALOG_PATH", str(missing))
    try:
        ServiceAPIDiscoveryClient.from_environment().list_services()
    except ServiceAPIDiscoveryError as exc:
        assert exc.code == "SERVICE_API_CATALOG_UNAVAILABLE"
    else:
        raise AssertionError("expected unavailable catalog")

    missing.write_text("{bad")
    try:
        ServiceAPIDiscoveryClient.from_environment().list_services()
    except ServiceAPIDiscoveryError as exc:
        assert exc.code == "SERVICE_API_CATALOG_MALFORMED"
    else:
        raise AssertionError("expected malformed catalog")


def test_b2_routes_are_auth_protected_and_do_not_execute_sources(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")
    monkeypatch.setenv("JWT_ISSUER", "iss")
    with patch("integrations.model_discovery.requests.get") as model_get, patch("integrations.service_api_discovery.Path.read_bytes") as catalog_read:
        response = client().get("/api/discovery/models")
        service_response = client().get("/api/discovery/services")
    assert response.status_code == 401
    assert service_response.status_code == 401
    model_get.assert_not_called()
    catalog_read.assert_not_called()

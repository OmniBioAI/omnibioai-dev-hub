"""B4 explicit relationship projections and safety boundaries."""

from __future__ import annotations

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.discovery import router
from integrations.b1_contract import envelope
from integrations.relationship_discovery import (
    SERVICE_EXPOSES_API,
    TOOL_COMPATIBLE_WITH_BACKEND,
    RelationshipDiscoveryClient,
    RelationshipDiscoveryError,
)


def _service(service_id: str = "auth") -> dict:
    return envelope(
        entity_type="service", source_namespace="docs/api-catalog", source_local_id=service_id,
        display_name=service_id, summary=None, source={"source_kind": "GENERATED_API_CATALOG"},
        authority={"evidence_class": "GENERATED_PROJECTION"},
        provenance={"source_repository": "omnibioai-docs", "source_path": "api-catalog.json", "source_revision": "r1"},
        freshness={"state": "SNAPSHOT"}, visibility={"discovery": "PUBLIC"}, payload={},
    )


def _api() -> dict:
    return envelope(
        entity_type="api", source_namespace="service/auth", source_local_id="auth:GET:/v1/auth",
        display_name="GET /v1/auth", summary=None, source={"source_kind": "GENERATED_API_CATALOG"},
        authority={"evidence_class": "GENERATED_PROJECTION"},
        provenance={"source_repository": "omnibioai-docs", "source_path": "api-catalog.json", "source_revision": "r1"},
        freshness={"state": "SNAPSHOT"}, visibility={"discovery": "PUBLIC"}, payload={},
    )


def test_service_api_relationship_is_explicit_and_provenanced():
    source = Mock()
    source.get_service.return_value = _service()
    source.list_apis.return_value = [_api()]
    edge = RelationshipDiscoveryClient(service_api=source).query(
        source_entity_type="service", source_local_id="auth", relationship_type=SERVICE_EXPOSES_API
    )[0]
    assert edge["relationship_type"] == SERVICE_EXPOSES_API
    assert edge["source_entity"] == "service:docs%2Fapi-catalog:auth"
    assert edge["target_entity"] == "api:service%2Fauth:auth%3AGET%3A%2Fv1%2Fauth"
    assert edge["evidence"]["classification"] == "EXPLICIT"
    assert edge["freshness"]["state"] == "SNAPSHOT"


def test_tool_backend_relationship_uses_explicit_backend_field():
    source = Mock()
    source.get_tool.return_value = {
        "tool_id": "fastqc", "backend_capabilities": ["HTTP", "Slurm"],
        "provenance": {"source_path": "configs/tools/fastqc.yaml", "source_revision": "r2"},
        "visibility": {"discovery": "PUBLIC"}, "freshness": {"state": "LIVE"},
    }
    edges = RelationshipDiscoveryClient(tes=source).query(
        source_entity_type="tool", source_local_id="fastqc", relationship_type=TOOL_COMPATIBLE_WITH_BACKEND
    )
    assert [edge["target_entity"] for edge in edges] == [
        "backend:tes:HTTP", "backend:tes:Slurm"
    ]
    assert all(edge["evidence"]["source_field_or_rule"] == "backend_capabilities" for edge in edges)


def test_unsupported_and_unroutable_relationships_are_typed():
    client = RelationshipDiscoveryClient()
    try:
        client.query(source_entity_type="workflow", source_local_id="x")
    except RelationshipDiscoveryError as exc:
        assert (exc.code, exc.status_code) == ("NOT_ROUTABLE_YET", 501)
    else:
        raise AssertionError("expected workflow deferral")
    try:
        client.query(source_entity_type="service", source_local_id="x", relationship_type="INFERRED")
    except RelationshipDiscoveryError as exc:
        assert (exc.code, exc.status_code) == ("UNSUPPORTED_RELATIONSHIP", 422)
    else:
        raise AssertionError("expected unsupported relationship")


def test_relationship_route_is_authenticated_and_non_executing(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")
    monkeypatch.setenv("JWT_ISSUER", "iss")
    response = TestClient(app).get("/api/discovery/relationships", params={"source_entity_type": "tool", "source_local_id": "fastqc"})
    assert response.status_code == 401


def test_unconfigured_sources_remain_typed_failures(monkeypatch):
    monkeypatch.delenv("TES_DISCOVERY_URL", raising=False)
    monkeypatch.delenv("SERVICE_API_CATALOG_PATH", raising=False)
    client = RelationshipDiscoveryClient()
    for entity_type, code in (("tool", "TES_NOT_CONFIGURED"), ("service", "SERVICE_API_CATALOG_NOT_CONFIGURED")):
        try:
            client.query(source_entity_type=entity_type, source_local_id="missing")
        except RelationshipDiscoveryError as exc:
            assert exc.code == code
        else:
            raise AssertionError("expected typed source configuration failure")

"""Behavioural tests for every /api/discovery route with REAL authentication.

AUTH_ENABLED=true and HS256 tokens are validated by the production
require_auth dependency; only the upstream discovery sources (TES, Model
Registry, service-API catalog, relationship sources) are replaced by fakes.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import discovery
from integrations.model_discovery import ModelDiscoveryError
from integrations.relationship_discovery import RelationshipDiscoveryError
from integrations.service_api_discovery import ServiceAPIDiscoveryError
from integrations.tes_discovery import TESDiscoveryError

JWT_SECRET = "synthetic-discovery-jwt-secret-0123456789abcdef"
AUD, ISS = "devhub-discovery-test", "v12g-test"


def token(**overrides) -> str:
    claims = {"sub": "alice", "aud": AUD, "iss": ISS, "exp": int(time.time()) + 300}
    claims.update(overrides)
    return jwt.encode(claims, JWT_SECRET, algorithm="HS256")


AUTH = {"Authorization": f"Bearer {token()}"}


class Recorder:
    """Fake source: records calls and returns/raises a configured outcome per method."""

    def __init__(self, **outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_") or name in ("outcomes", "calls"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            outcome = self.outcomes[name]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return method


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("JWT_AUDIENCE", AUD)
    monkeypatch.setenv("JWT_ISSUER", ISS)
    app = FastAPI()
    app.include_router(discovery.router)
    return TestClient(app)


@pytest.fixture
def sources(monkeypatch):
    fakes = {"tes": Recorder(), "model": Recorder(), "service": Recorder(), "relationship": Recorder()}
    monkeypatch.setattr(discovery, "_client", lambda: fakes["tes"])
    monkeypatch.setattr(discovery.ModelDiscoveryClient, "from_environment", classmethod(lambda cls: fakes["model"]))
    monkeypatch.setattr(discovery.ServiceAPIDiscoveryClient, "from_environment", classmethod(lambda cls: fakes["service"]))
    monkeypatch.setattr(discovery, "RelationshipDiscoveryClient", lambda: fakes["relationship"])
    return fakes


ALL_ROUTES = [
    "/api/discovery/tools/fastqc",
    "/api/discovery/search?q=align",
    "/api/discovery/data-types",
    "/api/discovery/categories",
    "/api/discovery/catalog/version",
    "/api/discovery/catalog/status",
    "/api/discovery/models",
    "/api/discovery/models/classification/resnet",
    "/api/discovery/services",
    "/api/discovery/services/auth",
    "/api/discovery/services/auth/apis",
    "/api/discovery/apis",
    "/api/discovery/query?entity_type=tool",
    "/api/discovery/relationships?source_entity_type=tool&source_local_id=fastqc",
]


# --- authentication ---------------------------------------------------------

@pytest.mark.parametrize("path", ALL_ROUTES)
@pytest.mark.parametrize("headers, why", [
    ({}, "missing"),
    ({"Authorization": "Bearer not.a.jwt"}, "Invalid token"),
    ({"Authorization": f"Bearer {token(exp=int(time.time()) - 10)}"}, "expired"),
    ({"Authorization": f"Bearer {token(aud='another-service')}"}, "Invalid token"),
    ({"Authorization": f"Bearer {jwt.encode({'sub': 'mallory', 'aud': AUD, 'iss': ISS, 'exp': int(time.time()) + 300}, 'wrong-key-0123456789-0123456789-xx', algorithm='HS256')}"}, "Invalid token"),
    ({"Authorization": f"Basic {token()}"}, "Bearer"),
    ({"X-Devhub-Internal": JWT_SECRET}, "missing"),
])
def test_every_discovery_route_rejects_unauthenticated_requests_without_touching_sources(client, sources, path, headers, why):
    response = client.get(path, headers=headers)
    assert response.status_code == 401
    assert why.lower() in response.json()["detail"].lower()
    assert all(not fake.calls for fake in sources.values())


# --- TES tool discovery -----------------------------------------------------

def test_get_tool_returns_source_record_for_authenticated_actor(client, sources):
    sources["tes"].outcomes["get_tool"] = {"tool_id": "fastqc", "configuration_status": "CONFIGURED"}
    response = client.get("/api/discovery/tools/fastqc", headers=AUTH)
    assert response.status_code == 200 and response.json()["tool_id"] == "fastqc"
    assert sources["tes"].calls == [("get_tool", ("fastqc",), {})]


def test_get_tool_missing_resource_is_a_typed_404(client, sources):
    sources["tes"].outcomes["get_tool"] = TESDiscoveryError("TOOL_NOT_FOUND", "no such tool", 404)
    response = client.get("/api/discovery/tools/nope", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["detail"] == {"code": "TOOL_NOT_FOUND", "message": "no such tool"}


def test_search_forwards_every_filter_and_paging(client, sources):
    sources["tes"].outcomes["search"] = {"items": [], "total": 0}
    response = client.get("/api/discovery/search", headers=AUTH, params={
        "q": "align", "input_type": "fastq", "output_type": "bam", "category": "alignment",
        "backend": "local", "architecture": "arm64", "tool_id": "bwa", "limit": 5, "offset": 10})
    assert response.status_code == 200 and response.json() == {"items": [], "total": 0}
    assert sources["tes"].calls == [("search", (), {
        "q": "align", "input_type": "fastq", "output_type": "bam", "category": "alignment",
        "backend": "local", "architecture": "arm64", "tool_id": "bwa", "limit": 5, "offset": 10})]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "many"}])
def test_search_rejects_out_of_range_paging_before_calling_the_source(client, sources, params):
    response = client.get("/api/discovery/search", headers=AUTH, params=params)
    assert response.status_code == 422
    assert not sources["tes"].calls


@pytest.mark.parametrize("path, method", [
    ("/api/discovery/data-types", "data_types"),
    ("/api/discovery/categories", "categories"),
    ("/api/discovery/catalog/version", "catalog_version"),
    ("/api/discovery/catalog/status", "catalog_status"),
])
def test_catalog_metadata_routes_return_source_payloads(client, sources, path, method):
    sources["tes"].outcomes[method] = {"method": method, "items": ["a"]}
    response = client.get(path, headers=AUTH)
    assert response.status_code == 200 and response.json() == {"method": method, "items": ["a"]}
    assert sources["tes"].calls == [(method, (), {})]


@pytest.mark.parametrize("path, method", [
    ("/api/discovery/data-types", "data_types"),
    ("/api/discovery/catalog/status", "catalog_status"),
])
def test_tes_backend_unavailability_is_a_typed_503_not_an_empty_result(client, sources, path, method):
    sources["tes"].outcomes[method] = TESDiscoveryError("TES_UNAVAILABLE", "connection refused")
    response = client.get(path, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "TES_UNAVAILABLE"


def test_unexpected_source_exception_is_not_swallowed_into_an_empty_result(client, sources):
    sources["tes"].outcomes["categories"] = ValueError("malformed upstream payload")
    with pytest.raises(ValueError, match="malformed upstream payload"):
        client.get("/api/discovery/categories", headers=AUTH)


# --- Model Registry -----------------------------------------------------------

def test_list_models_forwards_the_callers_authorization_to_the_registry(client, sources):
    sources["model"].outcomes["list_models"] = [{"model_name": "resnet"}]
    response = client.get("/api/discovery/models", headers=AUTH, params={"task": "classification", "version": "3"})
    assert response.status_code == 200 and response.json() == {"items": [{"model_name": "resnet"}]}
    assert sources["model"].calls == [("list_models", (), {
        "authorization": AUTH["Authorization"], "task": "classification", "model_name": None, "version": "3"})]


def test_list_models_empty_result_is_an_empty_list(client, sources):
    sources["model"].outcomes["list_models"] = []
    response = client.get("/api/discovery/models", headers=AUTH)
    assert response.status_code == 200 and response.json() == {"items": []}


def test_registry_authorization_failure_is_propagated_not_hidden(client, sources):
    sources["model"].outcomes["list_models"] = ModelDiscoveryError("MODEL_REGISTRY_FORBIDDEN", "actor may not list", 403)
    response = client.get("/api/discovery/models", headers=AUTH)
    assert response.status_code == 403
    assert response.json()["detail"] == {"code": "MODEL_REGISTRY_FORBIDDEN", "message": "actor may not list"}


def test_get_model_by_alias(client, sources):
    sources["model"].outcomes["get_model"] = {"model_name": "resnet", "alias": "prod"}
    response = client.get("/api/discovery/models/classification/resnet", headers=AUTH, params={"alias": "prod"})
    assert response.status_code == 200 and response.json()["alias"] == "prod"
    assert sources["model"].calls == [("get_model", ("classification", "resnet"), {
        "version": None, "alias": "prod", "authorization": AUTH["Authorization"]})]


def test_get_model_with_version_and_alias_is_ambiguous_and_never_queried(client, sources):
    response = client.get("/api/discovery/models/classification/resnet", headers=AUTH, params={"version": "1", "alias": "prod"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "MODEL_ID_AMBIGUOUS"
    assert not sources["model"].calls


def test_get_model_missing_is_a_typed_404(client, sources):
    sources["model"].outcomes["get_model"] = ModelDiscoveryError("MODEL_NOT_FOUND", "unknown model", 404)
    response = client.get("/api/discovery/models/classification/ghost", headers=AUTH)
    assert response.status_code == 404 and response.json()["detail"]["code"] == "MODEL_NOT_FOUND"


# --- service / API catalog ----------------------------------------------------

def test_list_services_and_get_service(client, sources):
    sources["service"].outcomes["list_services"] = [{"service_id": "auth"}]
    sources["service"].outcomes["get_service"] = {"service_id": "auth", "apis": 3}
    assert client.get("/api/discovery/services", headers=AUTH, params={"query": "au"}).json() == {"items": [{"service_id": "auth"}]}
    assert client.get("/api/discovery/services/auth", headers=AUTH).json() == {"service_id": "auth", "apis": 3}
    assert sources["service"].calls == [("list_services", (), {"query": "au"}), ("get_service", ("auth",), {})]


def test_service_catalog_missing_service_is_a_typed_404(client, sources):
    sources["service"].outcomes["get_service"] = ServiceAPIDiscoveryError("SERVICE_NOT_FOUND", "unknown service", 404)
    response = client.get("/api/discovery/services/ghost", headers=AUTH)
    assert response.status_code == 404 and response.json()["detail"]["code"] == "SERVICE_NOT_FOUND"


def test_service_api_listing_forwards_the_visibility_restriction(client, sources):
    sources["service"].outcomes["list_apis"] = [{"route_id": "auth:GET:/v1/me", "visibility": "AUTHENTICATED_PUBLIC"}]
    response = client.get("/api/discovery/services/auth/apis", headers=AUTH,
                          params={"method": "GET", "visibility": "AUTHENTICATED_PUBLIC"})
    assert response.status_code == 200 and len(response.json()["items"]) == 1
    assert sources["service"].calls == [("list_apis", (), {"service_id": "auth", "method": "GET", "visibility": "AUTHENTICATED_PUBLIC"})]


def test_global_api_listing_forwards_every_filter(client, sources):
    sources["service"].outcomes["list_apis"] = []
    response = client.get("/api/discovery/apis", headers=AUTH,
                          params={"service_id": "auth", "method": "POST", "visibility": "INTERNAL", "route_id": "r1"})
    assert response.status_code == 200 and response.json() == {"items": []}
    assert sources["service"].calls == [("list_apis", (), {"service_id": "auth", "method": "POST", "visibility": "INTERNAL", "route_id": "r1"})]


def test_malformed_service_catalog_is_a_typed_error(client, sources):
    sources["service"].outcomes["list_apis"] = ServiceAPIDiscoveryError("SERVICE_API_CATALOG_MALFORMED", "bad json", 502)
    response = client.get("/api/discovery/apis", headers=AUTH)
    assert response.status_code == 502 and response.json()["detail"]["code"] == "SERVICE_API_CATALOG_MALFORMED"


# --- structured query routing ------------------------------------------------

def test_structured_query_requires_an_entity_type(client, sources):
    assert client.get("/api/discovery/query", headers=AUTH).status_code == 422


@pytest.mark.parametrize("entity_type, status, code", [
    ("banana", 400, "UNKNOWN_STRUCTURED_INTENT"),
    ("workflow", 501, "WORKFLOW_NOT_ROUTABLE_YET"),
    ("plugin", 501, "PLUGIN_NOT_ROUTABLE_YET"),
])
def test_structured_query_unroutable_intents_are_typed_and_never_fall_back(client, sources, entity_type, status, code):
    response = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": entity_type, "q": "x"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert all(not fake.calls for fake in sources.values())


def test_structured_query_tool_uses_only_tes_search(client, sources):
    sources["tes"].outcomes["search"] = {"items": [{"tool_id": "bwa"}]}
    body = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": "tool", "q": "align", "tool_id": "bwa"}).json()
    assert body["routing"]["selected_source"] == "tes-discovery" and body["result"] == {"items": [{"tool_id": "bwa"}]}
    assert sources["tes"].calls == [("search", (), {"q": "align", "tool_id": "bwa"})]
    assert not sources["model"].calls and not sources["service"].calls


def test_structured_query_model_uses_only_the_registry(client, sources):
    sources["model"].outcomes["list_models"] = []
    body = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": "model", "task": "t", "model_name": "m"}).json()
    assert body["routing"]["selected_source"] == "model-registry" and body["result"] == []
    assert sources["model"].calls == [("list_models", (), {"authorization": AUTH["Authorization"], "task": "t", "model_name": "m", "version": None})]
    assert not sources["tes"].calls


@pytest.mark.parametrize("params, expected_query", [({"q": "auth"}, "auth"), ({"service_id": "billing"}, "billing")])
def test_structured_query_service_uses_the_catalog(client, sources, params, expected_query):
    sources["service"].outcomes["list_services"] = [{"service_id": expected_query}]
    body = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": "service", **params}).json()
    assert body["routing"]["selected_source"] == "docs-api-catalog"
    assert sources["service"].calls == [("list_services", (), {"query": expected_query})]


def test_structured_query_api_forwards_visibility_to_the_catalog(client, sources):
    sources["service"].outcomes["list_apis"] = []
    body = client.get("/api/discovery/query", headers=AUTH,
                      params={"entity_type": "api", "service_id": "auth", "visibility": "AUTHENTICATED_PUBLIC"}).json()
    assert body["result"] == []
    assert sources["service"].calls == [("list_apis", (), {"service_id": "auth", "method": None, "visibility": "AUTHENTICATED_PUBLIC", "route_id": None})]


def test_structured_query_documentation_points_to_rag_without_querying_sources(client, sources):
    body = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": "documentation", "q": "What is Dev Hub?"}).json()
    assert body["next_route"] == "/rag/query" and body["query"] == "What is Dev Hub?"
    assert all(not fake.calls for fake in sources.values())


def test_structured_query_source_failure_is_returned_not_rerouted(client, sources):
    sources["model"].outcomes["list_models"] = ModelDiscoveryError("MODEL_REGISTRY_UNAVAILABLE", "down", 503)
    response = client.get("/api/discovery/query", headers=AUTH, params={"entity_type": "model"})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "MODEL_REGISTRY_UNAVAILABLE"
    assert not sources["tes"].calls and not sources["service"].calls


# --- relationships ------------------------------------------------------------

def test_relationships_are_returned_bounded_and_never_executable(client, sources):
    edge = {"relationship_type": "TOOL_HAS_BACKEND", "target_entity": "backend:local"}
    sources["relationship"].outcomes["query"] = [edge]
    response = client.get("/api/discovery/relationships", headers=AUTH,
                          params={"source_entity_type": "tool", "source_local_id": "fastqc", "relationship_type": "TOOL_HAS_BACKEND"})
    assert response.status_code == 200
    assert response.json() == {"items": [edge], "count": 1, "execution_allowed": False}
    assert sources["relationship"].calls == [("query", (), {
        "source_entity_type": "tool", "source_local_id": "fastqc", "relationship_type": "TOOL_HAS_BACKEND"})]


def test_relationships_empty_result(client, sources):
    sources["relationship"].outcomes["query"] = []
    response = client.get("/api/discovery/relationships", headers=AUTH, params={"source_entity_type": "service", "source_local_id": "x"})
    assert response.json() == {"items": [], "count": 0, "execution_allowed": False}


def test_relationship_source_errors_are_typed(client, sources):
    sources["relationship"].outcomes["query"] = RelationshipDiscoveryError("RELATIONSHIP_SOURCE_UNSUPPORTED", "unsupported entity", 400)
    response = client.get("/api/discovery/relationships", headers=AUTH, params={"source_entity_type": "workflow", "source_local_id": "x"})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "RELATIONSHIP_SOURCE_UNSUPPORTED"


def test_relationships_require_both_source_identifiers(client, sources):
    assert client.get("/api/discovery/relationships", headers=AUTH, params={"source_entity_type": "tool"}).status_code == 422
    assert not sources["relationship"].calls


# --- internal proxy credential (nginx path) ----------------------------------

def test_internal_proxy_header_authenticates_only_with_the_independent_secret(client, sources, monkeypatch):
    from api import auth
    monkeypatch.setattr(auth, "_internal_proxy_secret", lambda: "synthetic-independent-proxy-secret-0123456789")
    sources["tes"].outcomes["categories"] = ["qc"]
    ok = client.get("/api/discovery/categories", headers={"X-Devhub-Internal": "synthetic-independent-proxy-secret-0123456789"})
    assert ok.status_code == 200 and ok.json() == ["qc"]
    for wrong in (JWT_SECRET, "", "synthetic-independent-proxy-secret-012345678"):
        assert client.get("/api/discovery/categories", headers={"X-Devhub-Internal": wrong}).status_code == 401


def test_tes_client_is_built_from_the_environment_per_request(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(discovery.TESDiscoveryClient, "from_environment", classmethod(lambda cls: sentinel))
    assert discovery._client() is sentinel

"""Dev Hub TES Discovery API adapter and route contract tests."""

from __future__ import annotations

from unittest.mock import patch

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.discovery import router
from integrations.tes_discovery import TESDiscoveryClient, TESDiscoveryError


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


def test_adapter_exact_lookup_and_encoded_id(monkeypatch):
    monkeypatch.setenv("TES_DISCOVERY_URL", "http://tes.test")
    response = Response({"tool_id": "ncbi_gene_search", "configuration_status": "CONFIGURED"})
    with patch("integrations.tes_discovery.requests.get", return_value=response) as get:
        result = TESDiscoveryClient.from_environment().get_tool("tool/name")
    assert result["tool_id"] == "ncbi_gene_search"
    get.assert_called_once_with("http://tes.test/api/discovery/tools/tool%2Fname", params={}, timeout=5.0)


def test_adapter_search_filters_and_facets(monkeypatch):
    monkeypatch.setenv("TES_DISCOVERY_URL", "http://tes.test")
    responses = [
        Response({"items": [{"tool_id": "bwa_mem_k8s"}], "total": 1}),
        Response({"items": [{"normalized_type": "BAM", "tool_count": 2}]}),
        Response({"items": [{"category": "Alignment & Mapping", "count": 2}]}),
        Response({"schema_version": "tes.discovery-api.v1", "canonical_tool_count": 12276}),
        Response({"state": "READY", "serving_status": None}),
    ]
    with patch("integrations.tes_discovery.requests.get", side_effect=responses) as get:
        api = TESDiscoveryClient.from_environment()
        search = api.search(q="alignment", input_type="PAIRED_END_FASTQ", limit=10, offset=20)
        assert search["items"][0]["tool_id"] == "bwa_mem_k8s"
        assert api.data_types()["items"][0]["normalized_type"] == "BAM"
        assert api.categories()["items"][0]["category"] == "Alignment & Mapping"
        assert api.catalog_version()["canonical_tool_count"] == 12276
        assert api.catalog_status()["state"] == "READY"
    assert get.call_args_list[0].kwargs["params"] == {
        "q": "alignment", "input_type": "PAIRED_END_FASTQ", "limit": 10, "offset": 20,
    }


def test_adapter_errors_are_classified(monkeypatch):
    monkeypatch.setenv("TES_DISCOVERY_URL", "http://tes.test")
    api = TESDiscoveryClient.from_environment()
    with patch("integrations.tes_discovery.requests.get", side_effect=requests.Timeout()):
        error = _raised(lambda: api.catalog_status())
        assert (error.code, error.status_code) == ("TES_TIMEOUT", 504)
    with patch("integrations.tes_discovery.requests.get", return_value=Response({}, 503)):
        error = _raised(lambda: api.catalog_status())
        assert (error.code, error.status_code) == ("TES_CATALOG_UNAVAILABLE", 503)
    with patch("integrations.tes_discovery.requests.get", return_value=Response(ValueError("bad"))):
        error = _raised(lambda: api.catalog_status())
        assert (error.code, error.status_code) == ("TES_MALFORMED_RESPONSE", 502)
    with patch("integrations.tes_discovery.requests.get", return_value=Response({}, 404)):
        error = _raised(lambda: api.get_tool("missing"))
        assert (error.code, error.status_code) == ("TOOL_NOT_FOUND", 404)


def test_unconfigured_client_is_explicit(monkeypatch):
    monkeypatch.delenv("TES_DISCOVERY_URL", raising=False)
    error = _raised(lambda: TESDiscoveryClient.from_environment())
    assert error.code == "TES_NOT_CONFIGURED"


def test_devhub_route_preserves_structured_tool_and_null_status(monkeypatch):
    monkeypatch.setenv("TES_DISCOVERY_URL", "http://tes.test")
    payload = {
        "tool_id": "bwa_mem_k8s", "display_name": "BWA", "normalized_inputs": [{"normalized_type": "BAM"}],
        "normalized_outputs": [{"normalized_type": "SAM"}], "serving_status": None,
        "registration_status": None, "tested_status": None, "operational_verification_status": None,
    }
    with patch("integrations.tes_discovery.requests.get", return_value=Response(payload)):
        response = client().get("/api/discovery/tools/bwa_mem_k8s")
    assert response.status_code == 200
    assert response.json()["tool_id"] == "bwa_mem_k8s"
    assert response.json()["serving_status"] is None
    assert response.json()["normalized_inputs"][0]["normalized_type"] == "BAM"


def test_devhub_route_errors_do_not_fallback_to_rag(monkeypatch):
    monkeypatch.delenv("TES_DISCOVERY_URL", raising=False)
    with patch("api.routes.rag.get_engine") as rag_engine:
        response = client().get("/api/discovery/search", params={"input_type": "BAM"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "TES_NOT_CONFIGURED"
    rag_engine.assert_not_called()


def test_devhub_discovery_uses_existing_auth_dependency(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")
    monkeypatch.setenv("JWT_ISSUER", "iss")
    response = client().get("/api/discovery/catalog/status")
    assert response.status_code == 401


def _raised(function):
    try:
        function()
    except TESDiscoveryError as error:
        return error
    raise AssertionError("expected TESDiscoveryError")

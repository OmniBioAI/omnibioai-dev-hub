"""Read-only adapter for the generated, source-derived API catalog."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.b1_contract import envelope


class ServiceAPIDiscoveryError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ServiceAPIDiscoveryClient:
    catalog_path: str

    @classmethod
    def from_environment(cls) -> ServiceAPIDiscoveryClient:
        path = os.environ.get("SERVICE_API_CATALOG_PATH", "").strip()
        if not path:
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_NOT_CONFIGURED", "Service/API catalog is not configured")
        return cls(path)

    def _catalog(self) -> tuple[dict[str, Any], str]:
        path = Path(self.catalog_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_UNAVAILABLE", "Service/API catalog is unavailable", 503) from exc
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_MALFORMED", "Service/API catalog is malformed", 502) from exc
        if not isinstance(body, dict) or not isinstance(body.get("services"), list) or not isinstance(body.get("routes"), list):
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "Service/API catalog schema is incompatible", 502)
        return body, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _meta(body: dict[str, Any]) -> dict[str, Any]:
        meta = body.get("_meta")
        if not isinstance(meta, dict):
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "Service/API catalog metadata is invalid", 502)
        return meta

    def _base_provenance(self, meta: dict[str, Any], catalog_hash: str, source_path: str, source_revision: Any, source_id: Any) -> dict[str, Any]:
        return {
            "source_repository": "omnibioai-docs",
            "source_namespace": "docs/api-catalog",
            "source_path": source_path,
            "source_entity_id": source_id,
            "source_revision": source_revision or "NOT_SPECIFIED",
            "catalog_hash": catalog_hash,
            "generator": meta.get("generator", "NOT_SPECIFIED"),
            "generator_version": meta.get("extractor", {}).get("version", "NOT_SPECIFIED"),
            "generated_at": meta.get("generated_at", "NOT_SPECIFIED"),
            "verification_state": meta.get("verification_state", "GENERATED_SOURCE_SNAPSHOT"),
            "evidence_links": [source_path],
        }

    def _service(self, record: dict[str, Any], meta: dict[str, Any], catalog_hash: str) -> dict[str, Any]:
        service_id = record.get("service_id")
        if not isinstance(service_id, str) or not service_id:
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "Service record lacks service_id", 502)
        payload = {k: record.get(k) for k in ("service_id", "repository", "framework", "source_derived_route_count", "published_route_count", "default_visibility", "auth_note") if k in record}
        return envelope(
            entity_type="service", source_namespace="docs/api-catalog", source_local_id=service_id,
            display_name=service_id, summary=None,
            source={"source_kind": "GENERATED_API_CATALOG", "claim_scope": ["source-derived service metadata"]},
            authority={"evidence_class": "GENERATED_PROJECTION", "authority_role": "source route catalog; not deployment truth"},
            provenance=self._base_provenance(meta, catalog_hash, "site/static/generated/api-catalog.json", record.get("source_revision"), service_id),
            freshness={"state": "SNAPSHOT", "comparison_target": "docs generated API catalog"},
            visibility={"discovery": record.get("default_visibility", "UNKNOWN"), "metadata": "PUBLIC_CATALOG", "use_permission": "SOURCE_CONTROLLED", "execution_permission": "SOURCE_CONTROLLED"},
            payload=payload,
            links=[{"kind": "source", "value": record.get("source_link")} ] if record.get("source_link") else [],
        )

    def _api(self, record: dict[str, Any], meta: dict[str, Any], catalog_hash: str) -> dict[str, Any]:
        route_id = record.get("route_id")
        service_id = record.get("service_id")
        method = record.get("method")
        path = record.get("declared_path") or record.get("effective_source_path")
        if not all(isinstance(value, str) and value for value in (route_id, service_id, method, path)):
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "API route lacks stable identity", 502)
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        source_path = source.get("source_path", "site/static/generated/api-catalog.json")
        source_revision = source.get("source_revision") or record.get("source_revision")
        payload = {k: record.get(k) for k in ("service_id", "method", "declared_path", "effective_source_path", "summary", "visibility", "auth_status", "auth_evidence", "request_body_types", "response_model", "deprecated", "gateway_path_status", "external_path_status") if k in record}
        return envelope(
            entity_type="api", source_namespace=f"service/{service_id}", source_local_id=route_id,
            display_name=f"{method} {path}", summary=record.get("summary"),
            source={"source_kind": "GENERATED_API_CATALOG", "claim_scope": ["route declaration", "catalog visibility"]},
            authority={"evidence_class": "GENERATED_PROJECTION", "authority_role": "route source reflection; not deployed/reachable truth"},
            provenance=self._base_provenance(meta, catalog_hash, source_path, source_revision, route_id),
            freshness={"state": "SNAPSHOT", "comparison_target": "docs generated API catalog"},
            visibility={"discovery": record.get("visibility", "UNKNOWN"), "metadata": record.get("visibility", "UNKNOWN"), "use_permission": "SOURCE_CONTROLLED", "execution_permission": "SOURCE_CONTROLLED"},
            lifecycle={"declared": True, "cataloged": True, "configured": "UNKNOWN", "deployed": "UNKNOWN", "reachable": "UNKNOWN", "authorized": record.get("auth_status", "UNKNOWN")},
            payload=payload,
            links=[{"kind": "source", "value": source.get("source_link")} ] if source.get("source_link") else [],
        )

    def list_services(self, query: str | None = None) -> list[dict[str, Any]]:
        body, digest = self._catalog()
        meta = self._meta(body)
        records = body["services"]
        if not all(isinstance(item, dict) for item in records):
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "Service catalog item is invalid", 502)
        return [self._service(item, meta, digest) for item in records if not query or query.lower() in str(item.get("service_id", "")).lower() or query.lower() in str(item.get("repository", "")).lower()]

    def get_service(self, service_id: str) -> dict[str, Any]:
        services = self.list_services()
        for item in services:
            if item["entity_id"]["source_local_id"] == service_id:
                return item
        raise ServiceAPIDiscoveryError("SERVICE_NOT_FOUND", "Service was not found", 404)

    def list_apis(self, service_id: str | None = None, method: str | None = None, visibility: str | None = None, route_id: str | None = None) -> list[dict[str, Any]]:
        body, digest = self._catalog()
        meta = self._meta(body)
        records = body["routes"]
        if not all(isinstance(item, dict) for item in records):
            raise ServiceAPIDiscoveryError("SERVICE_API_CATALOG_SCHEMA", "API catalog item is invalid", 502)
        return [self._api(item, meta, digest) for item in records if (not service_id or item.get("service_id") == service_id) and (not method or item.get("method") == method.upper()) and (not visibility or item.get("visibility") == visibility) and (not route_id or item.get("route_id") == route_id)]

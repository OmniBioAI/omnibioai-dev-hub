"""Read-only, source-backed relationship projections for B4."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from integrations.b1_contract import canonical_key
from integrations.service_api_discovery import (
    ServiceAPIDiscoveryClient,
    ServiceAPIDiscoveryError,
)
from integrations.tes_discovery import TESDiscoveryClient, TESDiscoveryError

RELATIONSHIP_VERSION = "1.0"
SERVICE_EXPOSES_API = "SERVICE_EXPOSES_API"
TOOL_COMPATIBLE_WITH_BACKEND = "TOOL_COMPATIBLE_WITH_BACKEND"
SUPPORTED_RELATIONSHIPS = frozenset({SERVICE_EXPOSES_API, TOOL_COMPATIBLE_WITH_BACKEND})


class RelationshipDiscoveryError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RelationshipDiscoveryClient:
    """Project only explicit relationships exposed by existing source adapters."""

    service_api: ServiceAPIDiscoveryClient | None = None
    tes: TESDiscoveryClient | None = None

    @staticmethod
    def _edge(
        *,
        source_key: str,
        relationship_type: str,
        target_key: str,
        evidence: dict[str, Any],
        provenance: dict[str, Any],
        freshness: str,
        visibility: str,
    ) -> dict[str, Any]:
        edge_seed = "|".join((source_key, relationship_type, target_key))
        return {
            "contract_version": RELATIONSHIP_VERSION,
            "relationship_id": f"relationship:{hashlib.sha256(edge_seed.encode()).hexdigest()}",
            "source_entity": source_key,
            "relationship_type": relationship_type,
            "target_entity": target_key,
            "evidence": evidence,
            "provenance": provenance,
            "freshness": {"state": freshness},
            "visibility": visibility,
            "verification_state": evidence.get("verification_state", "SOURCE_VERIFIED"),
        }

    @staticmethod
    def _error(exc: Exception) -> RelationshipDiscoveryError:
        if isinstance(exc, (ServiceAPIDiscoveryError, TESDiscoveryError)):
            return RelationshipDiscoveryError(exc.code, exc.message, exc.status_code)
        return RelationshipDiscoveryError("SOURCE_UNAVAILABLE", "Relationship source is unavailable", 503)

    def service_apis(self, service_id: str) -> list[dict[str, Any]]:
        if not service_id:
            raise RelationshipDiscoveryError("RELATIONSHIP_NOT_FOUND", "A service identity is required", 404)
        try:
            client = self.service_api or ServiceAPIDiscoveryClient.from_environment()
            service = client.get_service(service_id)
            apis = client.list_apis(service_id=service_id)
        except Exception as exc:
            raise self._error(exc) from exc
        source_key = service["canonical_key"]
        service_visibility = service.get("visibility", {}).get("discovery", "UNKNOWN")
        edges = []
        for api in apis:
            target_key = api["canonical_key"]
            api_visibility = api.get("visibility", {}).get("discovery", "UNKNOWN")
            visibility = service_visibility if service_visibility == api_visibility else "RESTRICTIVE_INTERSECTION"
            provenance = dict(api.get("provenance") or {})
            edges.append(self._edge(
                source_key=source_key,
                relationship_type=SERVICE_EXPOSES_API,
                target_key=target_key,
                evidence={
                    "classification": "EXPLICIT",
                    "source_repository": "omnibioai-docs",
                    "source_path": provenance.get("source_path", "site/static/generated/api-catalog.json"),
                    "source_field_or_rule": "routes[].service_id",
                    "source_revision": str(provenance.get("source_revision", "NOT_SPECIFIED")),
                    "verification_state": "CATALOG_SOURCE_FIELD",
                },
                provenance=provenance,
                freshness=(api.get("freshness") or {}).get("state", "SNAPSHOT"),
                visibility=visibility,
            ))
        return edges

    def tool_backends(self, tool_id: str) -> list[dict[str, Any]]:
        if not tool_id:
            raise RelationshipDiscoveryError("RELATIONSHIP_NOT_FOUND", "A tool identity is required", 404)
        try:
            client = self.tes or TESDiscoveryClient.from_environment()
            tool = client.get_tool(tool_id)
        except Exception as exc:
            raise self._error(exc) from exc
        backends = tool.get("backend_capabilities")
        source_field = "backend_capabilities"
        if backends is None:
            backends = tool.get("backends")
            source_field = "backends"
        if not isinstance(backends, list) or not all(isinstance(item, str) and item for item in backends):
            raise RelationshipDiscoveryError("UNSUPPORTED_RELATIONSHIP", "TES response has no explicit backend compatibility field", 422)
        source_key = canonical_key("tool", "tes-discovery", tool_id)
        source_revision = tool.get("source_revision") or (tool.get("provenance") or {}).get("source_revision") or "NOT_SPECIFIED"
        freshness = (tool.get("freshness") or {}).get("state", "UNKNOWN")
        visibility = (tool.get("visibility") or {}).get("discovery", "UNKNOWN")
        provenance = dict(tool.get("provenance") or {})
        edges = []
        for backend in sorted(set(backends)):
            target_key = canonical_key("backend", "tes", backend)
            edges.append(self._edge(
                source_key=source_key,
                relationship_type=TOOL_COMPATIBLE_WITH_BACKEND,
                target_key=target_key,
                evidence={
                    "classification": "EXPLICIT",
                    "source_repository": "omnibioai-tes",
                    "source_path": provenance.get("source_path", "/api/discovery/tools/{tool_id}"),
                    "source_field_or_rule": source_field,
                    "source_revision": str(source_revision),
                    "verification_state": "TES_DISCOVERY_RESPONSE",
                },
                provenance=provenance,
                freshness=freshness,
                visibility=visibility,
            ))
        return edges

    def query(
        self,
        *,
        source_entity_type: str,
        source_local_id: str,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if relationship_type and relationship_type not in SUPPORTED_RELATIONSHIPS:
            raise RelationshipDiscoveryError("UNSUPPORTED_RELATIONSHIP", "Relationship type is not supported", 422)
        if source_entity_type == "service":
            if relationship_type and relationship_type != SERVICE_EXPOSES_API:
                return []
            edges = self.service_apis(source_local_id)
        elif source_entity_type == "tool":
            if relationship_type and relationship_type != TOOL_COMPATIBLE_WITH_BACKEND:
                return []
            edges = self.tool_backends(source_local_id)
        else:
            raise RelationshipDiscoveryError("NOT_ROUTABLE_YET", "Relationship source entity is not routable in B4", 501)
        return edges

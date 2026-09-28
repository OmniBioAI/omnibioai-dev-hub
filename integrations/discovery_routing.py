"""Deterministic B3 routing over existing structured discovery adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_ENTITY_TYPES = frozenset({"documentation", "tool", "model", "service", "api", "workflow", "plugin"})
ROUTABLE_ENTITY_TYPES = frozenset({"tool", "model", "service", "api"})


class DiscoveryRoutingError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class RoutingDecision:
    requested_entity_type: str
    selected_source: str
    authority_class: str
    routing_reason: str
    fallback_policy: str
    execution_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_entity_type": self.requested_entity_type,
            "selected_source": self.selected_source,
            "authority_class": self.authority_class,
            "routing_reason": self.routing_reason,
            "fallback_policy": self.fallback_policy,
            "execution_allowed": self.execution_allowed,
        }


def route_query(entity_type: str | None) -> RoutingDecision:
    """Select a source from explicit entity type only; never classify semantically."""
    normalized = (entity_type or "").strip().lower().replace("_", "-")
    aliases = {"docs": "documentation", "service-api": "service", "services": "service", "apis": "api", "tools": "tool", "models": "model", "workflows": "workflow", "plugins": "plugin"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_ENTITY_TYPES:
        raise DiscoveryRoutingError("UNKNOWN_STRUCTURED_INTENT", "An explicit supported entity type is required")
    if normalized == "documentation":
        return RoutingDecision(normalized, "devhub-documentation-rag", "DOCUMENTATION", "explicit documentation mode", "no_structured_fallback")
    if normalized == "tool":
        return RoutingDecision(normalized, "tes-discovery", "SOURCE_DECLARATION", "explicit tool mode", "no_rag_fallback")
    if normalized == "model":
        return RoutingDecision(normalized, "model-registry", "IMPLEMENTATION", "explicit model mode", "no_rag_fallback")
    if normalized in {"service", "api"}:
        return RoutingDecision(normalized, "docs-api-catalog", "GENERATED_PROJECTION", "explicit service/API mode", "no_rag_fallback")
    raise DiscoveryRoutingError(f"{normalized.upper()}_NOT_ROUTABLE_YET", f"{normalized.title()} structured discovery is not routable in B3", 501)

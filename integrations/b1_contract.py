"""Small source-neutral helpers for the B1 structured-discovery envelope."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

CONTRACT_VERSION = "1.0"


def _local_id(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def canonical_key(entity_type: str, source_namespace: str, source_local_id: Any) -> str:
    """Return the B1 escaped display key while retaining structured IDs."""
    return ":".join(
        (
            entity_type,
            quote(source_namespace, safe=""),
            quote(_local_id(source_local_id), safe=""),
        )
    )


def envelope(
    *,
    entity_type: str,
    source_namespace: str,
    source_local_id: Any,
    display_name: str,
    summary: str | None,
    source: dict[str, Any],
    authority: dict[str, Any],
    provenance: dict[str, Any],
    freshness: dict[str, Any],
    visibility: dict[str, Any],
    payload: dict[str, Any],
    lifecycle: dict[str, Any] | None = None,
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "entity_type": entity_type,
        "entity_id": {
            "source_namespace": source_namespace,
            "source_local_id": source_local_id,
        },
        "canonical_key": canonical_key(entity_type, source_namespace, source_local_id),
        "display": {"name": display_name, "summary": summary},
        "source": source,
        "authority": authority,
        "provenance": provenance,
        "freshness": freshness,
        "visibility": visibility,
        "lifecycle": lifecycle or {},
        "relationships": [],
        "links": links or [],
        "payload": payload,
    }

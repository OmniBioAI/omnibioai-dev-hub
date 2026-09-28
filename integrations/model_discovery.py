"""Read-only, server-side adapter for the Model Registry metadata API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from integrations.b1_contract import envelope


class ModelDiscoveryError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ModelDiscoveryClient:
    base_url: str
    timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> "ModelDiscoveryClient":
        base_url = os.environ.get("MODEL_REGISTRY_URL", "").strip().rstrip("/")
        if not base_url:
            raise ModelDiscoveryError("MODEL_REGISTRY_NOT_CONFIGURED", "Model Registry discovery is not configured")
        try:
            timeout = float(os.environ.get("MODEL_REGISTRY_TIMEOUT_SECONDS", "5"))
        except ValueError:
            timeout = 5.0
        return cls(base_url=base_url, timeout_seconds=max(0.1, min(timeout, 30.0)))

    def _get(self, path: str, authorization: str | None = None, params: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": authorization} if authorization else {}
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params={k: v for k, v in (params or {}).items() if v not in (None, "")},
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise ModelDiscoveryError("MODEL_REGISTRY_TIMEOUT", "Model Registry discovery timed out", 504) from exc
        except requests.RequestException as exc:
            raise ModelDiscoveryError("MODEL_REGISTRY_UNAVAILABLE", "Model Registry discovery is unavailable", 503) from exc
        if response.status_code in (401, 403):
            code = "MODEL_REGISTRY_UNAUTHORIZED" if response.status_code == 401 else "MODEL_REGISTRY_FORBIDDEN"
            raise ModelDiscoveryError(code, "Model Registry authorization failed", response.status_code)
        if response.status_code == 404:
            raise ModelDiscoveryError("MODEL_NOT_FOUND", "Model was not found", 404)
        if response.status_code >= 400:
            raise ModelDiscoveryError("MODEL_REGISTRY_ERROR", "Model Registry request failed", 502)
        try:
            body = response.json()
        except ValueError as exc:
            raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model Registry returned malformed metadata", 502) from exc
        if not isinstance(body, (dict, list)):
            raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model Registry returned invalid metadata", 502)
        return body

    @staticmethod
    def _safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
        """Keep discovery metadata; never return paths, credentials, or package contents."""
        allowed = {
            "task", "model_name", "version", "name", "display_name", "description",
            "framework", "feature_schema", "schema_version", "sha256", "sha256sums",
            "integrity", "promotion", "stage", "aliases", "metrics", "tags",
            "ownership_status", "visibility", "source_revision", "verification_state",
        }
        return {key: value for key, value in meta.items() if key in allowed}

    def _envelope(self, meta: dict[str, Any], *, authorization: str | None, source_endpoint: str) -> dict[str, Any]:
        task = meta.get("task")
        model_name = meta.get("model_name") or meta.get("name")
        version = meta.get("version")
        if not task or not model_name:
            raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model metadata lacks task/model identity", 502)
        local_id = {"task": task, "model_name": model_name, "version": version}
        display = str(meta.get("display_name") or model_name)
        return envelope(
            entity_type="model",
            source_namespace="model-registry",
            source_local_id=local_id,
            display_name=display,
            summary=meta.get("description"),
            source={"source_kind": "MODEL_REGISTRY_API", "claim_scope": ["model metadata", "registry state"]},
            authority={"evidence_class": "IMPLEMENTATION", "authority_role": "Model Registry"},
            provenance={
                "source_repository": "omnibioai-model-registry",
                "source_namespace": "model-registry",
                "source_path": source_endpoint,
                "source_entity_id": local_id,
                "source_revision": meta.get("source_revision", "NOT_SPECIFIED"),
                "verification_state": meta.get("verification_state", "AUTHENTICATED_SOURCE"),
                "evidence_links": [source_endpoint],
            },
            freshness={"state": "LIVE", "comparison_target": "Model Registry API response"},
            visibility={
                "discovery": meta.get("visibility", "ORG_SCOPED"),
                "metadata": "MODEL_READ",
                "use_permission": "MODEL_USE",
                "execution_permission": "MODEL_USE",
                "administrative_permission": "MODEL_ADMINISTRATION",
            },
            lifecycle={"promotion": meta.get("promotion", meta.get("stage", "NOT_SPECIFIED"))},
            links=[{"kind": "source", "value": source_endpoint}],
            payload=self._safe_meta(meta),
        )

    def list_models(self, *, authorization: str | None, task: str | None = None, model_name: str | None = None, version: str | None = None) -> list[dict[str, Any]]:
        body = self._get("/v1/models", authorization, {"task": task, "model_name": model_name})
        if not isinstance(body, list):
            raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model Registry model list is invalid", 502)
        result = []
        for meta in body:
            if not isinstance(meta, dict):
                raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model Registry model item is invalid", 502)
            if version and meta.get("version") != version:
                continue
            result.append(self._envelope(meta, authorization=authorization, source_endpoint="/v1/models"))
        return result

    def get_model(self, task: str, model_name: str, *, version: str | None = None, alias: str | None = None, authorization: str | None) -> dict[str, Any]:
        ref = version or alias or "latest"
        body = self._get(
            "/v1/show",
            authorization,
            {"task": task, "ref": ref, "verify": "false"},
        )
        if not isinstance(body, dict) or not isinstance(body.get("meta"), dict):
            raise ModelDiscoveryError("MODEL_REGISTRY_MALFORMED_RESPONSE", "Model Registry model response is invalid", 502)
        return self._envelope(body["meta"], authorization=authorization, source_endpoint="/v1/show")

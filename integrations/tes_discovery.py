"""Small read-only client for the TES Discovery API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


class TESDiscoveryError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class TESDiscoveryClient:
    base_url: str
    timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> TESDiscoveryClient:
        base_url = os.environ.get("TES_DISCOVERY_URL", "").strip().rstrip("/")
        if not base_url:
            raise TESDiscoveryError("TES_NOT_CONFIGURED", "TES discovery is not configured")
        try:
            timeout = float(os.environ.get("TES_DISCOVERY_TIMEOUT_SECONDS", "5"))
        except ValueError:
            timeout = 5.0
        return cls(base_url=base_url, timeout_seconds=max(0.1, min(timeout, 30.0)))

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params={key: value for key, value in (params or {}).items() if value not in (None, "")},
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise TESDiscoveryError("TES_TIMEOUT", "TES discovery request timed out", 504) from exc
        except requests.RequestException as exc:
            raise TESDiscoveryError("TES_UNAVAILABLE", "TES discovery is unavailable", 503) from exc

        if response.status_code == 404:
            raise TESDiscoveryError("TOOL_NOT_FOUND", "TES tool was not found", 404)
        if response.status_code == 503:
            raise TESDiscoveryError("TES_CATALOG_UNAVAILABLE", "TES discovery catalog is unavailable", 503)
        if response.status_code >= 400:
            raise TESDiscoveryError("TES_ERROR", "TES discovery request failed", 502)
        try:
            body = response.json()
        except ValueError as exc:
            raise TESDiscoveryError("TES_MALFORMED_RESPONSE", "TES returned malformed discovery data", 502) from exc
        if not isinstance(body, (dict, list)):
            raise TESDiscoveryError("TES_MALFORMED_RESPONSE", "TES returned an invalid discovery response", 502)
        return body

    @staticmethod
    def _object(body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise TESDiscoveryError("TES_MALFORMED_RESPONSE", "TES returned an invalid discovery object", 502)
        return body

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        return self._object(self._get(f"/api/discovery/tools/{quote(tool_id, safe='')}"))

    def search(self, **filters: Any) -> dict[str, Any]:
        body = self._object(self._get("/api/discovery/search", filters))
        if not isinstance(body.get("items"), list) or not isinstance(body.get("total"), int):
            raise TESDiscoveryError("TES_MALFORMED_RESPONSE", "TES returned an invalid discovery search", 502)
        return body

    def data_types(self) -> dict[str, Any]:
        return self._facet("/api/discovery/data-types")

    def categories(self) -> dict[str, Any]:
        return self._facet("/api/discovery/categories")

    def _facet(self, path: str) -> dict[str, Any]:
        body = self._object(self._get(path))
        if not isinstance(body.get("items"), list):
            raise TESDiscoveryError("TES_MALFORMED_RESPONSE", "TES returned an invalid discovery facet", 502)
        return body

    def catalog_version(self) -> dict[str, Any]:
        return self._object(self._get("/api/discovery/catalog/version"))

    def catalog_status(self) -> dict[str, Any]:
        return self._object(self._get("/api/discovery/catalog/status"))

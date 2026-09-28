from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_auth
from integrations.tes_discovery import TESDiscoveryClient, TESDiscoveryError

router = APIRouter(prefix="/api/discovery", tags=["tool-discovery"])


def _client() -> TESDiscoveryClient:
    return TESDiscoveryClient.from_environment()


def _call(method_name: str, *args, **kwargs):
    try:
        return getattr(_client(), method_name)(*args, **kwargs)
    except TESDiscoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


@router.get("/tools/{tool_id}")
def get_tool(tool_id: str, actor: str = Depends(require_auth)):
    return _call("get_tool", tool_id)


@router.get("/search")
def search_tools(
    q: str | None = None,
    input_type: str | None = None,
    output_type: str | None = None,
    category: str | None = None,
    backend: str | None = None,
    architecture: str | None = None,
    tool_id: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    actor: str = Depends(require_auth),
):
    return _call(
        "search",
        q=q, input_type=input_type, output_type=output_type, category=category,
        backend=backend, architecture=architecture, tool_id=tool_id,
        limit=limit, offset=offset,
    )


@router.get("/data-types")
def data_types(actor: str = Depends(require_auth)):
    return _call("data_types")


@router.get("/categories")
def categories(actor: str = Depends(require_auth)):
    return _call("categories")


@router.get("/catalog/version")
def catalog_version(actor: str = Depends(require_auth)):
    return _call("catalog_version")


@router.get("/catalog/status")
def catalog_status(actor: str = Depends(require_auth)):
    return _call("catalog_status")

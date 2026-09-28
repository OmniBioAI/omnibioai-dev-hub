from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from api.auth import require_auth
from integrations.tes_discovery import TESDiscoveryClient, TESDiscoveryError
from integrations.model_discovery import ModelDiscoveryClient, ModelDiscoveryError
from integrations.service_api_discovery import ServiceAPIDiscoveryClient, ServiceAPIDiscoveryError
from integrations.discovery_routing import DiscoveryRoutingError, route_query

router = APIRouter(prefix="/api/discovery", tags=["tool-discovery"])


def _client() -> TESDiscoveryClient:
    return TESDiscoveryClient.from_environment()


def _call(method_name: str, *args, **kwargs):
    try:
        return getattr(_client(), method_name)(*args, **kwargs)
    except TESDiscoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


def _model_call(method_name: str, *args, **kwargs):
    try:
        return getattr(ModelDiscoveryClient.from_environment(), method_name)(*args, **kwargs)
    except ModelDiscoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


def _service_api_call(method_name: str, *args, **kwargs):
    try:
        return getattr(ServiceAPIDiscoveryClient.from_environment(), method_name)(*args, **kwargs)
    except ServiceAPIDiscoveryError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


def _routing_error(exc: DiscoveryRoutingError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message})


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


@router.get("/models")
def list_models(
    task: str | None = None,
    model_name: str | None = None,
    version: str | None = None,
    authorization: str | None = Header(default=None),
    actor: str = Depends(require_auth),
):
    """Read-only metadata listing; Model Registry remains the auth authority."""
    return {"items": _model_call("list_models", authorization=authorization, task=task, model_name=model_name, version=version)}


@router.get("/models/{task}/{model_name}")
def get_model(
    task: str,
    model_name: str,
    version: str | None = None,
    alias: str | None = None,
    authorization: str | None = Header(default=None),
    actor: str = Depends(require_auth),
):
    if version and alias:
        raise HTTPException(status_code=400, detail={"code": "MODEL_ID_AMBIGUOUS", "message": "Specify version or alias, not both"})
    return _model_call("get_model", task, model_name, version=version, alias=alias, authorization=authorization)


@router.get("/services")
def list_services(query: str | None = None, actor: str = Depends(require_auth)):
    return {"items": _service_api_call("list_services", query=query)}


@router.get("/services/{service_id}")
def get_service(service_id: str, actor: str = Depends(require_auth)):
    return _service_api_call("get_service", service_id)


@router.get("/services/{service_id}/apis")
def list_service_apis(
    service_id: str,
    method: str | None = None,
    visibility: str | None = None,
    actor: str = Depends(require_auth),
):
    return {"items": _service_api_call("list_apis", service_id=service_id, method=method, visibility=visibility)}


@router.get("/apis")
def list_apis(
    service_id: str | None = None,
    method: str | None = None,
    visibility: str | None = None,
    route_id: str | None = None,
    actor: str = Depends(require_auth),
):
    return {"items": _service_api_call("list_apis", service_id=service_id, method=method, visibility=visibility, route_id=route_id)}


@router.get("/query")
def structured_query(
    entity_type: str,
    q: str | None = None,
    task: str | None = None,
    model_name: str | None = None,
    version: str | None = None,
    alias: str | None = None,
    service_id: str | None = None,
    method: str | None = None,
    visibility: str | None = None,
    route_id: str | None = None,
    tool_id: str | None = None,
    authorization: str | None = Header(default=None),
    actor: str = Depends(require_auth),
):
    """Route explicit structured intent without semantic or execution fallback."""
    try:
        decision = route_query(entity_type)
    except DiscoveryRoutingError as exc:
        raise _routing_error(exc) from exc

    if decision.requested_entity_type == "documentation":
        return {"routing": decision.as_dict(), "next_route": "/rag/query", "query": q}
    if decision.requested_entity_type == "tool":
        return {"routing": decision.as_dict(), "result": _call("search", q=q, tool_id=tool_id)}
    if decision.requested_entity_type == "model":
        return {"routing": decision.as_dict(), "result": _model_call("list_models", authorization=authorization, task=task, model_name=model_name, version=version)}
    if decision.requested_entity_type == "service":
        return {"routing": decision.as_dict(), "result": _service_api_call("list_services", query=q or service_id)}
    return {"routing": decision.as_dict(), "result": _service_api_call("list_apis", service_id=service_id, method=method, visibility=visibility, route_id=route_id)}

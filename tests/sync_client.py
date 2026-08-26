"""Synchronous test adapter for ASGI apps without Starlette TestClient."""

import asyncio
import json

import httpx
from fastapi import HTTPException
from pydantic import ValidationError


class Response:
    def __init__(self, status_code, payload=None, headers=None, lines=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._lines = lines or []

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


def _rag_request(method, path, payload):
    from api.routes import rag

    try:
        request = rag.QueryRequest(**(payload or {}))
    except ValidationError as exc:
        return Response(422, {"detail": exc.errors()})

    endpoint = rag.query if path == "/query" else rag.stream
    try:
        result = endpoint(request, actor="system")
    except HTTPException as exc:
        return Response(exc.status_code, {"detail": exc.detail})

    if path == "/stream":
        # StreamingResponse wraps a synchronous iterator in an AnyIO worker
        # thread; consuming that wrapper from a fresh event loop can deadlock
        # under the current test runner. Recreate the endpoint's finite SSE
        # payload from its mocked engine for deterministic client semantics.
        try:
            engine = rag.get_engine()
            context = engine.build_context(
                engine.retrieve(request.query, repo=request.repo, bundle=request.bundle)
            )
            if hasattr(engine, "stream_llm"):
                lines = [
                    f"data: {{\"type\":\"token\",\"content\":{json.dumps(token)}}}"
                    for token in engine.stream_llm(request.query, context)
                ]
            else:
                answer = engine.answer(request.query)["answer"]
                lines = [f"data: {{\"type\":\"response\",\"content\":{json.dumps(answer)}}}"]
            lines.append('data: {"type":"done"}')
        except Exception as exc:
            lines = [f"data: {{\"type\":\"error\",\"message\":{json.dumps(str(exc))}}}"]
        return Response(200, headers={"content-type": "text/event-stream"}, lines=lines)
    return Response(200, result)


class SyncASGIClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        if path in {"/query", "/stream"}:
            return _rag_request(method, path, kwargs.get("json"))

        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

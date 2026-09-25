"""Focused branch coverage for the HTTP auth and RAG route helpers."""

import asyncio
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

from api import auth
from api.routes import rag


def test_auth_token_helpers_cover_success_and_failures(monkeypatch):
    assert auth.extract_token("Bearer abc") == "abc"
    with pytest.raises(auth.AuthError, match="missing"):
        auth.extract_token(None)
    with pytest.raises(auth.AuthError, match="Bearer"):
        auth.extract_token("Token abc")

    token = jwt.encode({"sub": "user-1"}, "secret", algorithm="HS256")
    assert auth.validate_token(token, "secret")["sub"] == "user-1"
    with pytest.raises(auth.AuthError, match="Invalid token"):
        auth.validate_token("bad", "secret")

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    assert auth._auth_enabled() is True
    assert auth._jwt_secret() == "secret"


@pytest.mark.asyncio
async def test_require_auth_all_modes_and_identity_fields(monkeypatch):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    assert await auth.require_auth(SimpleNamespace()) == "system"

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "secret")
    assert await auth.require_auth(SimpleNamespace(), x_devhub_internal="secret") == "devhub-ui"

    token = jwt.encode({"email": "user@example.com"}, "secret", algorithm="HS256")
    assert await auth.require_auth(SimpleNamespace(), authorization=f"Bearer {token}") == "user@example.com"

    unknown = jwt.encode({}, "secret", algorithm="HS256")
    assert await auth.require_auth(SimpleNamespace(), authorization=f"Bearer {unknown}") == "unknown"

    with pytest.raises(HTTPException, match="Authorization header"):
        await auth.require_auth(SimpleNamespace(), authorization="bad")


def test_rag_stream_success_fallback_and_error(monkeypatch):
    captured = {}

    class FakeStreamingResponse:
        def __init__(self, content, media_type):
            captured["content"] = content
            captured["media_type"] = media_type
            self.content = content
            self.media_type = media_type

    monkeypatch.setattr(rag, "StreamingResponse", FakeStreamingResponse)
    req = rag.QueryRequest(query="hello")

    engine = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [{"text": "ctx"}],
        build_context=lambda docs: "context",
        stream_llm=lambda query, context: ["one", "two"],
    )
    monkeypatch.setattr(rag, "get_engine", lambda: engine)
    response = rag.stream(req, actor="system")
    assert response.media_type == "text/event-stream"
    events = list(response.content)
    assert '"type": "token"' in events[0]
    assert '"type": "done"' in events[-1]

    fallback = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [],
        build_context=lambda docs: "",
        answer=lambda query: {"answer": "single"},
    )
    monkeypatch.setattr(rag, "get_engine", lambda: fallback)
    response = rag.stream(req, actor="system")
    events = list(response.content)
    assert '"type": "response"' in events[0]

    monkeypatch.setattr(rag, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    response = rag.stream(req, actor="system")
    assert '"type": "error"' in list(response.content)[0]

"""Focused branch coverage for the HTTP auth and RAG route helpers."""

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


def test_rag_stream_success_and_error(monkeypatch):
    class FakeStreamingResponse:
        def __init__(self, content, media_type):
            self.content = content
            self.media_type = media_type

    monkeypatch.setattr(rag, "StreamingResponse", FakeStreamingResponse)
    req = rag.QueryRequest(query="hello")

    engine = SimpleNamespace(
        retrieve=lambda *args, **kwargs: [{"text": "ctx"}],
        answer_from_docs=lambda query, docs: {
            "answer": "single", "grounded": True, "answer_status": "ANSWERED", "answer_contract": "v1",
            "citations": [], "context_used": len(docs), "llm_invoked": True,
        },
    )
    monkeypatch.setattr(rag, "get_engine", lambda: engine)
    response = rag.stream(req, actor="system")
    assert response.media_type == "text/event-stream"
    events = list(response.content)
    assert '"type": "status"' in events[0]
    assert '"type": "response"' in events[1] and '"content": "single"' in events[1]
    assert '"type": "done"' in events[-1]

    monkeypatch.setattr(rag, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    response = rag.stream(req, actor="system")
    events = list(response.content)
    assert '"code": "internal_error"' in events[0] and "boom" not in events[0]

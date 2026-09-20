"""Tests for the RAG API routes (api.routes.rag) mounted on a throwaway FastAPI app with the control
plane mocked: engine access, query, streaming, error reporting, and repo/bundle scoping.

Developer: Manish Kumar <manish@omnibioai.org>
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import the router and models from the target file
from api.routes.rag import get_engine, router

# Create a dummy app to test the router
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    """Provide a TestClient for a throwaway app that mounts only the RAG router."""
    return TestClient(app)

@pytest.fixture
def mock_control_plane():
    """Patch the routes module's CONTROL_PLANE with a mock so no real engine is built."""
    with patch("api.routes.rag.CONTROL_PLANE") as mock:
        yield mock

# =========================================================
# UNIT TESTS FOR get_engine()
# =========================================================

def test_get_engine_success(mock_control_plane):
    """Return the control plane's engine when it exposes a query method."""
    mock_engine = MagicMock()
    # Mock hasattr to return True for "query"
    mock_engine.query = MagicMock()
    mock_control_plane.get_engine.return_value = mock_engine
    
    engine = get_engine()
    assert engine == mock_engine

def test_get_engine_none(mock_control_plane):
    """Raise 'RAG engine not initialized' when the control plane has no engine."""
    mock_control_plane.get_engine.return_value = None
    with pytest.raises(RuntimeError, match="RAG engine not initialized"):
        get_engine()

def test_get_engine_missing_query(mock_control_plane):
    """Raise an error when the engine lacks the V6 query method."""
    mock_engine = MagicMock(spec=[]) # No query method
    mock_control_plane.get_engine.return_value = mock_engine
    with pytest.raises(RuntimeError, match="Engine missing V6 query method"):
        get_engine()

def test_get_engine_exception(mock_control_plane):
    """Wrap an exception raised while fetching the engine in a RuntimeError that carries its
    message."""
    mock_control_plane.get_engine.side_effect = Exception("Internal Error")
    with pytest.raises(RuntimeError, match="Engine access failed: Internal Error"):
        get_engine()

# =========================================================
# ENDPOINT TESTS: /query
# =========================================================

RESULT = {
    "query": "hello", "answer": "test answer [1]", "grounded": True, "answer_status": "GROUNDED",
    "answer_contract": "ask.v1", "citations": [{"index": 1}], "context_used": 1, "llm_invoked": True,
    "sources": ["s"], "context": [{"text": "c"}], "version": "v6-faiss",
}


def _engine(retrieved=None, result=None):
    engine = MagicMock()
    engine.retrieve.return_value = [{"text": "c"}] if retrieved is None else retrieved
    engine.answer_from_docs.return_value = RESULT if result is None else result
    return engine


def test_query_endpoint_success(client, mock_control_plane):
    """Retrieve under the server-side policy, answer through the contract, add the v6 API version."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "test answer [1]" and data["grounded"] is True and data["api_version"] == "v6"
    engine.retrieve.assert_called_once_with("hello", repo=None, bundle=None,
                                            allowed_visibilities={"PUBLIC"}, min_relevance=0.64)
    engine.answer_from_docs.assert_called_once_with("hello", [{"text": "c"}])


def test_query_endpoint_failure_traceback_enabled(client, mock_control_plane, monkeypatch):
    """Return a 500 with the error message and a traceback when DEBUG_TRACEBACKS is enabled."""
    monkeypatch.setenv("DEBUG_TRACEBACKS", "true")
    engine = _engine()
    engine.retrieve.side_effect = Exception("Query Failed")
    mock_control_plane.get_engine.return_value = engine

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    data = response.json()
    assert data["detail"]["error"] == "Query Failed"
    assert "trace" in data["detail"]


def test_query_endpoint_failure_traceback_disabled(client, mock_control_plane, monkeypatch):
    """Return a 500 with the error message but no traceback when DEBUG_TRACEBACKS is unset."""
    monkeypatch.delenv("DEBUG_TRACEBACKS", raising=False)
    engine = _engine()
    engine.retrieve.side_effect = Exception("Query Failed")
    mock_control_plane.get_engine.return_value = engine

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Query Failed"
    assert "trace" not in response.json()["detail"]


# =========================================================
# ENDPOINT TESTS: /stream
# =========================================================

def _events(response):
    return [json.loads(line.replace("data: ", "")) for line in response.iter_lines() if line]


def test_stream_endpoint_emits_status_then_verified_response_then_done(client, mock_control_plane):
    """The stream sends the VERIFIED answer (never raw tokens): status, response, done."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine

    response = client.post("/stream", json={"query": "hello"})

    assert response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
    events = _events(response)
    assert [e["type"] for e in events] == ["status", "response", "done"]
    assert events[0] == {"type": "status", "stage": "retrieved", "context_used": 1}
    assert events[1]["content"] == "test answer [1]" and events[1]["grounded"] is True
    assert events[1]["answer_status"] == "GROUNDED" and events[1]["citations"] == [{"index": 1}]
    assert not any(e["type"] == "token" for e in events)
    engine.stream_llm.assert_not_called()  # the raw-token path is not used by the API


def test_stream_endpoint_error(client, mock_control_plane):
    """Report an engine failure as a single error event on the stream instead of an HTTP error."""
    with patch("api.routes.rag.get_engine") as mock_get_engine:
        mock_get_engine.side_effect = Exception("Stream Init Error")

        response = client.post("/stream", json={"query": "hello"})

        assert response.status_code == 200
        assert _events(response) == [{"type": "error", "message": "Stream Init Error"}]


# =========================================================
# EDGE CASES
# =========================================================

def test_query_request_validation(client):
    """Reject a /query request that omits the required query field with a 422."""
    assert client.post("/query", json={}).status_code == 422


# =========================================================
# SCOPED QUERY TESTS (repo / bundle params)
# =========================================================

def test_query_endpoint_with_bundle_scope(client, mock_control_plane):
    """Pass the requested bundle scope through to retrieval."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine
    client.post("/query", json={"query": "metagenomics", "bundle": "metagenomics"})
    engine.retrieve.assert_called_once_with("metagenomics", repo=None, bundle="metagenomics",
                                            allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


def test_query_endpoint_with_repo_scope(client, mock_control_plane):
    """Pass the requested repo scope through to retrieval."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine
    client.post("/query", json={"query": "model versioning", "repo": "omnibioai-model-registry"})
    engine.retrieve.assert_called_once_with("model versioning", repo="omnibioai-model-registry", bundle=None,
                                            allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


def test_query_endpoint_unscoped_passes_none_filters(client, mock_control_plane):
    """Pass None for both repo and bundle when the request is unscoped."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine
    client.post("/query", json={"query": "hello"})
    engine.retrieve.assert_called_once_with("hello", repo=None, bundle=None,
                                            allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


def test_stream_endpoint_with_bundle_scope(client, mock_control_plane):
    """Pass the requested bundle scope through to retrieval when streaming."""
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine
    response = client.post("/stream", json={"query": "q", "bundle": "metagenomics"})
    assert response.status_code == 200
    engine.retrieve.assert_called_once_with("q", repo=None, bundle="metagenomics",
                                            allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


# ---- PUBLIC-only + relevance cutoff are server-side policy, not client input ----

def test_client_cannot_widen_visibility_or_lower_relevance_via_request_body(client, mock_control_plane):
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine

    client.post("/query", json={
        "query": "q", "allowed_visibilities": ["PUBLIC", "INTERNAL", "REVIEW_REQUIRED"],
        "min_relevance": 0.0, "visibility": "INTERNAL",
    })

    engine.retrieve.assert_called_once_with("q", repo=None, bundle=None, allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


def test_stream_also_ignores_client_policy_fields(client, mock_control_plane):
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine

    client.post("/stream", json={"query": "q", "allowed_visibilities": ["INTERNAL"], "min_relevance": 0})

    engine.retrieve.assert_called_once_with("q", repo=None, bundle=None, allowed_visibilities={"PUBLIC"}, min_relevance=0.64)


@pytest.mark.parametrize("env,expected", [("", 0.64), ("0.7", 0.7), ("0", 0.0), ("abc", 0.64), ("1.5", 0.64), ("-1", 0.64)])
def test_min_relevance_env_override_and_fail_safe(monkeypatch, env, expected):
    from api.routes.rag import _min_relevance
    monkeypatch.setenv("DEVHUB_MIN_RELEVANCE", env)
    assert _min_relevance() == expected


# ---- malformed / empty queries are rejected before anything runs ----

@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   \n\t "}, {"query": None}, {"query": 123}, {"query": ["a"]},
    {"query": "x" * 2001}, {"query": "bad\u0000byte"}, {"query": "ok", "repo": "r" * 201},
])
@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_malformed_or_empty_queries_get_422_and_touch_nothing(client, mock_control_plane, path, payload):
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine

    response = client.post(path, json=payload)

    assert response.status_code == 422
    engine.retrieve.assert_not_called()
    engine.answer_from_docs.assert_not_called()


def test_query_text_is_stripped_before_use(client, mock_control_plane):
    engine = _engine()
    mock_control_plane.get_engine.return_value = engine
    client.post("/query", json={"query": "   hello   "})
    assert engine.retrieve.call_args.args[0] == "hello"

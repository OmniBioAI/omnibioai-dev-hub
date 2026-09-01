import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json

# Import the router and models from the target file
from api.routes.rag import router, get_engine

# Create a dummy app to test the router
app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_control_plane():
    with patch("api.routes.rag.CONTROL_PLANE") as mock:
        yield mock

# =========================================================
# UNIT TESTS FOR get_engine()
# =========================================================

def test_get_engine_success(mock_control_plane):
    mock_engine = MagicMock()
    # Mock hasattr to return True for "query"
    mock_engine.query = MagicMock()
    mock_control_plane.get_engine.return_value = mock_engine
    
    engine = get_engine()
    assert engine == mock_engine

def test_get_engine_none(mock_control_plane):
    mock_control_plane.get_engine.return_value = None
    with pytest.raises(RuntimeError, match="RAG engine not initialized"):
        get_engine()

def test_get_engine_missing_query(mock_control_plane):
    mock_engine = MagicMock(spec=[]) # No query method
    mock_control_plane.get_engine.return_value = mock_engine
    with pytest.raises(RuntimeError, match="Engine missing V6 query method"):
        get_engine()

def test_get_engine_exception(mock_control_plane):
    mock_control_plane.get_engine.side_effect = Exception("Internal Error")
    with pytest.raises(RuntimeError, match="Engine access failed: Internal Error"):
        get_engine()


# =========================================================
# ENDPOINT TESTS: /query
# =========================================================

def test_query_endpoint_success(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.query.return_value = {"answer": "test answer"}
    mock_control_plane.get_engine.return_value = mock_engine
    
    response = client.post("/query", json={"query": "hello"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "test answer"
    assert data["api_version"] == "v6"

def test_query_endpoint_failure_traceback_enabled(client, mock_control_plane, monkeypatch):
    monkeypatch.setenv("DEBUG_TRACEBACKS", "true")
    mock_engine = MagicMock()
    mock_engine.query.side_effect = Exception("Query Failed")
    mock_control_plane.get_engine.return_value = mock_engine

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "Query Failed"
    assert "trace" in data["detail"]

def test_query_endpoint_failure_traceback_disabled(client, mock_control_plane, monkeypatch):
    monkeypatch.delenv("DEBUG_TRACEBACKS", raising=False)
    mock_engine = MagicMock()
    mock_engine.query.side_effect = Exception("Query Failed")
    mock_control_plane.get_engine.return_value = mock_engine

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "Query Failed"
    assert "trace" not in data["detail"]


# =========================================================
# ENDPOINT TESTS: /stream
# =========================================================

def test_stream_endpoint_with_llm_streaming(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = [{"text": "context"}]
    mock_engine.build_context.return_value = "built context"
    
    # Mock stream_llm which is an iterable
    mock_engine.stream_llm.return_value = ["token1", " ", "token2"]
    
    mock_control_plane.get_engine.return_value = mock_engine
    
    response = client.post("/stream", json={"query": "hello"})
    
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    lines = [line for line in response.iter_lines() if line]
    assert len(lines) == 4 # 3 tokens + 1 done
    assert json.loads(lines[0].replace("data: ", "")) == {"type": "token", "content": "token1"}
    assert json.loads(lines[1].replace("data: ", "")) == {"type": "token", "content": " "}
    assert json.loads(lines[2].replace("data: ", "")) == {"type": "token", "content": "token2"}
    assert json.loads(lines[3].replace("data: ", "")) == {"type": "done"}

def test_stream_endpoint_fallback_single_response(client, mock_control_plane):
    mock_engine = MagicMock()
    # Explicitly remove stream_llm to trigger fallback
    del mock_engine.stream_llm
    
    mock_engine.retrieve.return_value = [{"text": "context"}]
    mock_engine.build_context.return_value = "built context"
    mock_engine.answer.return_value = {"answer": "single answer"}
    
    mock_control_plane.get_engine.return_value = mock_engine
    
    response = client.post("/stream", json={"query": "hello"})
    
    assert response.status_code == 200
    lines = [line for line in response.iter_lines() if line]
    assert len(lines) == 2 # 1 response + 1 done
    assert json.loads(lines[0].replace("data: ", "")) == {"type": "response", "content": "single answer"}
    assert json.loads(lines[1].replace("data: ", "")) == {"type": "done"}

def test_stream_endpoint_error(client, mock_control_plane):
    with patch("api.routes.rag.get_engine") as mock_get_engine:
        mock_get_engine.side_effect = Exception("Stream Init Error")
        
        response = client.post("/stream", json={"query": "hello"})
        
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]
        assert len(lines) == 1
        assert json.loads(lines[0].replace("data: ", "")) == {"type": "error", "message": "Stream Init Error"}

# =========================================================
# EDGE CASES
# =========================================================

def test_query_request_validation(client):
    # Test Pydantic validation
    response = client.post("/query", json={}) # Missing 'query' field
    assert response.status_code == 422


# =========================================================
# SCOPED QUERY TESTS (repo / bundle params)
# =========================================================

def test_query_endpoint_with_bundle_scope(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.query.return_value = {"answer": "scoped answer"}
    mock_control_plane.get_engine.return_value = mock_engine

    response = client.post("/query", json={"query": "metagenomics", "bundle": "metagenomics"})

    assert response.status_code == 200
    mock_engine.query.assert_called_once_with(
        "metagenomics", repo=None, bundle="metagenomics"
    )


def test_query_endpoint_with_repo_scope(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.query.return_value = {"answer": "repo answer"}
    mock_control_plane.get_engine.return_value = mock_engine

    response = client.post("/query", json={"query": "model versioning", "repo": "omnibioai-model-registry"})

    assert response.status_code == 200
    mock_engine.query.assert_called_once_with(
        "model versioning", repo="omnibioai-model-registry", bundle=None
    )


def test_query_endpoint_unscoped_passes_none_filters(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.query.return_value = {"answer": "answer"}
    mock_control_plane.get_engine.return_value = mock_engine

    client.post("/query", json={"query": "hello"})

    mock_engine.query.assert_called_once_with("hello", repo=None, bundle=None)


def test_stream_endpoint_with_bundle_scope(client, mock_control_plane):
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = [{"text": "ctx"}]
    mock_engine.build_context.return_value = "ctx"
    mock_engine.stream_llm.return_value = ["tok"]
    mock_control_plane.get_engine.return_value = mock_engine

    response = client.post("/stream", json={"query": "q", "bundle": "metagenomics"})

    assert response.status_code == 200
    mock_engine.retrieve.assert_called_once_with("q", repo=None, bundle="metagenomics")

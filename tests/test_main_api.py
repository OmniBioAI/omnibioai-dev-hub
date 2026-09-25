"""Tests for the Dev Hub FastAPI app (api.main): health and status endpoints, the readiness guard
middleware, seed builders, and control-plane startup, with faiss and sentence_transformers mocked
out.

Developer: Manish Kumar <manish@omnibioai.org>
"""

import importlib
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import sys
from starlette.requests import Request
from fastapi.testclient import TestClient

# Mock heavy dependencies properly
mock_faiss = MagicMock()
mock_faiss.__spec__ = MagicMock()
sys.modules['faiss'] = mock_faiss

mock_st = MagicMock()
mock_st.__spec__ = MagicMock()
sys.modules['sentence_transformers'] = mock_st

# Mock dependencies at their source to avoid issues during import of api.main
with patch("index.vector_store.VectorStore"),      patch("index.graph_store.GraphStore"),      patch("index.plugin_index.PluginIndex"):
    from api.main import app

from api import main as main_module

def test_health_endpoint():
    """Report an ok status from /health while the control plane is ready."""
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        response = main_module.health()
        assert response["status"] == "ok"

def test_status_endpoint():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        with patch("api.main.graph_store") as mock_gs:
            mock_gs.size.return_value = {"nodes": 3, "edges": 4}
            response = main_module.status()
            assert response["graph_edges"] == 4


@pytest.mark.asyncio
async def test_guard_requests_middleware_ready():
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}):
        with patch("api.routes.rag.get_engine"):
            request = Request({"type": "http", "method": "POST", "path": "/rag/query", "headers": [], "query_string": b""})
            response = await main_module.guard_requests(request, lambda _: asyncio.sleep(0))
            assert response is None

@pytest.mark.asyncio
async def test_guard_requests_middleware_not_ready():
    """Report the control-plane status and the graph edge count from /status."""
    with (
        patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}),
        patch("api.main.graph_store") as mock_gs,
    ):
        mock_gs.size.return_value = {"nodes": 3, "edges": 4}
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["graph_edges"] == 4


def test_status_endpoint_counts_distinct_omnibioai_repos():
    """Count each distinct omnibioai repository once in repos_loaded and ignore sources outside any
    omnibioai directory."""
    with (
        patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}),
        patch("api.main.graph_store") as mock_gs,
        patch("api.main.vector_store") as mock_vs,
    ):
        mock_gs.size.return_value = {"nodes": 1, "edges": 2}
        mock_vs.index = None
        mock_vs.metadata = [
            {"source": "/data/omnibioai-workbench/README.md"},
            {"source": "/data/omnibioai-workbench/docs/notes.md"},  # same repo, deduped
            {"source": "/data/omnibioai-auth/README.md"},
            {"source": "/data/unrelated/notes.md"},  # no "omnibioai*" segment at all
        ]
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["repos_loaded"] == 2


def test_warns_when_no_persisted_faiss_index_is_found(caplog):
    """Log a warning at import time when no persisted FAISS index can be loaded."""
    import api.main as main_module
    from index.vector_store import VectorStore

    try:
        with (
            patch.object(VectorStore, "load", return_value=False),
            caplog.at_level(logging.WARNING, logger="api.main"),
        ):
            importlib.reload(main_module)
        assert any(
            "No FAISS index loaded" in record.message for record in caplog.records
        )
    finally:
        # Restore api.main to a normal (unpatched) import for any test that
        # runs after this one and reaches for its module-level state.
        importlib.reload(main_module)


def test_guard_requests_middleware_ready():
    """Let RAG requests pass the guard middleware once the control plane is READY."""
    with (
        patch("api.main.CONTROL_PLANE.status", return_value={"status": "READY"}),
        patch("api.routes.rag.get_engine"),
    ):
        response = client.post("/rag/query", json={"query": "q"})
        assert response.status_code != 503

def test_guard_requests_middleware_not_ready():
    """Reject RAG requests with a 503 'Control plane not ready' while the control plane is not
    READY."""
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "INIT"}):
        request = Request({"type": "http", "method": "POST", "path": "/rag/query", "headers": [], "query_string": b""})
        response = await main_module.guard_requests(request, lambda _: asyncio.sleep(0))
        assert response.status_code == 503
        assert response.body == b'{"detail":"Control plane not ready"}'

def test_build_graph_seed():
    """Seed the application-level graph store with its four starter edges."""
    from api.main import build_graph_seed
    mock_gs = MagicMock()
    with patch("api.main.graph_store", mock_gs):
        build_graph_seed()
        assert mock_gs.add_edge.call_count == 4

def test_build_plugin_index():
    """Seed the plugin index with its three starter documents."""
    from api.main import build_plugin_index
    mock_pi = MagicMock()
    with patch("api.main.plugin_index", mock_pi):
        build_plugin_index()
        assert len(mock_pi.docs) == 3

@pytest.mark.asyncio
async def test_init_control_plane():
    """Initialize the control plane exactly once."""
    from api.main import init_control_plane
    with patch("api.main.CONTROL_PLANE.init") as mock_init:
        await init_control_plane()
        mock_init.assert_called_once()

@pytest.mark.asyncio
async def test_startup_event():
    """Initialize the control plane during application startup."""
    from api.main import startup_event
    with patch("api.main.init_control_plane", new_callable=AsyncMock) as mock_init:
        await startup_event()
        mock_init.assert_called_once()

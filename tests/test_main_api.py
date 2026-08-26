import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import sys
from starlette.requests import Request

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
    with patch("api.main.CONTROL_PLANE.status", return_value={"status": "INIT"}):
        request = Request({"type": "http", "method": "POST", "path": "/rag/query", "headers": [], "query_string": b""})
        response = await main_module.guard_requests(request, lambda _: asyncio.sleep(0))
        assert response.status_code == 503
        assert response.body == b'{"detail":"Control plane not ready"}'

def test_build_graph_seed():
    from api.main import build_graph_seed, graph_store
    mock_gs = MagicMock()
    with patch("api.main.graph_store", mock_gs):
        build_graph_seed()
        assert mock_gs.add_edge.call_count == 4

def test_build_plugin_index():
    from api.main import build_plugin_index, plugin_index
    mock_pi = MagicMock()
    with patch("api.main.plugin_index", mock_pi):
        build_plugin_index()
        assert len(mock_pi.docs) == 3

@pytest.mark.asyncio
async def test_init_control_plane():
    from api.main import init_control_plane
    with patch("api.main.CONTROL_PLANE.init") as mock_init:
        await init_control_plane()
        mock_init.assert_called_once()

@pytest.mark.asyncio
async def test_startup_event():
    from api.main import startup_event
    with patch("api.main.init_control_plane", new_callable=AsyncMock) as mock_init:
        await startup_event()
        mock_init.assert_called_once()

from unittest.mock import MagicMock, patch

import pytest

from rag.tool_executor import ToolExecutorV4


@pytest.fixture
def mock_vs():
    return MagicMock()

@pytest.fixture
def mock_gs():
    return MagicMock()

@pytest.fixture
def mock_pi():
    return MagicMock()

@pytest.fixture
def mock_emb():
    return MagicMock()

@pytest.fixture
def executor(mock_vs, mock_gs, mock_pi, mock_emb):
    return ToolExecutorV4(mock_vs, mock_gs, mock_pi, mock_emb)

def test_run_all_steps(executor, mock_vs, mock_gs, mock_pi, mock_emb):
    plan = {
        "steps": ["vector_search", "graph_search", "plugin_search", "memory_search", "hybrid_expand"]
    }
    mock_emb.encode.return_value = [[0.1]*768]
    mock_vs.search.return_value = [{"text": "v", "source": "vector"}]
    mock_gs.search.return_value = [{"text": "g", "source": "graph"}]
    mock_pi.search.return_value = [{"text": "p", "source": "plugin"}]
    
    results = executor.run(plan, "query")
    assert len(results) >= 5
    # memory search returns a placeholder
    assert any(r.get("source") == "memory" for r in results)

def test_run_error_handling(executor):
    plan = {"steps": ["vector_search"]}
    executor.vector_store = MagicMock()
    executor.embedder.encode.side_effect = Exception("Embed fail")
    
    # In run(), individual tool errors are caught and added to results
    results = executor.run(plan, "query")
    assert results[0]["source"] == "vector_error"

def test_run_global_error(executor):
    # Test the broad try-except in the loop
    # We can trigger it by making _vector raise instead of returning a list
    with patch.object(executor, "_vector", side_effect=ValueError("Global fail")):
        plan = {"steps": ["vector_search"]}
        results = executor.run(plan, "q")
        assert "[TOOL_ERROR] vector_search: Global fail" in results[0]["text"]

def test_vector_search_failure(executor):
    executor.vector_store = MagicMock()
    executor.embedder.encode.side_effect = Exception("Embed fail")
    res = executor._vector("q")
    assert res[0]["source"] == "vector_error"
    
    executor.vector_store = None
    assert executor._vector("q") == []

def test_graph_search_failure(executor):
    executor.graph_store = MagicMock()
    executor.graph_store.search.side_effect = Exception("Graph fail")
    res = executor._graph("q")
    assert res[0]["source"] == "graph_error"
    
    executor.graph_store = None
    assert executor._graph("q") == []

def test_plugin_search_failure(executor):
    executor.plugin_index = MagicMock()
    executor.plugin_index.search.side_effect = Exception("Plugin fail")
    res = executor._plugin("q")
    assert res[0]["source"] == "plugin_error"
    
    executor.plugin_index = None
    assert executor._plugin("q") == []

def test_hybrid_expand(executor):
    with patch.object(executor, "_vector", return_value=[{"text": "v"}]),          patch.object(executor, "_graph", return_value=[{"text": "g"}]):
        res = executor._hybrid_expand("q")
        assert len(res) == 2

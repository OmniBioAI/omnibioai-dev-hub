"""Unit tests for RAGQueryRouterV4 with mocked stores and embedder: intent detection, per-source
search with error handling, context building, LLM streaming, and the engine singleton.

Developer: Manish Kumar <manish@omnibioai.org>
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.query_router import RAGQueryRouterV4, get_engine, init_engine


@pytest.fixture
def mock_vs():
    """Provide a MagicMock standing in for the vector store."""
    return MagicMock()

@pytest.fixture
def mock_gs():
    """Provide a MagicMock standing in for the graph store."""
    return MagicMock()

@pytest.fixture
def mock_pi():
    """Provide a MagicMock standing in for the plugin index."""
    return MagicMock()

@pytest.fixture
def mock_emb():
    """Provide a MagicMock standing in for the embedder."""
    return MagicMock()

@pytest.fixture
def router(mock_vs, mock_gs, mock_pi, mock_emb):
    """Build a RAGQueryRouterV4 wired to the mocked collaborators."""
    return RAGQueryRouterV4(mock_vs, mock_gs, mock_pi, mock_emb)

def test_detect_intent(router):
    """Route graph-related, plugin-related, and generic queries to the graph, plugin, and hybrid
    intents."""
    assert router.detect_intent("show me the graph") == "graph"
    assert router.detect_intent("run workflow") == "plugin"
    assert router.detect_intent("hello world") == "hybrid"

def test_vector_search_success(router, mock_vs, mock_emb):
    """Encode the query and return the vector store's results, searching with the router's top_k."""
    mock_emb.encode.return_value = [[0.1, 0.2]]
    mock_vs.search.return_value = [{"text": "result"}]
    
    res = router.vector_search("query")
    assert res == [{"text": "result"}]
    mock_vs.search.assert_called_once_with([0.1, 0.2], top_k=router.top_k)

def test_vector_search_no_store(router):
    """Return no results when no vector store is configured."""
    router.vector_store = None
    assert router.vector_search("q") == []

def test_vector_search_error(router, mock_emb):
    """Return no results instead of raising when encoding fails."""
    mock_emb.encode.side_effect = Exception("Encode error")
    assert router.vector_search("q") == []

def test_graph_search(router, mock_gs):
    """Return the graph store's results, or nothing when no graph store is configured."""
    mock_gs.search.return_value = [{"text": "g"}]
    assert router.graph_search("q") == [{"text": "g"}]
    
    router.graph_store = None
    assert router.graph_search("q") == []

def test_graph_search_error(router, mock_gs):
    """Return no results instead of raising when the graph search fails."""
    mock_gs.search.side_effect = Exception("G error")
    assert router.graph_search("q") == []

def test_plugin_search(router, mock_pi):
    """Return the plugin index's results, or nothing when no plugin index is configured."""
    mock_pi.search.return_value = [{"text": "p"}]
    assert router.plugin_search("q") == [{"text": "p"}]
    
    router.plugin_index = None
    assert router.plugin_search("q") == []

def test_plugin_search_error(router, mock_pi):
    """Return no results instead of raising when the plugin search fails."""
    mock_pi.search.side_effect = Exception("P error")
    assert router.plugin_search("q") == []

def test_hybrid_retrieve(router):
    """Combine the detected intent with the vector, graph, and plugin results."""
    with (
        patch.object(router, "detect_intent", return_value="intent"),
        patch.object(router, "vector_search", return_value=[]),
        patch.object(router, "graph_search", return_value=[]),
        patch.object(router, "plugin_search", return_value=[]),
    ):
        res = router.hybrid_retrieve("q")
        assert res["intent"] == "intent"

def test_build_context(router):
    """Label each source's results as VECTOR, GRAPH, or PLUGIN in the assembled context."""
    results = {
        "vector": [{"text": "v1"}],
        "graph": [{"text": "g1"}],
        "plugin": [{"text": "p1"}]
    }
    ctx = router.build_context(results)
    assert "[VECTOR] v1" in ctx
    assert "[GRAPH] g1" in ctx
    assert "[PLUGIN] p1" in ctx

@patch("rag.query_router.requests.post")
def test_stream_llm_success(mock_post, router):
    """Yield the tokens streamed back from the mocked LLM endpoint."""
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        json.dumps({"message": {"content": "tok1"}}).encode("utf-8"),
        json.dumps({"message": {"content": "tok2"}}).encode("utf-8")
    ]
    mock_post.return_value = mock_response
    
    tokens = list(router.stream_llm("q", "ctx"))
    assert tokens == ["tok1", "tok2"]

@patch("rag.query_router.requests.post")
def test_stream_llm_error(mock_post, router):
    """Yield a STREAM_ERROR token when the LLM request fails."""
    mock_post.side_effect = Exception("HTTP error")
    tokens = list(router.stream_llm("q", "ctx"))
    assert "[STREAM_ERROR] HTTP error" in tokens[0]

def test_query(router):
    """Return the intent and the streamed answer for a query."""
    with (
        patch.object(router, "hybrid_retrieve", return_value={"intent": "i", "vector": [], "graph": [], "plugin": []}),
        patch.object(router, "stream_llm", return_value=["ans"]),
    ):
        res = router.query("q")
        assert res["intent"] == "i"
        assert res["answer"] == "ans"

def test_engine_singleton():
    """Return the initialized engine and raise while no engine has been initialized."""
    init_engine(MagicMock())
    assert get_engine() is not None
    
    # Test error
    import rag.query_router
    rag.query_router._ENGINE = None
    with pytest.raises(RuntimeError, match="RAG engine not initialized"):
        get_engine()

@patch("rag.query_router.requests.post")
def test_stream_llm_empty_line_and_invalid_json(mock_post, router):
    """Skip empty and invalid-JSON lines in the LLM stream while still yielding the valid token."""
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b"", # Empty line
        b"invalid json", # Invalid JSON
        json.dumps({"message": {"content": "ok"}}).encode("utf-8")
    ]
    mock_post.return_value = mock_response
    
    tokens = list(router.stream_llm("q", "ctx"))
    assert tokens == ["ok"]

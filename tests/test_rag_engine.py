"""Unit tests for rag.engine with requests mocked out: the Ollama embed and generate helpers, cosine
similarity, and RAGEngine embedding, scoped retrieval, cross-encoder reranking, answer
generation, LLM streaming, and model-config loading.

Developer: Manish Kumar <manish@omnibioai.org>
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

import rag.engine as _engine_mod
from rag.engine import RAGEngine, _load_llm_model, cosine, ollama_embed, ollama_generate

# =========================================================
# UNIT TESTS FOR ollama_embed
# =========================================================

@patch("rag.engine.requests.post")
def test_ollama_embed_success(mock_post):
    """Return a float32 embedding of the expected 768 dimensions from the Ollama response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 768}
    mock_post.return_value = mock_response
    
    vec = ollama_embed("test text")
    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    mock_post.assert_called_once()

@patch("rag.engine.requests.post")
def test_ollama_embed_batch_dim(mock_post):
    """Flatten a nested single-vector embedding response into a 768-dimension vector."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [[0.1] * 768]}
    mock_post.return_value = mock_response
    
    vec = ollama_embed("test text")
    assert vec.shape == (768,)

@patch("rag.engine.requests.post")
def test_ollama_embed_dim_mismatch(mock_post):
    """Reject an Ollama embedding whose dimension is not 768."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 512}
    mock_post.return_value = mock_response
    
    with pytest.raises(ValueError, match="Embedding dim mismatch"):
        ollama_embed("test text")

# =========================================================
# UNIT TESTS FOR ollama_generate
# =========================================================

@patch("rag.engine.requests.post")
def test_ollama_generate_success(mock_post):
    """Return the generated text from the Ollama generate response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "generated answer"}
    mock_post.return_value = mock_response
    
    resp = ollama_generate("prompt")
    assert resp == "generated answer"

# =========================================================
# UNIT TESTS FOR cosine
# =========================================================

def test_cosine():
    """Score identical vectors as 1.0 and orthogonal vectors as 0.0, and score a zero vector as 0.0."""
    a = [1, 0]
    b = [1, 0]
    assert cosine(a, b) == 1.0
    
    c = [0, 1]
    assert cosine(a, c) == 0.0
    
    d = [0, 0]
    assert cosine(a, d) == 0.0

# =========================================================
# RAGEngine TESTS
# =========================================================

@pytest.fixture
def mock_vector_store():
    """Provide a MagicMock standing in for the vector store."""
    return MagicMock()

@pytest.fixture
def engine(mock_vector_store):
    """Build a RAGEngine around the mocked vector store."""
    return RAGEngine(mock_vector_store)

def test_engine_init(engine, mock_vector_store):
    """Store the vector store and default to the nomic-embed-text embedding model."""
    assert engine.vector_store == mock_vector_store
    assert engine.embed_model == "nomic-embed-text"

@patch("rag.engine.ollama_embed")
def test_engine_embed_success(mock_embed, engine):
    """Embed a query string into a 768-dimension vector."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    vec = engine._embed("query")
    assert vec.shape == (768,)

@patch("rag.engine.ollama_embed")
def test_engine_embed_invalid_type(mock_embed, engine):
    """Reject a non-string query with a TypeError."""
    with pytest.raises(TypeError, match="Query must be a string"):
        engine._embed(123)

@patch("rag.engine.ollama_embed")
def test_engine_embed_dim_mismatch(mock_embed, engine):
    """Reject a query embedding whose dimension is not 768."""
    mock_embed.return_value = np.array([0.1] * 512, dtype=np.float32)
    with pytest.raises(ValueError, match="Query embedding mismatch"):
        engine._embed("query")

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_empty(mock_embed, engine, mock_vector_store):
    """Return no results when the vector store has no index."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    mock_vector_store.index = None
    
    results = engine.retrieve("query")
    assert results == []

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_success(mock_embed, engine, mock_vector_store):
    """Return the stored documents for the indices the index search reports, in order."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    
    mock_index = MagicMock()
    mock_index.ntotal = 10
    mock_index.search.return_value = (
        np.array([[0.9, 0.8]]), # scores
        np.array([[1, 2]])      # indices
    )
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [
        {}, # 0
        {"text": "text1", "source": "src1"}, # 1
        {"text": "text2", "source": "src2"}  # 2
    ]
    
    results = engine.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "text1"
    assert results[1]["text"] == "text2"

@patch("rag.engine.ollama_embed")
def test_engine_retrieve_invalid_indices(mock_embed, engine, mock_vector_store):
    """Return no results when the search reports indices that have no stored document."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)
    
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (
        np.array([[0.9]]), # scores
        np.array([[-1]])   # invalid index
    )
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [{"text": "text1"}]
    
    results = engine.retrieve("query")
    assert results == []

def test_build_context(engine):
    """Format each document as its bracketed source followed by its text."""
    docs = [
        {"text": "t1", "source": "s1"},
        {"text": "t2", "source": "s2"}
    ]
    ctx = engine.build_context(docs)
    assert "[s1]\nt1" in ctx
    assert "[s2]\nt2" in ctx

def test_build_context_empty(engine):
    """Return the 'No relevant context found.' placeholder when there are no documents."""
    assert engine.build_context([]) == "No relevant context found."

def test_build_prompt(engine):
    """Include both the context and the question in the prompt."""
    prompt = engine.build_prompt("q", "ctx")
    assert "ctx" in prompt
    assert "q" in prompt

@patch("rag.engine.ollama_generate")
def test_engine_answer_success(mock_gen, engine, mock_vector_store):
    """Return the generated answer, the retrieved sources, and the v6-faiss version."""
    mock_gen.return_value = "final answer"
    
    # Mock retrieve to return something
    with patch.object(engine, "retrieve", return_value=[{"source": "s1"}]) as mock_retrieve:
        res = engine.answer("query")
        mock_retrieve.assert_called_once_with("query", repo=None, bundle=None)
        assert res["answer"] == "final answer"
        assert res["sources"] == ["s1"]
        assert res["version"] == "v6-faiss"

@patch("rag.engine.ollama_generate")
def test_engine_answer_failure(mock_gen, engine, mock_vector_store):
    """Return an LLM_ERROR answer instead of raising when generation fails."""
    mock_gen.side_effect = Exception("Gen failed")
    
    with patch.object(engine, "retrieve", return_value=[]):
        res = engine.answer("query")
        assert "[LLM_ERROR] Gen failed" in res["answer"]

def test_engine_query(engine):
    """Delegate query to answer with no repo or bundle scope."""
    with patch.object(engine, "answer", return_value={"ok": True}) as mock_answer:
        res = engine.query("q")
        assert res["ok"] is True
        mock_answer.assert_called_once_with("q", repo=None, bundle=None)


def test_engine_query_passes_scope(engine):
    """Pass the repo and bundle scope from query through to answer."""
    with patch.object(engine, "answer", return_value={"ok": True}) as mock_answer:
        engine.query("q", repo="my-repo", bundle="my-bundle")
        mock_answer.assert_called_once_with("q", repo="my-repo", bundle="my-bundle")


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_with_bundle_filter(mock_embed, engine, mock_vector_store):
    """Use the vector store's filtered search when a bundle scope is given."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []

    # filter_search is present on the mock
    mock_vector_store.filter_search.return_value = [
        {"score": 0.9, "text": "filtered", "source": "s1", "repo": "r", "bundle": "b"}
    ]

    results = engine.retrieve("query", top_k=5, bundle="b")
    mock_vector_store.filter_search.assert_called_once()
    assert results[0]["text"] == "filtered"


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_with_repo_filter(mock_embed, engine, mock_vector_store):
    """Use the vector store's filtered search on the repo field when a repo scope is given."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []
    mock_vector_store.filter_search.return_value = [
        {"score": 0.8, "text": "repo-result", "source": "s2", "repo": "my-repo", "bundle": None}
    ]

    results = engine.retrieve("query", repo="my-repo")
    # Should call filter_search with field="repo"
    call_kwargs = mock_vector_store.filter_search.call_args
    assert call_kwargs[1].get("field") == "repo" or call_kwargs[0][2] == "repo"
    assert results[0]["text"] == "repo-result"


@patch("rag.engine.ollama_embed")
def test_engine_retrieve_bundle_takes_priority_over_repo(mock_embed, engine, mock_vector_store):
    """Filter on the bundle when both a bundle and a repo are given."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = []
    mock_vector_store.filter_search.return_value = []

    engine.retrieve("query", repo="r", bundle="b")
    call_kwargs = mock_vector_store.filter_search.call_args
    # bundle takes priority — field should be "bundle"
    args = call_kwargs[0]
    kwargs = call_kwargs[1]
    field_used = kwargs.get("field") or (args[2] if len(args) > 2 else None)
    assert field_used == "bundle"


@patch("rag.engine.ollama_embed")
def test_engine_answer_passes_scope(mock_embed, engine, mock_vector_store):
    """Pass the repo and bundle scope from answer through to retrieve."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    with (
        patch.object(engine, "retrieve", return_value=[]) as mock_retrieve,
        patch("rag.engine.ollama_generate", return_value="ans"),
    ):
        engine.answer("q", repo="r", bundle="b")
        mock_retrieve.assert_called_once_with("q", repo="r", bundle="b")


# =========================================================
# RERANKING TESTS
# =========================================================

# =========================================================
# _get_cross_encoder — unit tests for the lazy-loader
# =========================================================

def test_get_cross_encoder_returns_cached_instance():
    """Return the cached cross-encoder without loading it again."""
    old = _engine_mod._CROSS_ENCODER
    sentinel = MagicMock()
    _engine_mod._CROSS_ENCODER = sentinel
    try:
        assert _engine_mod._get_cross_encoder() is sentinel
    finally:
        _engine_mod._CROSS_ENCODER = old


def test_get_cross_encoder_false_sentinel_returns_none():
    """Return None when an earlier load failure is cached as False."""
    old = _engine_mod._CROSS_ENCODER
    _engine_mod._CROSS_ENCODER = False
    try:
        assert _engine_mod._get_cross_encoder() is None
    finally:
        _engine_mod._CROSS_ENCODER = old


def test_get_cross_encoder_loads_on_first_call():
    """Load and return the cross-encoder on the first call when none is cached."""
    old = _engine_mod._CROSS_ENCODER
    _engine_mod._CROSS_ENCODER = None
    mock_ce_instance = MagicMock()
    mock_st = MagicMock()
    mock_st.CrossEncoder = MagicMock(return_value=mock_ce_instance)
    try:
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = _engine_mod._get_cross_encoder()
        assert result is mock_ce_instance
    finally:
        _engine_mod._CROSS_ENCODER = old


def test_get_cross_encoder_handles_import_failure():
    """Return None and cache False when sentence_transformers cannot be imported."""
    old = _engine_mod._CROSS_ENCODER
    _engine_mod._CROSS_ENCODER = None
    try:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = _engine_mod._get_cross_encoder()
        assert result is None
        assert _engine_mod._CROSS_ENCODER is False
    finally:
        _engine_mod._CROSS_ENCODER = old


# =========================================================
# RERANKING TESTS
# =========================================================

def test_rerank_empty_docs_returns_empty(engine):
    """Return an empty list when there are no documents to rerank."""
    assert engine.rerank("q", [], top_k=5) == []


def test_rerank_returns_top_k(engine):
    """Return the top_k documents ordered by cross-encoder score."""
    docs = [{"text": f"doc{i}", "source": f"s{i}", "score": float(i)} for i in range(10)]
    mock_ce = MagicMock()
    # Assign descending scores so doc9 ranks highest
    mock_ce.predict.return_value = [float(i) for i in range(10)]

    with patch("rag.engine._get_cross_encoder", return_value=mock_ce):
        result = engine.rerank("query", docs, top_k=3)

    assert len(result) == 3
    assert result[0]["text"] == "doc9"  # highest CE score


def test_rerank_attaches_ce_score(engine):
    """Attach a ce_score to each reranked document, highest first."""
    docs = [{"text": "a", "source": "s1"}, {"text": "b", "source": "s2"}]
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.3, 0.9]

    with patch("rag.engine._get_cross_encoder", return_value=mock_ce):
        result = engine.rerank("q", docs, top_k=2)

    assert "ce_score" in result[0]
    assert result[0]["ce_score"] > result[1]["ce_score"]


def test_rerank_falls_back_when_ce_unavailable(engine):
    """Return the documents unchanged when no cross-encoder is available."""
    docs = [{"text": "x", "source": "s"}]
    with patch("rag.engine._get_cross_encoder", return_value=None):
        result = engine.rerank("q", docs, top_k=5)
    assert result == docs  # unchanged


@patch("rag.engine.ollama_embed")
def test_retrieve_with_rerank_fetches_wider_set(mock_embed, engine, mock_vector_store):
    """Fetch three times top_k candidates from the index when reranking, then return top_k results."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 20
    mock_index.search.return_value = (
        np.array([[0.9] * 15]),
        np.array([[i for i in range(15)]])
    )
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [
        {"text": f"t{i}", "source": f"s{i}"} for i in range(20)
    ]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = list(range(15))  # ascending scores

    with patch("rag.engine._get_cross_encoder", return_value=mock_ce):
        result = engine.retrieve("q", top_k=5, rerank=True)

    # With top_k=5 and rerank=True, FAISS was searched with k=15 (5*3)
    call_args = mock_index.search.call_args
    assert call_args[0][1] == 15  # fetch_k = top_k * 3
    assert len(result) == 5


@patch("rag.engine.ollama_embed")
def test_retrieve_without_rerank_fetches_exact_top_k(mock_embed, engine, mock_vector_store):
    """Fetch exactly top_k candidates from the index when not reranking."""
    mock_embed.return_value = np.array([0.1] * 768, dtype=np.float32)

    mock_index = MagicMock()
    mock_index.ntotal = 20
    mock_index.search.return_value = (np.array([[0.9] * 5]), np.array([[i for i in range(5)]]))
    mock_vector_store.index = mock_index
    mock_vector_store.metadata = [{"text": f"t{i}", "source": f"s{i}"} for i in range(20)]

    engine.retrieve("q", top_k=5, rerank=False)

    call_args = mock_index.search.call_args
    assert call_args[0][1] == 5  # exact top_k, no multiplier


# =========================================================
# UNIT TESTS FOR _load_llm_model
# =========================================================

def test_load_llm_model_defaults_when_file_missing(tmp_path):
    """Fall back to the default model when the config file does not exist."""
    missing = tmp_path / "does-not-exist.yaml"
    assert _load_llm_model(default="fallback-model", path=str(missing)) == "fallback-model"


def test_load_llm_model_defaults_on_malformed_yaml(tmp_path):
    """Fall back to the default model when the config file is malformed YAML."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm_model: [unterminated")
    assert _load_llm_model(default="fallback-model", path=str(bad)) == "fallback-model"


def test_load_llm_model_defaults_when_yaml_is_not_a_mapping(tmp_path):
    """Fall back to the default model when the config file is not a mapping."""
    not_a_map = tmp_path / "list.yaml"
    not_a_map.write_text("- one\n- two\n")
    assert _load_llm_model(default="fallback-model", path=str(not_a_map)) == "fallback-model"


def test_load_llm_model_defaults_when_key_is_absent(tmp_path):
    """Fall back to the default model when llm_model is absent from the config file."""
    no_key = tmp_path / "no_key.yaml"
    no_key.write_text("other_setting: true\n")
    assert _load_llm_model(default="fallback-model", path=str(no_key)) == "fallback-model"


def test_load_llm_model_reads_configured_value(tmp_path):
    """Return the llm_model value configured in the file."""
    configured = tmp_path / "config.yaml"
    configured.write_text("llm_model: mixtral\n")
    assert _load_llm_model(default="fallback-model", path=str(configured)) == "mixtral"


# =========================================================
# UNIT TESTS FOR ollama_embed RETRY BEHAVIOR
# =========================================================

@patch("rag.engine.time.sleep")
@patch("rag.engine.requests.post")
def test_ollama_embed_retries_then_succeeds(mock_post, mock_sleep):
    """Retry a transient connection failure with a backoff sleep and succeed on the second attempt."""
    failure = requests.exceptions.ConnectionError("transient CUDA context failure")
    success = MagicMock()
    success.json.return_value = {"embedding": [0.1] * 768}
    success.raise_for_status.return_value = None
    mock_post.side_effect = [failure, success]

    vec = ollama_embed("test text")

    assert vec.shape == (768,)
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(2)  # 2 * attempt(1)


@patch("rag.engine.time.sleep")
@patch("rag.engine.requests.post")
def test_ollama_embed_raises_after_exhausting_all_retries(mock_post, mock_sleep):
    """Raise the connection error after three failed attempts, sleeping only between attempts."""
    mock_post.side_effect = requests.exceptions.ConnectionError("still down")

    with pytest.raises(requests.exceptions.ConnectionError):
        ollama_embed("test text")

    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2  # slept between attempts 1->2 and 2->3, not after the last


# =========================================================
# UNIT TESTS FOR RAGEngine.stream_llm
# =========================================================

@patch("rag.engine.requests.post")
def test_stream_llm_yields_tokens_and_stops_on_done(mock_post, engine):
    """Yield each streamed response token and stop at the done marker."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [
        b'{"response": "Hello"}',
        b"",  # blank keep-alive line -- must be skipped, not yielded
        b'{"response": " world"}',
        b'{"response": "", "done": true}',
        b'{"response": "unreachable after done"}',
    ]
    mock_post.return_value = response

    tokens = list(engine.stream_llm("query", "context"))

    assert tokens == ["Hello", " world"]
    _, kwargs = mock_post.call_args
    assert kwargs["stream"] is True


@patch("rag.engine.requests.post")
def test_stream_llm_yields_inline_error_token_on_failure(mock_post, engine):
    """Yield a single LLM_ERROR token when the streaming request fails."""
    mock_post.side_effect = requests.exceptions.RequestException("ollama unreachable")

    tokens = list(engine.stream_llm("query", "context"))

    assert len(tokens) == 1
    assert tokens[0].startswith("[LLM_ERROR]")
    assert "ollama unreachable" in tokens[0]

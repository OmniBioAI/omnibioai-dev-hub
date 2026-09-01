import json
import logging
import os
import time
from collections.abc import Generator
from typing import Any

import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)

# =========================================================
# CROSS-ENCODER (lazy import — optional dependency)
# =========================================================
_CROSS_ENCODER = None
_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder(_CE_MODEL, device="cpu")
            logger.info(f"CrossEncoder loaded: {_CE_MODEL}")
        except Exception as e:  # noqa: BLE001 -- optional dependency: any load failure degrades to no reranking rather than crashing
            logger.warning(f"CrossEncoder unavailable ({e}); reranking will be skipped")
            _CROSS_ENCODER = False  # sentinel: tried and failed
    return _CROSS_ENCODER if _CROSS_ENCODER is not False else None


# =========================================================
# OLLAMA CONFIG
# =========================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api")
EMBED_DIM = 768

_INDEX_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "index_config.yaml"
)
_DEFAULT_LLM_MODEL = "llama3"


def _load_llm_model(default: str = _DEFAULT_LLM_MODEL, path: str = _INDEX_CONFIG_PATH) -> str:
    """Read `llm_model` from configs/index_config.yaml.

    Falls back to `default` if the file is missing, empty, fails to parse,
    or doesn't set llm_model -- so behavior is unchanged unless the config
    is explicitly edited. Both ollama_generate()'s default arg and
    RAGEngine.stream_llm() read this same LLM_MODEL value.
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Could not read {path} ({e}); defaulting llm_model to {default!r}")
        return default

    if not isinstance(data, dict):
        return default

    value = data.get("llm_model")
    return str(value) if value else default


LLM_MODEL = _load_llm_model()


def ollama_embed(text: str, model: str = "nomic-embed-text"):
    # Ollama's llama-server subprocess can transiently fail CUDA context
    # init under host memory pressure on unified-memory GPUs (issue #12) —
    # a short retry recovers once pressure passes, without masking a
    # persistently broken Ollama.
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            res = requests.post(
                f"{OLLAMA_URL}/embeddings",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=60
            )
            res.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == attempts:
                raise
            logger.warning(
                f"ollama_embed attempt {attempt}/{attempts} failed, retrying"
            )
            time.sleep(2 * attempt)

    vec = res.json()["embedding"]

    # =====================================================
    # V6 HARD NORMALIZATION (FAISS SAFE)
    # =====================================================
    vec = np.array(vec, dtype=np.float32)

    # remove batch dimension if present
    if vec.ndim == 2:
        vec = vec[0]

    # enforce correct dimension
    if vec.shape[0] != EMBED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {vec.shape[0]} expected {EMBED_DIM}"
        )

    return vec


def ollama_generate(prompt: str, model: str = LLM_MODEL):
    res = requests.post(
        f"{OLLAMA_URL}/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )
    res.raise_for_status()
    return res.json().get("response", "")


# =========================================================
# COSINE (fallback only)
# =========================================================
def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# =========================================================
# RAG ENGINE (V6 FAISS-NATIVE)
# =========================================================
class RAGEngine:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embed_model = "nomic-embed-text"

    # =====================================================
    # EMBEDDING (SINGLE SOURCE OF TRUTH)
    # =====================================================
    def _embed(self, text: str):

        if not isinstance(text, str):
            raise TypeError("Query must be a string")

        vec = ollama_embed(text, model=self.embed_model)

        # ensure numpy + correct shape
        vec = np.array(vec, dtype=np.float32).reshape(-1)

        if vec.shape[0] != EMBED_DIM:
            raise ValueError(
                f"Query embedding mismatch: got {vec.shape[0]} expected {EMBED_DIM}"
            )

        return vec

    # =====================================================
    # CROSS-ENCODER RERANKING
    # =====================================================
    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """Re-order docs by cross-encoder relevance score.

        Returns the top_k highest-scoring docs.  Falls back to the original
        FAISS order if the cross-encoder model cannot be loaded.
        """
        if not docs:
            return docs

        ce = _get_cross_encoder()
        if ce is None:
            logger.warning("rerank() called but CrossEncoder is unavailable; returning FAISS order")
            return docs[:top_k]

        pairs = [(query, d.get("text", "")) for d in docs]
        scores = ce.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = [d for _, d in ranked[:top_k]]

        # Attach cross-encoder score for transparency
        for score, doc in zip([s for s, _ in ranked[:top_k]], reranked):
            doc["ce_score"] = float(score)

        return reranked

    # =====================================================
    # RETRIEVAL (FAISS ONLY)
    # =====================================================
    def retrieve(self, query: str, top_k: int = 5, repo: str | None = None, bundle: str | None = None,
                 rerank: bool = False):

        query_vec = self._embed(query).reshape(1, -1)

        vs = self.vector_store
        index = getattr(vs, "index", None)
        metadata = getattr(vs, "metadata", [])

        if index is None or index.ntotal == 0:
            return []

        # When reranking, fetch a wider candidate set for the cross-encoder to score
        fetch_k = top_k * 3 if rerank else top_k

        # Route through filter_search when a scope is requested
        if (repo is not None or bundle is not None) and hasattr(vs, "filter_search"):
            field = "bundle" if bundle is not None else "repo"
            value = bundle if bundle is not None else repo
            candidates = vs.filter_search(query_vec, fetch_k, field=field, value=value)
        else:
            k = min(fetch_k, index.ntotal)
            scores, indices = index.search(query_vec, k)

            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(metadata):
                    continue
                candidates.append({
                    "score": float(score),
                    "text": metadata[idx].get("text", ""),
                    "source": metadata[idx].get("source", "unknown")
                })

        if rerank:
            return self.rerank(query, candidates, top_k=top_k)

        return candidates

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================
    def build_context(self, docs: list[dict[str, Any]]) -> str:

        if not docs:
            return "No relevant context found."

        return "\n\n".join(
            f"[{d.get('source')}]\n{d.get('text')}"
            for d in docs
        )

    # =====================================================
    # PROMPT BUILDER
    # =====================================================
    def build_prompt(self, query: str, context: str) -> str:

        return f"""
You are OmniBioAI Dev Hub Assistant (V6).

Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly, technically, and concisely:
"""

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def answer(self, query: str, repo: str | None = None, bundle: str | None = None):

        docs = self.retrieve(query, repo=repo, bundle=bundle)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, context)

        try:
            response = ollama_generate(prompt)
        except Exception as e:  # noqa: BLE001 -- LLM call boundary: any failure becomes an inline error string instead of crashing the request
            response = f"[LLM_ERROR] {e!s}"

        return {
            "query": query,
            "answer": response,
            "sources": [d.get("source") for d in docs],
            "context": docs,
            "context_used": len(docs),
            "version": "v6-faiss"
        }

    # =====================================================
    # TOKEN STREAMING (Ollama stream=True NDJSON)
    # =====================================================
    def stream_llm(self, query: str, context: str) -> Generator[str, None, None]:
        prompt = self.build_prompt(query, context)
        try:
            res = requests.post(
                f"{OLLAMA_URL}/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
                stream=True,
                timeout=300,
            )
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        except Exception as e:  # noqa: BLE001 -- LLM streaming boundary: any failure becomes an inline error token instead of killing the generator
            yield f"[LLM_ERROR] {e!s}"

    # =====================================================
    # FASTAPI COMPATIBILITY
    # =====================================================
    def query(self, question: str, repo: str | None = None, bundle: str | None = None):
        return self.answer(question, repo=repo, bundle=bundle)
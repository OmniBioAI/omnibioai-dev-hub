import logging
import os
import pickle

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Single canonical dimension for nomic-embed-text.
# If this ever changes, delete the index and re-run build_index.py.
CANONICAL_DIM = 768


class VectorStore:

    def __init__(self):
        self.index = None
        self.metadata = []
        self.dim = None  # None until first vector is inserted; immutable after that

    # ------------------------------------------------------------------
    # Internal: lazy FAISS init
    # ------------------------------------------------------------------

    def _init_index(self, dim: int):
        if dim != CANONICAL_DIM:
            raise ValueError(
                f"VectorStore: refusing to initialize with dim={dim}. "
                f"Only nomic-embed-text ({CANONICAL_DIM}-d) embeddings are supported. "
                f"Delete stale index data and re-run build_index.py."
            )
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        logger.info(f"FAISS IndexFlatIP initialized: dim={dim}")

    # ------------------------------------------------------------------
    # Internal: normalize any input to shape (N, D) float32
    # ------------------------------------------------------------------

    def _coerce_shape(self, vectors) -> np.ndarray:
        vecs = np.array(vectors, dtype=np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim == 3:
            # (N, 1, D) or similar — flatten last two dims into D
            vecs = vecs.reshape(vecs.shape[0], -1)

        if vecs.ndim != 2:
            raise ValueError(
                f"Cannot coerce vectors to (N, D): "
                f"input resolved to shape {vecs.shape}"
            )
        return vecs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vectors, metadata):
        vecs = self._coerce_shape(vectors)

        # Lazy index creation on first insert
        if self.index is None:
            self._init_index(vecs.shape[1])

        if vecs.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch on add(): "
                f"index is {self.dim}-d, got {vecs.shape[1]}-d. "
                f"Do not mix embedding models in the same index."
            )

        self.index.add(vecs)
        self.metadata.extend(metadata)
        logger.debug(f"VectorStore.add: +{len(vecs)} vectors, total={self.index.ntotal}")

    def search(self, query_vec, top_k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return []

        q = self._coerce_shape(query_vec)

        if q.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: index expects {self.dim}-d, "
                f"query is {q.shape[1]}-d. "
                f"Ensure nomic-embed-text is used for both indexing and querying."
            )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0 or i >= len(self.metadata):
                continue
            results.append({
                "score": float(s),
                "text": self.metadata[i].get("text", ""),
                "source": self.metadata[i].get("source", "unknown"),
            })

        return results

    def filter_search(self, query_vec, top_k: int = 5, field: str | None = None, value: str | None = None):
        """FAISS search with post-filtering on a metadata field.

        Retrieves top_k * 3 candidates from FAISS then keeps only those whose
        metadata[field] == value, returning up to top_k results.  Falls back to
        plain search when no filter is specified.
        """
        if field is None or value is None:
            return self.search(query_vec, top_k)

        if self.index is None or self.index.ntotal == 0:
            return []

        q = self._coerce_shape(query_vec)
        if q.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: index expects {self.dim}-d, "
                f"query is {q.shape[1]}-d."
            )

        k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0 or i >= len(self.metadata):
                continue
            meta = self.metadata[i]
            if meta.get(field) != value:
                continue
            results.append({
                "score": float(s),
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "repo": meta.get("repo"),
                "bundle": meta.get("bundle"),
            })
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if self.index is None:
            raise RuntimeError("VectorStore.save: index is empty, nothing to save")
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump({"metadata": self.metadata, "dim": self.dim}, f)
        logger.info(f"VectorStore saved to {directory} ({self.index.ntotal} vectors)")

    def load(self, directory: str) -> bool:
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"VectorStore.load: no saved index found at {directory}")
            return False
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            saved = pickle.load(f)
        self.metadata = saved["metadata"]
        self.dim = saved["dim"]
        logger.info(f"VectorStore loaded from {directory} ({self.index.ntotal} vectors)")
        return True

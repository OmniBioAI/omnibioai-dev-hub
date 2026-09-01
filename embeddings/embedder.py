
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# EMBEDDER V4 - OMNIBIOAI
# RAG-optimized embedding layer
# =========================================================

class Embedder:
    """
    SentenceTransformer wrapper optimized for RAG pipelines.

    V4 upgrades:
    - normalized embeddings (cosine-safe)
    - batch-safe encoding
    - memory-safe large input handling
    - stable output format
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # =========================================================
    # MAIN ENCODE FUNCTION
    # =========================================================
    def encode(self, texts: str | list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Convert text → embeddings
        """

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # batch encoding for memory safety
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # normalize for cosine similarity stability
        embeddings = self._normalize(embeddings)

        return embeddings.tolist()

    # =========================================================
    # NORMALIZATION (CRITICAL FOR RAG QUALITY)
    # =========================================================
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalization for cosine similarity stability
        """

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # avoid division by zero
        norms[norms == 0] = 1e-9

        return embeddings / norms

    # =========================================================
    # UTILITY: SINGLE EMBEDDING
    # =========================================================
    def encode_single(self, text: str) -> list[float]:
        """
        Convenience method for single query embedding
        """

        return self.encode([text])[0]
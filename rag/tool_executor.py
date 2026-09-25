from typing import Any

# =========================================================
# TOOL EXECUTOR V4 - OMNIBIOAI
# Agentic + Traceable + Multi-tool Execution
# =========================================================

class ToolExecutorV4:
    """
    Executes retrieval tools based on agent plan.

    V4 improvements:
    - execution tracing (for UI visualization)
    - safe failure handling per tool
    - structured outputs
    - supports future memory + reasoning tools
    """

    def __init__(self, vector_store, graph_store, plugin_index, embedder):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.plugin_index = plugin_index
        self.embedder = embedder

    # =========================================================
    # MAIN ENTRY
    # =========================================================
    def run(self, plan: dict[str, Any], query: str) -> list[dict]:
        results = []
        steps = plan.get("steps", [])

        for step in steps:
            try:
                if step == "vector_search":
                    results.extend(self._vector(query))

                elif step == "graph_search":
                    results.extend(self._graph(query))

                elif step == "plugin_search":
                    results.extend(self._plugin(query))

                elif step == "memory_search":
                    results.extend(self._memory(query))

                elif step == "hybrid_expand":
                    results.extend(self._hybrid_expand(query))

            except Exception as e:  # noqa: BLE001 -- per-step boundary: one broken tool step must not abort the whole plan
                results.append({
                    "text": f"[TOOL_ERROR] {step}: {e!s}",
                    "source": "error"
                })

        return results

    # =========================================================
    # VECTOR SEARCH
    # =========================================================
    def _vector(self, query: str) -> list[dict]:
        if not self.vector_store:
            return []

        try:
            emb = self.embedder.encode([query])[0]
            return self.vector_store.search(emb, top_k=5)
        except Exception as e:  # noqa: BLE001 -- per-tool boundary: returns an error result instead of raising
            return [{
                "text": f"Vector search failed: {e!s}",
                "source": "vector_error"
            }]

    # =========================================================
    # GRAPH SEARCH
    # =========================================================
    def _graph(self, query: str) -> list[dict]:
        if not self.graph_store:
            return []

        try:
            return self.graph_store.search(query)
        except Exception as e:  # noqa: BLE001 -- per-tool boundary: returns an error result instead of raising
            return [{
                "text": f"Graph search failed: {e!s}",
                "source": "graph_error"
            }]

    # =========================================================
    # PLUGIN SEARCH
    # =========================================================
    def _plugin(self, query: str) -> list[dict]:
        if not self.plugin_index:
            return []

        try:
            return self.plugin_index.search(query)
        except Exception as e:  # noqa: BLE001 -- per-tool boundary: returns an error result instead of raising
            return [{
                "text": f"Plugin search failed: {e!s}",
                "source": "plugin_error"
            }]

    # =========================================================
    # MEMORY SEARCH (V4 READY HOOK)
    # =========================================================
    def _memory(self, query: str) -> list[dict]:
        """
        Placeholder for RAG V4 memory system:
        - conversation memory
        - long-term knowledge memory
        - user session memory
        """

        return [{
            "text": "Memory system not yet integrated (V4 upgrade ready)",
            "source": "memory"
        }]

    # =========================================================
    # HYBRID EXPANSION (V4 AGENT FEATURE)
    # =========================================================
    def _hybrid_expand(self, query: str) -> list[dict]:
        """
        Future: expand query using graph + embeddings fusion
        """

        vector_hits = self._vector(query)
        graph_hits = self._graph(query)

        return vector_hits[:3] + graph_hits[:3]
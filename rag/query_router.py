import json

import requests

# =========================================================
# RAG QUERY ROUTER V4
# =========================================================

class RAGQueryRouterV4:

    def __init__(
        self,
        vector_store,
        graph_store=None,
        plugin_index=None,
        embedder=None
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.plugin_index = plugin_index

        self.embedder = embedder  # FIX

        self.llm_url = "http://localhost:11434/api/chat"
        self.model = "llama3"
        self.top_k = 8

    # -----------------------------
    # INTENT
    # -----------------------------
    def detect_intent(self, query: str) -> str:
        q = query.lower()

        if any(x in q for x in ["graph", "relation", "edge"]):
            return "graph"

        if any(x in q for x in ["plugin", "workflow"]):
            return "plugin"

        return "hybrid"

    # -----------------------------
    # VECTOR SEARCH
    # -----------------------------
    def vector_search(self, query: str) -> list[dict]:

        if not self.vector_store or not self.embedder:
            return []

        try:
            query_vector = self.embedder.encode(query)

            if isinstance(query_vector, list) and len(query_vector) > 0:
                query_vector = query_vector[0]

            return self.vector_store.search(query_vector, top_k=self.top_k)

        except Exception as e:  # noqa: BLE001 -- per-source boundary: a broken subsystem must not break hybrid_retrieve's other sources
            print("[Vector Error]", e)
            return []

    # -----------------------------
    # GRAPH SEARCH
    # -----------------------------
    def graph_search(self, query: str):
        if not self.graph_store:
            return []

        try:
            return self.graph_store.search(query)
        except Exception as e:  # noqa: BLE001 -- per-source boundary: a broken subsystem must not break hybrid_retrieve's other sources
            print("[Graph Error]", e)
            return []

    # -----------------------------
    # PLUGIN SEARCH
    # -----------------------------
    def plugin_search(self, query: str):
        if not self.plugin_index:
            return []

        try:
            return self.plugin_index.search(query)
        except Exception as e:  # noqa: BLE001 -- per-source boundary: a broken subsystem must not break hybrid_retrieve's other sources
            print("[Plugin Error]", e)
            return []

    # -----------------------------
    # HYBRID RETRIEVAL
    # -----------------------------
    def hybrid_retrieve(self, query: str):

        return {
            "intent": self.detect_intent(query),
            "vector": self.vector_search(query),
            "graph": self.graph_search(query),
            "plugin": self.plugin_search(query)
        }

    # -----------------------------
    # CONTEXT BUILDER
    # -----------------------------
    def build_context(self, results: dict):

        blocks = []

        def add(item, tag):
            text = item.get("text", "").strip()
            if text:
                blocks.append(f"[{tag}] {text}")

        for r in results["vector"]:
            add(r, "VECTOR")

        for r in results["graph"]:
            add(r, "GRAPH")

        for r in results["plugin"]:
            add(r, "PLUGIN")

        return "\n".join(blocks[:12])

    # -----------------------------
    # STREAM LLM
    # -----------------------------
    def stream_llm(self, query: str, context: str):

        prompt = f"""
You are OmniBioAI RAG Assistant (V4).

CONTEXT:
{context}

QUESTION:
{query}
"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        try:
            response = requests.post(
                self.llm_url,
                json=payload,
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, KeyError) as e:
                    print("[Stream Parse Error]", e)
                    continue

        except Exception as e:  # noqa: BLE001 -- streaming boundary: any failure becomes an inline error token instead of killing the generator
            yield f"[STREAM_ERROR] {e!s}"

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def query(self, query: str):

        results = self.hybrid_retrieve(query)
        context = self.build_context(results)

        output = ""

        for token in self.stream_llm(query, context):
            output += token

        return {
            "intent": results["intent"],
            "answer": output,
            "context": context
        }


# =========================================================
# ENGINE SINGLETON (FIXED - OUTSIDE CLASS)
# =========================================================

_ENGINE = None


def init_engine(vector_store, graph_store=None, plugin_index=None, embedder=None):
    global _ENGINE

    _ENGINE = RAGQueryRouterV4(
        vector_store=vector_store,
        graph_store=graph_store,
        plugin_index=plugin_index,
        embedder=embedder
    )


def get_engine():
    if _ENGINE is None:
        raise RuntimeError("RAG engine not initialized")
    return _ENGINE
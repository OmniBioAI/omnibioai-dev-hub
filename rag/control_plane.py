import threading
import time
from dataclasses import dataclass, field
from typing import Any

# =========================================================
# CONTROL PLANE STATE
# =========================================================

@dataclass
class ControlPlaneState:
    status: str = "INIT"   # INIT | BUILDING | READY | FAILED
    started_at: float = field(default_factory=time.time)
    last_error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


# =========================================================
# CONTROL PLANE (VERSION-AGNOSTIC V5)
# =========================================================

class ControlPlane:

    def __init__(self):
        self.state = ControlPlaneState()
        self.lock = threading.Lock()

        self._initialized = False

        # runtime components
        self.vector_store = None
        self.graph_store = None
        self.plugin_index = None

        # engine
        self.engine = None

    # =========================================================
    # INIT
    # =========================================================

    def init(self, vector_store, graph_store, plugin_index):

        with self.lock:
            if self._initialized:
                return

            self.state.status = "BUILDING"

        try:
            self.vector_store = vector_store
            self.graph_store = graph_store
            self.plugin_index = plugin_index

            self.state.metrics["init_time"] = time.time()

            self._build()
            self._init_engine()

            with self.lock:
                self.state.status = "READY"
                self._initialized = True

        except Exception as e:
            with self.lock:
                self.state.status = "FAILED"
                self.state.last_error = str(e)
            raise

    # =========================================================
    # ENGINE INIT (FIXED)
    # =========================================================

    def _init_engine(self):

        from rag.engine import RAGEngine

        # match actual constructor signature
        self.engine = RAGEngine(self.vector_store)

    # =========================================================
    # BUILD PIPELINE
    # =========================================================

    def _build(self):

        start = time.time()

        self._build_graph_seed()
        self._build_plugin_seed()

        self.state.metrics["build_time_ms"] = round(
            (time.time() - start) * 1000, 2
        )

    # =========================================================
    # GRAPH SEED
    # =========================================================

    def _build_graph_seed(self):

        if not self.graph_store:
            return

        self.graph_store.add_edge("OmniBioAI", "RAG Engine", "powers")
        self.graph_store.add_edge("RAG Engine", "Vector Store", "uses")
        self.graph_store.add_edge("RAG Engine", "Graph Store", "augments")
        self.graph_store.add_edge("Plugin System", "Workflow Engine", "drives")

    # =========================================================
    # PLUGIN SEED
    # =========================================================

    def _build_plugin_seed(self):

        if not self.plugin_index:
            return

        self.plugin_index.docs = [
            {"text": "workflow_builder manages DAG pipelines", "plugin": "workflow_builder"},
            {"text": "plugin_executor runs bioinformatics workflows", "plugin": "plugin_executor"},
            {"text": "qc_plots generates sequencing QC reports", "plugin": "qc_plots"},
        ]

    # =========================================================
    # STATUS
    # =========================================================

    def status(self):

        return {
            "status": self.state.status,
            "uptime_sec": round(time.time() - self.state.started_at, 2),
            "metrics": self.state.metrics,
            "error": self.state.last_error,
            "initialized": self._initialized,
            "engine_ready": self.engine is not None
        }

    # =========================================================
    # SAFETY GATE
    # =========================================================

    def ensure_ready(self):

        if self.state.status != "READY":
            raise RuntimeError(
                f"Control plane not ready: {self.state.status}"
            )

    # =========================================================
    # ENGINE ACCESS
    # =========================================================

    def get_engine(self):

        if self.engine is None:
            raise RuntimeError("RAG engine not initialized")

        return self.engine


# =========================================================
# SINGLETON
# =========================================================

CONTROL_PLANE = ControlPlane()
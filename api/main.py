import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import validate_auth_config
from api.routes import rag
from index.graph_store import GraphStore
from index.plugin_index import PluginIndex
from index.vector_store import VectorStore
from rag.control_plane import CONTROL_PLANE

logger = logging.getLogger(__name__)


# =========================================================
# APP INIT
# =========================================================

app = FastAPI(title="OmniBioAI Dev Hub Control Plane V5")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL STATE
# =========================================================

vector_store = VectorStore()
_INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "faiss_index")
loaded = vector_store.load(_INDEX_DIR)
if not loaded:
    logger.warning("No persisted FAISS index found; will build from scratch on startup")
graph_store = GraphStore()
plugin_index = PluginIndex([])

# =========================================================
# GRAPH SEED
# =========================================================

def build_graph_seed():
    graph_store.add_edge("OmniBioAI", "RAG Engine", "powers")
    graph_store.add_edge("RAG Engine", "Vector Store", "uses")
    graph_store.add_edge("RAG Engine", "Graph Store", "augments")
    graph_store.add_edge("Plugin System", "Workflow Engine", "drives")


# =========================================================
# PLUGIN SEED
# =========================================================

def build_plugin_index():
    plugin_index.docs = [
        {"text": "workflow_builder manages DAG pipelines", "plugin": "workflow_builder"},
        {"text": "plugin_executor runs bioinformatics workflows", "plugin": "plugin_executor"},
        {"text": "qc_plots generates sequencing QC reports", "plugin": "qc_plots"}
    ]


# =========================================================
# CONTROL PLANE INIT
# =========================================================

async def init_control_plane():
    # Seeding is owned by CONTROL_PLANE._build() to avoid double-seeding.
    # build_graph_seed() and build_plugin_index() are kept for test compatibility
    # but are not called here.
    CONTROL_PLANE.init(
        vector_store=vector_store,
        graph_store=graph_store,
        plugin_index=plugin_index,
    )


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():
    # Fail loudly and refuse to start rather than silently serving a
    # deployment that looks auth-protected but isn't -- see api/auth.py's
    # validate_auth_config() for why.
    validate_auth_config()
    await init_control_plane()


# =========================================================
# ROUTES
# =========================================================

app.include_router(rag.router, prefix="/rag")


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "omnibioai-dev-hub",
        "control_plane": CONTROL_PLANE.status()
    }


# =========================================================
# STATUS
# =========================================================

@app.get("/status")
def status():
    repos_loaded = 0
    if vector_store.metadata:
        repo_names: set = set()
        for m in vector_store.metadata:
            for part in m.get("source", "").split("/"):
                if part.startswith("omnibioai"):
                    repo_names.add(part)
                    break
        repos_loaded = len(repo_names)

    return {
        "control_plane": CONTROL_PLANE.status(),
        "index_vectors": vector_store.index.ntotal if vector_store.index else 0,
        "graph_edges": graph_store.size()["edges"],
        "plugins_loaded": len(plugin_index.docs),
        "repos_loaded": repos_loaded,
    }


# =========================================================
# SAFETY MIDDLEWARE (FIXED)
# =========================================================

@app.middleware("http")
async def guard_requests(request: Request, call_next):

    if request.url.path.startswith("/rag") and CONTROL_PLANE.status()["status"] != "READY":
        return JSONResponse(
            status_code=503,
            content={"detail": "Control plane not ready"}
        )

    return await call_next(request)
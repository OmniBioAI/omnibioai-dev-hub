# OmniBioAI Dev Hub — RAG V6

Production-grade Retrieval-Augmented Generation (RAG) system powering the OmniBioAI ecosystem documentation, architecture search, workflow discovery, and developer assistant APIs.

---

# Features

* FAISS-native vector search (IndexFlatIP, 768-dim)
* Incremental per-repo indexing with hash-based dedup
* Ollama local embeddings (`nomic-embed-text`) + local LLM inference (`llama3`)
* FastAPI API server
* Real token-level SSE streaming via Ollama `stream: true`
* Chunk-level document retrieval with source attribution
* Repository-wide multi-project indexing (19 repos)
* Fully local execution — no OpenAI dependency
* Production-safe embedding normalization
* V6 dimension consistency enforcement

---

# Architecture

```text
Repositories  (REPO_BASE/omnibioai-*)
     ↓
Document Loader  (ingestion/doc_loader.py)
     ↓
Chunker  (processing/chunker.py, 500-word windows / 2000-char max)
     ↓
Ollama Embeddings  (nomic-embed-text, 768-d, normalized)
     ↓
FAISS Vector Index  (IndexFlatIP, data/faiss_index/)
     ↓
RAG Engine  (rag/engine.py)
     ↓
FastAPI API  (api/main.py + api/routes/rag.py)
     ↓
LLM Answer Generation  (llama3 via Ollama, blocking or token-streamed)
```

---

# V6 Major Improvements

## FAISS Native Retrieval

Previous versions used brute-force cosine scanning across vectors.

V6 uses:

```python
faiss.IndexFlatIP
```

Benefits:

* 10–50x faster retrieval
* scalable search
* lower latency
* future ANN support

---

## Embedding Consistency Fix

A major issue in previous builds was embedding mismatch.

### Old Problem

| Stage    | Model            | Dimension |
| -------- | ---------------- | --------- |
| Indexing | all-MiniLM-L6-v2 | 384       |
| Querying | nomic-embed-text | 768       |

This caused FAISS assertion failures:

```python
AssertionError: d == self.d
```

### V6 Fix

Both ingestion (`scripts/build_index.py`) and retrieval (`rag/engine.py`) now call the same `ollama_embed("nomic-embed-text")` function — the single source of truth.

`embeddings/embedder.py` (sentence-transformers, 384-dim) is retained for test coverage only and is not on any live request path.

---

# Repository Structure

```text
omnibioai-dev-hub/
│
├── api/
│   ├── main.py              # FastAPI app, startup, /status, /health
│   ├── auth.py               # JWT dependency for /rag/*, gated by AUTH_ENABLED
│   │                          # — see "Authentication" below
│   ├── middleware/
│   │   └── trace.py           # TraceContext — written but not wired into
│   │                          # main.py or tested; not on any live request path
│   └── routes/
│       └── rag.py           # /rag/query and /rag/stream endpoints (auth-gated)
│
├── rag/
│   ├── engine.py            # RAGEngine: retrieve, build_context, answer, stream_llm
│   ├── control_plane.py     # Singleton lifecycle manager
│   ├── query_router.py      # RAGQueryRouterV4 — superseded agentic router,
│   ├── tool_executor.py     # ToolExecutorV4 — from a prior "V4" architecture,
│   └── memory_store.py      # MemoryStoreV4 — tests only (see tests/test_rag_*),
│                              # not on any live request path in the current V6 engine
│
├── index/
│   ├── vector_store.py      # FAISS wrapper (add, search, save, load)
│   ├── graph_store.py       # In-memory knowledge graph (BFS expansion)
│   └── plugin_index.py      # Plugin doc registry
│
├── embeddings/
│   └── embedder.py          # SentenceTransformer wrapper (tests only, not live)
│
├── retrieval/
│   └── retriever.py         # Retriever class (tests only, not live)
│
├── ingestion/
│   └── doc_loader.py        # Markdown document loader with SKIP_DIRS/SKIP_PATH_SEGMENTS
│
├── processing/
│   └── chunker.py           # 500-word / 2000-char chunker
│
├── scripts/
│   ├── build_index.py       # Index builder entry point
│   ├── ingest.py             # Standalone ingestion helper
│   ├── run_eval.py           # Recall@K eval harness (tests/eval/)
│   └── check_and_reindex.sh  # Rebuilds the index on new Studio releases (hourly cron)
│
├── data/
│   └── faiss_index/         # index.faiss + metadata.pkl (gitignored)
│
├── configs/
│   └── repos.yaml           # Repo list — actual source of truth build_index.py reads
│
├── omnibioai-dev-hub-ui/     # Dev Hub UI (React + TypeScript) — see "Frontend" below
│
└── .env.example             # Environment variable template
```

---

# Frontend

`omnibioai-dev-hub-ui/` is a React + TypeScript UI (Vite, served at
`/_svc/devhub`), built and shipped alongside the API — not a separate
repo. The `Dockerfile`'s `ui-builder` stage builds it and copies the
static output into the same image the FastAPI backend runs in
(`EXPOSE 8082 5173` — API and UI dev server ports both exposed).

```bash
cd omnibioai-dev-hub-ui
npm install
npm run dev       # Vite dev server on :5173, proxies /api, /rag, /health to :8082
npm run build      # production build
```

The dev server's proxy (`vite.config.ts`) rewrites `/api/*` to the FastAPI
backend at `http://127.0.0.1:8082` (stripping the `/api` prefix) and
forwards `/rag/*` and `/health` there directly — so `AUTH_ENABLED` on the
backend (see "Authentication" below) applies to UI-issued requests the
same as any other client.

---

# Requirements

## Python

Recommended:

```text
Python 3.11 or 3.12
```

> **Note:** Python 3.13 removes `numpy.distutils`, breaking `faiss-cpu`. Use 3.11 or 3.12. In Docker the provided image uses Python 3.12.

---

## Ollama

Install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Required Ollama Models

### Embedding Model

```bash
ollama pull nomic-embed-text
```

### Generation Model

The system uses `llama3` by default (hardcoded in `rag/engine.py`):

```bash
ollama pull llama3
```

Optional alternatives (change `model=` in `engine.py` if needed):

```bash
ollama pull mistral
ollama pull deepseek-coder
```

---

# Installation

## Create Environment

```bash
conda create -n omnibioai-dev-hub python=3.12 -y
conda activate omnibioai-dev-hub
```

---

## Install Dependencies

```bash
pip install fastapi uvicorn requests numpy faiss-cpu sentence-transformers
```

> `sentence-transformers` is required for cross-encoder reranking (`rerank=True` in `retrieve()`). It is also used by the test suite. If reranking is not needed you can omit it — the engine degrades gracefully to FAISS order when the model is unavailable.

---

# Configuration

## REPO_BASE

`REPO_BASE` is the only required configuration. It must point to the directory that contains the `omnibioai-*` repos as immediate children.

```bash
export REPO_BASE=/home/manish/Desktop/machine
```

The indexer **exits immediately with a clear error** if no repos are found under `REPO_BASE`:

```text
❌  No repos found under REPO_BASE='/some/wrong/path'
    Set REPO_BASE to the directory that contains the omnibioai-* repos.
    Example:  export REPO_BASE=/home/manish/Desktop/machine
    Docker:   -e REPO_BASE=/repos  (with repos volume mounted at /repos)
```

In Docker the image sets `ENV REPO_BASE=/repos` automatically — no action needed.

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
# edit REPO_BASE as needed
```

---

# Build Index

## Clean Existing Data

```bash
rm -rf data/faiss_index/*
```

---

## Build V6 Index

```bash
python scripts/build_index.py
```

Expected output:

```text
🚀 Incremental V6 Indexing Starting...
⚠️  2 repos not found, will be skipped: ['omnibioai-security-audit', 'omnibioai-hpc-policy-engine']
📄 Loaded N documents
...
💾 Index saved to .../data/faiss_index (10877 vectors)
✅ V6 Index Complete
{'too_short': 4, 'deduped': 31, 'embed_failed': 4, 'new': 10881, 'chunks_indexed': 10877}
```

### How deduplication works

`seen_hashes` is **reset per repo** so cross-repo identical chunks each get their own index entry under their canonical source path. Within a single repo, duplicate chunks (e.g. shared boilerplate across plugin READMEs) are deduplicated.

Chunks shorter than 10 characters are discarded (`MIN_CHUNK_CHARS = 10`) to eliminate overflow tails produced by the hard character-slice in `chunker.py`.

### Excluded paths

`ingestion/doc_loader.py` skips the following during the walk:

**By directory name (`SKIP_DIRS`):**

```python
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", "obsolete"}
```

Note: both `.venv` and bare `venv` are excluded. `.pytest_cache` is excluded because it contains auto-generated `README.md` stubs (present in 17 of the 19 repos) that would otherwise pollute the index with boilerplate.

**By path segment (`SKIP_PATH_SEGMENTS`):**

```python
SKIP_PATH_SEGMENTS = {"work"}
```

This excludes `omnibioai/work/` which contains UUID-named runtime copies of workflow bundle READMEs. Without this exclusion, those copies would claim index slots before the canonical `omnibioai-workflow-bundles/` paths are processed, causing all 50 bundles to appear indexed under `omnibioai/work/` instead.

---

# Current Index Stats

As of 2026-06-14 (Phase 3 — metadata filtering + cross-encoder reranking).
**Not re-verified since** — `omnibioai-security-audit` and
`omnibioai-hpc-policy-engine` (both listed as "not present on disk" below
and in this table's "17 of 19") now exist on disk, so a rebuild today
would very likely index closer to 19 of 19 and produce different vector
counts than shown here. Regenerate with `python scripts/build_index.py`
for current numbers before relying on this table.

| Metric | Value |
|--------|-------|
| Total vectors | 10,877 |
| Unique source files | ~965 |
| Repos indexed | 17 of 19 |
| Workflow bundles covered | 50 / 50 |
| Chunks filtered (too short) | 4 |
| Cross-repo deduped | 31 |
| **Recall@5 (eval set)** | **96.8% (30/31)** |
| Eval set size | 31 queries |

**Chunk metadata fields:** Each indexed chunk carries `text`, `source`, `hash`, `repo` (basename of the containing repository directory), and `bundle` (first subdirectory under the repo root, or `None` for root-level files). The `repo` and `bundle` fields enable post-filtered scoped queries that restrict FAISS candidates to a specific repository or workflow bundle.

**Chunking strategy:** Markdown-structure-aware — splits on H1/H2/H3 headers as natural section boundaries, never splits inside fenced code blocks, falls back to paragraph boundaries (`\n\n`) for long sections, and word-boundary splitting for oversized paragraphs. Each chunk is prefixed with its ancestor header breadcrumb (e.g. `# ATACseq Pipeline > ## Parameters`) so retrieval context carries section identity. Previous fixed 500-word window chunker produced 2,067 vectors; the markdown chunker produces 10,877 (5.3× more granular chunks).

**Remaining failure:** The one unscoped failure (metagenomics shotgun profiling vs. microbiome/kraken sub-workflows) passes with either `bundle="metagenomics"` scoped filtering or `rerank=True` cross-encoder reranking.

At the time these stats were captured, `omnibioai-security-audit` and
`omnibioai-hpc-policy-engine` were not present on disk and the indexer
skipped them with a warning. **Both now exist** — the "skipped" comments
on those two entries in [Supported Repositories](#supported-repositories)
below are stale; a rebuild today should index them like any other repo,
gracefully, no code change needed (the indexer's on-disk check is dynamic,
not a hardcoded list).

> **Note:** `data/faiss_index/` is excluded from git (see `.gitignore`). Regenerate with `python scripts/build_index.py`.

---

# Run API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8082 --reload
```

In Docker the server starts automatically as PID 1.

---

# Authentication

`/rag/query` and `/rag/stream` are gated by a JWT bearer-token dependency
(`api/auth.py`), off by default and controlled by `AUTH_ENABLED`:

```bash
export AUTH_ENABLED=true
export JWT_SECRET=...   # HS256 secret, must match omnibioai-auth's SECRET_KEY
```

`AUTH_ENABLED=false` (the default for local/`uvicorn --reload` use) runs
both endpoints open, no token required — this is what the curl examples
below assume. **`omnibioai-studio`'s deployed compose config sets
`AUTH_ENABLED=true`**, so against a real deployment both endpoints require:

```bash
curl -X POST http://localhost:8082/rag/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt-token>" \
  -d '{"query":"..."}'
```

A missing/invalid/expired token returns `401`. This follows the same
shared-secret pattern as `omnibioai-model-registry`'s `require_auth`.

---

# Test Query API

```bash
curl -X POST http://localhost:8082/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is workflow engine in OmniBioAI?"}'
```

> Add `-H "Authorization: Bearer <jwt-token>"` if the server you're
> talking to has `AUTH_ENABLED=true` — see "Authentication" above.

Example response:

```json
{
  "query": "What is workflow engine in OmniBioAI?",
  "answer": "According to the provided context...",
  "sources": [
    "/repos/omnibioai-workflow-bundles/README.md"
  ],
  "context": [
    {"score": 0.87, "text": "...", "source": "/repos/omnibioai-workflow-bundles/README.md"}
  ],
  "context_used": 5,
  "version": "v6-faiss",
  "api_version": "v6"
}
```

---

## Scoped Queries

Both `/rag/query` and `/rag/stream` accept optional `repo` and `bundle` parameters to post-filter FAISS candidates to a specific repository or workflow bundle:

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo` | `string` (optional) | Restrict results to chunks from this repository (e.g. `"omnibioai-model-registry"`) |
| `bundle` | `string` (optional) | Restrict results to chunks from this workflow bundle subdirectory (e.g. `"metagenomics"`) |

When both are provided, `bundle` takes priority. Omitting both parameters performs a global unscoped search.

```bash
curl -X POST http://localhost:8082/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "shotgun profiling steps", "bundle": "metagenomics"}'
```

---

# Streaming API

Endpoint:

```text
POST /rag/stream
```

Body: `{"query": "your question"}` — also accepts optional `repo` and `bundle` parameters (same semantics as `/rag/query`).

### How it works

1. The server retrieves the top-5 chunks from FAISS (same path as `/rag/query`).
2. It builds the prompt from the retrieved context.
3. It calls Ollama's `/api/generate` with `"stream": true`.
4. Ollama returns newline-delimited JSON (NDJSON), one object per token:
   ```json
   {"model":"llama3","response":"Based","done":false}
   {"model":"llama3","response":" on","done":false}
   ...
   {"model":"llama3","response":"","done":true}
   ```
5. Each token is immediately forwarded as an SSE event:
   ```text
   data: {"type": "token", "content": "Based"}

   data: {"type": "token", "content": " on"}

   ...

   data: {"type": "done"}
   ```

### Client-side consumption

```typescript
ragStream(
  "your query",
  (token) => appendToUI(token),      // called per token
  () => markComplete(),               // called on done
  (err) => showError(err)            // called on error
);
```

The UI client (`src/api/client.ts`) parses the `data:` envelope and dispatches `type: "token"` and `type: "done"` events.

---

# Reranking

## Cross-Encoder Reranking

The retrieval pipeline supports optional cross-encoder reranking:

```python
docs = engine.retrieve(query, top_k=5, rerank=True)
```

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence-transformers`)

**How it works:**

1. FAISS retrieves `top_k × 3` candidates (15 for the default `top_k=5`).
2. The cross-encoder scores each `(query, chunk)` pair jointly, capturing fine-grained relevance that the bi-encoder embedding alone misses.
3. The top `top_k` by cross-encoder score are returned, each with a `ce_score` field attached.

**Graceful degradation:** If `sentence-transformers` is not installed or the model fails to load, `rerank()` logs a warning and falls back to the original FAISS order silently. No code change needed.

**When to use:** Unscoped queries where semantic collision is likely — for example, when a broad term (e.g. "Kraken2") matches many sub-workflow chunks and the pipeline overview you actually want is pushed below rank 5. For narrowly scoped queries (with `repo` or `bundle`), the candidate pool is already restricted and reranking adds latency with little benefit.

---

# Supported Repositories

The indexer targets 19 repositories. All paths are relative to `REPO_BASE`:

```python
repos = [
    "omnibioai",                  # main platform repo
    "omnibioai-rag",
    "omnibioai-toolserver",
    "omnibioai-sdk",
    "omnibioai-workflow-bundles",  # 50 workflow bundle subdirectories
    "omnibioai-control-center",
    "omnibioai-lims",
    "omnibioai-model-registry",
    "omnibioai-dev-docker",
    "omnibioai-api-gateway",
    "omnibioai-docs",
    "omnibioai-studio",
    "omnibioai-auth",
    "omnibioai-tool-runtime",
    "omnibioai-iam-client",
    "omnibioai-policy-engine",
    "omnibioai-security-sdk",
    "omnibioai-security-audit",
    "omnibioai-hpc-policy-engine",
]
```

`omnibioai-security-audit` and `omnibioai-hpc-policy-engine` were absent
on disk when [Current Index Stats](#current-index-stats) above was last
captured (2026-06-14) — that table's "skipped with warning" framing
reflects that snapshot, not necessarily today's state. Either way, a
missing repo is skipped gracefully at build time and needs no code
change to pick up once it appears — the indexer checks the filesystem at
run time, not a hardcoded present/absent list.

---

# V6 Retrieval Pipeline

## Step 1 — Chunking

Documents are split using a 500-word sliding window, hard-capped at 2000 characters per chunk. Chunks shorter than 10 characters are discarded.

> **Known limitation:** The hard character slice at 2000 chars can produce 1–2 word overflow fragments at word boundaries (e.g. "onment", "abases"). These are above the 10-char filter threshold. A future fix will snap the slice to the nearest word boundary. See *Future Work* below.

---

## Step 2 — Embedding

Each chunk is embedded using:

```text
nomic-embed-text  (768-dim, L2-normalized)
```

---

## Step 3 — FAISS Indexing

Vectors are stored in:

```python
faiss.IndexFlatIP
```

Pre-normalized vectors make inner product equivalent to cosine similarity.

---

## Step 4 — Query Embedding

User query is embedded using the same `nomic-embed-text` model at query time.

---

## Step 5 — Vector Search

FAISS retrieves the top-5 nearest chunks (configurable via `top_k`).

---

## Step 6 — Prompt Assembly

Retrieved chunks become the context block in a structured prompt.

---

## Step 7 — LLM Generation

Prompt sent to local Ollama `llama3` model, either blocking (`/rag/query`) or token-streamed (`/rag/stream`).

---

# Performance

## Before V6

* brute-force cosine scan
* slow retrieval
* embedding dimension mismatch
* unstable indexing
* global dedup silencing canonical paths

## After V6

* FAISS-native inner product retrieval
* stable 768-dim pipeline end-to-end
* per-repo dedup with canonical path preservation
* real SSE token streaming
* local-only execution

---

# Troubleshooting

## REPO_BASE not set or wrong path

Symptom:

```text
❌  No repos found under REPO_BASE='/repos'
```

Cause: `REPO_BASE` defaults to `/home/manish/Desktop/machine` locally and `/repos` in Docker. If neither is correct, set it explicitly:

```bash
export REPO_BASE=/path/to/parent/of/omnibioai-repos
python scripts/build_index.py
```

---

## Index loads but all queries return empty results

Cause: The index on disk was built with a different embedding model or dimension. The loaded vectors won't match query vectors.

Fix: Delete and rebuild:

```bash
rm data/faiss_index/index.faiss data/faiss_index/metadata.pkl
python scripts/build_index.py
```

---

## FAISS Dimension Mismatch

Error:

```text
AssertionError: d == self.d
```

Cause: Different embedding models used during indexing vs querying (the pre-V6 problem). Rebuild the index — V6 enforces 768-dim at both stages.

---

## Ollama Timeout

Error:

```text
Read timed out
```

Fix: Use a smaller generation model. Change the `model=` argument in `rag/engine.py`:

```python
model="mistral"  # faster than llama3 on smaller hardware
```

---

## Empty Retrieval Results

Verify the index loaded correctly:

```bash
python -c "
from index.vector_store import VectorStore
vs = VectorStore()
vs.load('data/faiss_index')
print('ntotal:', vs.index.ntotal if vs.index else 'no index')
"
```

Expected: `ntotal: 2067` (or your current count).

---

# Known Limitations / Future Work

## Chunker word-wrap (planned)

`chunker.py` slices at a hard 2000-character boundary without snapping to word boundaries. This produces 1–2 word fragments at the tail of some documents (e.g. "onment", "abases" — 16–18 chars, above the `MIN_CHUNK_CHARS=10` filter). These fragments are harmless but pollute the index with low-information chunks. The fix is to snap the slice to the nearest preceding space.

## Planned V7 Features

* Chunker word-boundary snapping
* IVF or HNSW indexes for million-scale corpora
* Hybrid BM25 + vector search
* Persistent storage and distributed / incremental index updates
* Graph RAG (graph store already seeded)
* Plugin-aware retrieval

---

# License

Internal OmniBioAI Development License.

---

# OmniBioAI Ecosystem

RAG V6 powers:

* architecture discovery
* workflow documentation search
* plugin documentation retrieval
* developer assistant APIs
* AI infrastructure exploration
* cross-repository semantic search
* internal engineering copilots

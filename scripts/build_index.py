import sys
import os
import hashlib
import numpy as np
import yaml

sys.path.append(os.path.abspath("."))

from index.vector_store import VectorStore
from rag.engine import ollama_embed   # SINGLE SOURCE OF TRUTH
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text

MIN_CHUNK_CHARS = 10  # discard overflow tails from chunker.py's hard char-slice

REPOS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "repos.yaml"
)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_repo_names_from_config(path: str = REPOS_CONFIG_PATH):
    """Read the `repos:` name list from configs/repos.yaml.

    Returns the list of names on success, or None if the file is missing,
    empty, fails to parse, or doesn't define a non-empty `repos:` list. In
    every one of those cases the caller falls back to the hardcoded list in
    build_index(), so behavior is unchanged unless the YAML is actually
    populated and wired in.
    """
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"⚠️  Could not read {path} ({e}); falling back to hardcoded repo list")
        return None

    if not isinstance(data, dict):
        return None

    names = data.get("repos")
    if not names or not isinstance(names, list):
        return None

    return list(names)


def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)

    if vec.ndim == 3:
        vec = vec.squeeze()

    if vec.ndim == 2:
        vec = vec.reshape(-1)

    return vec.astype(np.float32)


def build_index():

    print("🚀 Incremental V6 Indexing Starting...")

    vector_store = VectorStore()

    # BUG-1 FIX: default to the actual local machine path.
    # In Docker the image sets ENV REPO_BASE=/repos (volume mount).
    # Locally: export REPO_BASE=/home/manish/Desktop/machine (or set in .env).
    REPO_BASE = os.environ.get("REPO_BASE", "/home/manish/Desktop/machine")

    # Hardcoded fallback -- kept in place (not deleted) so behavior is
    # unchanged if configs/repos.yaml is missing, empty, or fails to parse.
    HARDCODED_REPO_NAMES = [
        "omnibioai",
        "omnibioai-rag",
        "omnibioai-toolserver",
        "omnibioai-sdk",
        "omnibioai-workflow-bundles",
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
        "omnibioai-security-audit",
        "omnibioai-security-sdk",
        "omnibioai-hpc-policy-engine",
    ]

    repo_names = _load_repo_names_from_config() or HARDCODED_REPO_NAMES
    repos = [f"{REPO_BASE}/{name}" for name in repo_names]

    # Fail fast: if not a single repo exists, the REPO_BASE is wrong.
    existing = [r for r in repos if os.path.isdir(r)]
    if not existing:
        raise SystemExit(
            f"\n❌  No repos found under REPO_BASE={REPO_BASE!r}\n"
            f"    Set REPO_BASE to the directory that contains the omnibioai-* repos.\n"
            f"    Example:  export REPO_BASE=/home/manish/Desktop/machine\n"
            f"    Docker:   -e REPO_BASE=/repos  (with repos volume mounted at /repos)\n"
        )

    missing = [os.path.basename(r) for r in repos if not os.path.isdir(r)]
    if missing:
        print(f"⚠️  {len(missing)} repos not found, will be skipped: {missing}")

    all_vectors = []
    all_meta = []

    stats = {"too_short": 0, "deduped": 0, "embed_failed": 0, "new": 0, "chunks_indexed": 0}

    # BUG-2 FIX: seen_hashes is reset per repo so cross-repo identical chunks
    # are each indexed under their own source path.  Within a single repo,
    # dedup still fires to avoid storing the same paragraph twice (e.g. a
    # shared boilerplate section that appears in multiple plugin READMEs).
    for repo_path in repos:
        if not os.path.isdir(repo_path):
            continue

        repo_name = os.path.basename(repo_path)
        seen_hashes: set = set()
        repo_docs = load_documents([repo_path])

        for doc in repo_docs:

            text = doc.get("text", "")
            if not text:
                continue

            source = doc.get("source", "unknown")

            # Extract the first subdirectory under the repo root as the bundle.
            # Files at the repo root (e.g. README.md) get bundle=None.
            try:
                rel = os.path.relpath(source, repo_path)
                parts = rel.replace("\\", "/").split("/")
                bundle_name = parts[0] if len(parts) > 1 else None
            except ValueError:
                bundle_name = None

            for chunk in chunk_text(text):

                if len(chunk) < MIN_CHUNK_CHARS:
                    stats["too_short"] += 1
                    continue

                h = hash_text(chunk)

                if h in seen_hashes:
                    stats["deduped"] += 1
                    continue

                seen_hashes.add(h)
                stats["new"] += 1

                try:
                    vec = ollama_embed(chunk)   # <<< SINGLE EMBEDDING SOURCE
                    vec = normalize_vector(vec)
                except Exception as e:
                    print(f"⚠️  Skipping chunk from {source}: {e}")
                    stats["embed_failed"] += 1
                    continue

                all_vectors.append(vec)
                all_meta.append({
                    "text": chunk,
                    "source": source,
                    "hash": h,
                    "repo": repo_name,
                    "bundle": bundle_name,
                })

                stats["chunks_indexed"] += 1

    if not all_vectors:
        print("⚠️ No vectors generated")
        return

    all_vectors = np.vstack(all_vectors).astype(np.float32)

    vector_store.add(all_vectors, all_meta)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "faiss_index")
    vector_store.save(save_path)
    print(f"💾 Index saved to {save_path} ({vector_store.index.ntotal} vectors)")

    print("✅ V6 Index Complete")
    print(stats)


if __name__ == "__main__":
    build_index()
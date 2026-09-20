import os
import sys

import yaml

sys.path.append(os.path.abspath("."))

from scripts.build_trusted_index import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBED_COOLDOWN_SECONDS,
    build_candidate,
)

REPOS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "repos.yaml"
)


def _load_repo_names_from_config(path: str = REPOS_CONFIG_PATH):
    """Read the `repos:` name list from configs/repos.yaml.

    Returns the list of names on success, or None if the file is missing,
    empty, fails to parse, or does not define a non-empty `repos:` list. In
    every one of those cases the caller falls back to the hardcoded list in
    build_index(), preserving the pre-Phase-18 config/fallback behavior.
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


def build_index():
    """Build a trusted candidate index without overwriting the current index.

    Phase 18 changes this active entry point from direct writes to
    data/faiss_index into a staging build under data/faiss_candidates/<build>.
    Promotion and rollback are explicit lifecycle operations.
    """

    print("🚀 Trusted V7 Candidate Indexing Starting...")

    repo_base = os.environ.get("REPO_BASE", "/home/manish/Desktop/machine")

    hardcoded_repo_names = [
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
        "omnibioai-workbench",
        "omnibioai-tes",
        "omnibioai-dev-hub",
    ]

    repo_names = _load_repo_names_from_config() or hardcoded_repo_names
    repos = [os.path.join(repo_base, name) for name in repo_names]

    existing = [r for r in repos if os.path.isdir(r)]
    if not existing:
        raise SystemExit(
            f"\n❌  No repos found under REPO_BASE={repo_base!r}\n"
            f"    Set REPO_BASE to the directory that contains the omnibioai-* repos.\n"
            f"    Example:  export REPO_BASE=/home/manish/Desktop/machine\n"
            f"    Docker:   -e REPO_BASE=/repos  (with repos volume mounted at /repos)\n"
        )

    missing = [os.path.basename(r) for r in repos if not os.path.isdir(r)]
    if missing:
        print(f"⚠️  {len(missing)} repos not found, will be recorded as missing: {missing}")

    staging_root = os.environ.get(
        "DEVHUB_STAGING_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "faiss_candidates"),
    )
    build_id = os.environ.get("DEVHUB_BUILD_ID")
    embed_batch_size = int(os.environ.get("DEVHUB_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE))
    embed_cooldown_seconds = float(os.environ.get("DEVHUB_EMBED_COOLDOWN_SECONDS", DEFAULT_EMBED_COOLDOWN_SECONDS))
    result = build_candidate(
        repo_base, staging_root, build_id=build_id, repo_names=repo_names,
        batch_size=embed_batch_size, cooldown_seconds=embed_cooldown_seconds,
    )
    manifest = result["manifest"]
    validation = result["validation"]
    print(f"💾 Candidate index saved to {result['candidate_dir']}")
    print(f"✅ Trusted candidate complete: {manifest['build_id']}")
    print({
        "configured_for_ingestion": len(manifest["configured_for_ingestion"]),
        "discovered": len(manifest["discovered"]),
        "parsed": manifest["parsed"],
        "chunked": manifest["chunked"],
        "embedded": manifest["embedded"],
        "indexed": manifest["indexed"],
        "retrievable": manifest["retrievable"],
        "faiss_ntotal": validation["faiss_ntotal"],
        "metadata_count": validation["metadata_count"],
        "embedding_failures": len(manifest.get("embedding_failures", [])),
        "ingestion_failures": len(manifest.get("ingestion_failures", [])),
        "skipped_documents": len(manifest.get("skipped_documents", [])),
        "build_status": manifest["build_status"],
        "build_elapsed_seconds": manifest.get("build_elapsed_seconds"),
        "embedding_telemetry": manifest.get("embedding_telemetry"),
    })


if __name__ == "__main__":
    build_index()

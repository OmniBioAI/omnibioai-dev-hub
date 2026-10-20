"""Trusted ingestion primitives for Dev Hub.

This module keeps source selection, provenance metadata and chunk identity
separate from embedding/index writes so tests can validate the safety contract
without contacting Ollama or mutating an index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from processing.chunker import chunk_text

SCHEMA_VERSION = "devhub.trusted-ingestion.v1"
SOURCE_POLICY_VERSION = "phase18-source-selection-v2"
VISIBILITIES = {"PUBLIC", "INTERNAL", "REVIEW_REQUIRED"}
GENERAL_RETRIEVAL_VISIBILITIES = {"PUBLIC"}
DOCS_VISIBILITY_INVENTORY = "visibility/VISIBILITY-INVENTORY.json"
# The docs site withholds pages this manifest classifies PLATFORM_OWNER or
# REVIEW_REQUIRED (site/docusaurus.config.js `exclude`), whatever their front
# matter says, so a PUBLIC inventory entry alone does not mean "published".
DOCS_PUBLICATION_MANIFEST = "reconciliation/OMNIBIOAI-DOCS-V1-2-VISIBILITY-MANIFEST.json"
DOCS_PUBLICATION_POLICY = "PUBLICATION-POLICY.yaml"
WITHHELD_PAGE_CLASSIFICATIONS = {"PLATFORM_OWNER": "INTERNAL", "REVIEW_REQUIRED": "REVIEW_REQUIRED"}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    "obsolete",
    "production-readiness",
    "dist",
    "build",
    "coverage",
}
SKIP_PATH_SEGMENTS = {"work", "archive", "backup", "legacy", "old"}

DEFAULT_REPOSITORIES = [
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


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def normalize_relpath(path: Path, repo_path: Path) -> str:
    return path.relative_to(repo_path).as_posix()


def is_markdown_path(path: Path) -> bool:
    return path.suffix.lower() == ".md"


def title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def document_type_for(repo_name: str, rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    if name == "readme.md":
        return "README"
    if repo_name == "omnibioai-docs" and rel_path.startswith("site/docs/"):
        return "PUBLICATION_PAGE"
    if "security" in rel_path.lower():
        return "SECURITY_DOCUMENTATION"
    if "test" in rel_path.lower():
        return "TEST_SOURCE"
    return "DOCUMENTATION"


_HISTORICAL_TOKENS = {"historical", "archive", "archived", "obsolete", "deprecated", "legacy", "old"}
# "target" is deliberately NOT a token: in this corpus it means drug/CRISPR/miRNA
# target ("drug_target_intelligence", "off_target_prediction"), not target-state docs.
_TARGET_TOKENS = {"roadmap", "future"}


def content_state_for(rel_path: str) -> str:
    # Whole-token match on path words, not substring: substring matching
    # labelled "alphafold" (contains "old") HISTORICAL.
    tokens = [t for t in re.split(r"[^a-z0-9]+", rel_path.lower()) if t]
    if _HISTORICAL_TOKENS & set(tokens):
        return "HISTORICAL"
    design_intent = any(a == "design" and b == "intent" for a, b in pairwise(tokens))
    if _TARGET_TOKENS & set(tokens) or design_intent:
        return "TARGET"
    return "CURRENT"


def bundle_for(rel_path: str) -> str | None:
    """First directory under the repo root; None for root-level files.

    Same semantics as the pre-Phase-18 indexer. The API/UI `bundle` scope
    filter matches on this field, so dropping it silently empties every
    bundle-scoped query.
    """
    parts = rel_path.split("/")
    return parts[0] if len(parts) > 1 else None


def authority_for(repo_name: str, rel_path: str) -> str:
    if repo_name == "omnibioai-docs" and rel_path.startswith("site/docs/"):
        return "AUTHORITATIVE_HANDWRITTEN_OR_GENERATED_DOCS"
    if rel_path.lower().endswith("readme.md"):
        return "AUTHORITATIVE_REPOSITORY_DOCS"
    return "UNKNOWN"


def verification_state_for(repo_name: str, rel_path: str) -> str:
    if repo_name == "omnibioai-docs" and rel_path.startswith("site/docs/"):
        return "CONFIGURED"
    return "UNKNOWN"


@dataclass
class SourcePolicy:
    """Deterministic source-selection policy for Phase 18."""

    repository_names: list[str] = field(default_factory=lambda: list(DEFAULT_REPOSITORIES))
    include_internal_index: bool = True
    include_review_required: bool = False
    policy_version: str = SOURCE_POLICY_VERSION

    def selected_repositories(self) -> list[str]:
        return list(self.repository_names)

    def should_walk_dir(self, root: Path, repo_path: Path, dirname: str) -> bool:
        if dirname in SKIP_DIRS:
            return False
        # No legitimate documentation lives under a dot-directory. This also
        # closes tool/agent scratch state (e.g. a `.claude/worktrees/` tree
        # of duplicate repo checkouts) that a named-directory denylist can't
        # anticipate up front.
        if dirname.startswith("."):
            return False
        rel = root.relative_to(repo_path).as_posix() if root != repo_path else ""
        parts = set(rel.split("/")) if rel else set()
        return not bool(parts & SKIP_PATH_SEGMENTS)

    def should_select_file(self, repo_name: str, rel_path: str) -> bool:
        if not is_markdown_path(Path(rel_path)):
            return False
        if any(part in rel_path.split("/") for part in SKIP_PATH_SEGMENTS):
            return False
        if repo_name == "omnibioai-docs":
            return True
        name = Path(rel_path).name.lower()
        return name == "readme.md" or rel_path.startswith("docs/")


def front_matter(text: str) -> dict[str, Any]:
    """The leading `---` YAML block of a markdown document, or {} if absent or unparsable."""
    m = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_route(route: str) -> str:
    """Site route in the form the docs sitemap uses: leading and trailing slash."""
    route = "/" + route.strip().strip("/")
    return route if route == "/" else route + "/"


class VisibilityResolver:
    """PUBLIC for omnibioai-docs means published on the docs site, not merely classified PUBLIC.

    A docs file is PUBLIC only if all of these hold, otherwise the stricter
    class wins (fail closed):
      * the visibility inventory (front-matter derived) says PUBLIC;
      * it sits in the public docs build input (PUBLICATION-POLICY.yaml
        build_inputs.docs); anything else is never built, so it is INTERNAL;
      * the publication manifest does not withhold it (PLATFORM_OWNER ->
        INTERNAL, REVIEW_REQUIRED -> REVIEW_REQUIRED).
    A missing manifest or policy makes every docs file REVIEW_REQUIRED.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._inventory: dict[str, str] = {}
        inv_path = repo_path / DOCS_VISIBILITY_INVENTORY
        if inv_path.exists():
            data = json.loads(inv_path.read_text(encoding="utf-8"))
            for artifact in data.get("artifacts", []):
                path = artifact.get("path")
                visibility = artifact.get("classification")
                if path and visibility in VISIBILITIES:
                    self._inventory[path] = visibility
        self._withheld: dict[str, str] | None = None
        manifest_path = repo_path / DOCS_PUBLICATION_MANIFEST
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._withheld = {e["path"]: WITHHELD_PAGE_CLASSIFICATIONS[e["page_classification"]]
                              for e in data.get("entries", [])
                              if e.get("path") and e.get("page_classification") in WITHHELD_PAGE_CLASSIFICATIONS}
        self._build_input: str | None = None
        policy_path = repo_path / DOCS_PUBLICATION_POLICY
        if policy_path.exists():
            policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            docs_input = (policy.get("build_inputs") or {}).get("docs") if isinstance(policy, dict) else None
            if isinstance(docs_input, str) and docs_input.strip("/"):
                self._build_input = docs_input.strip("/") + "/"

    def in_public_build_input(self, rel_path: str) -> bool:
        return self._build_input is not None and rel_path.startswith(self._build_input)

    def resolve(self, repo_name: str, rel_path: str) -> tuple[str, str]:
        if repo_name != "omnibioai-docs":
            return "INTERNAL", "repository-docs-default-internal"
        visibility = self._inventory.get(rel_path)
        if visibility not in VISIBILITIES:
            return "REVIEW_REQUIRED", "missing-docs-visibility-fail-closed"
        if visibility != "PUBLIC":
            return visibility, "docs-visibility-inventory"
        if self._withheld is None or self._build_input is None:
            return "REVIEW_REQUIRED", "missing-docs-publication-boundary-fail-closed"
        if rel_path in self._withheld:
            return self._withheld[rel_path], "docs-publication-manifest"
        if not self.in_public_build_input(rel_path):
            return "INTERNAL", "docs-not-in-public-build"
        return "PUBLIC", "docs-visibility-inventory"


def document_id(repo: str, rel_path: str, revision: str, content_hash: str) -> str:
    return sha256_text(f"{repo}\0{rel_path}\0{revision}\0{content_hash}")


def chunk_id(document_id_value: str, chunk_index: int, chunk_hash: str) -> str:
    return sha256_text(f"{document_id_value}\0{chunk_index}\0{chunk_hash}")


def iter_candidate_files(repo_path: Path, repo_name: str, policy: SourcePolicy):
    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if policy.should_walk_dir(root_path, repo_path, d)]
        rel_root = root_path.relative_to(repo_path).as_posix() if root_path != repo_path else ""
        if rel_root and any(part in SKIP_PATH_SEGMENTS for part in rel_root.split("/")):
            dirs.clear()
            continue
        for fname in sorted(files):
            path = root_path / fname
            rel = normalize_relpath(path, repo_path)
            if policy.should_select_file(repo_name, rel):
                yield path, rel


def discover_documents(repo_base: str, policy: SourcePolicy | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = policy or SourcePolicy()
    base = Path(repo_base)
    docs: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "configured_repositories": policy.selected_repositories(),
        "repositories_discovered": [],
        "missing_repositories": [],
        "skipped_documents": [],
        "failures": [],
    }

    for repo_name in policy.selected_repositories():
        repo_path = base / repo_name
        if not repo_path.is_dir():
            stats["missing_repositories"].append(repo_name)
            continue
        stats["repositories_discovered"].append(repo_name)
        revision = git_revision(repo_path)
        visibility_resolver = VisibilityResolver(repo_path)
        for path, rel_path in iter_candidate_files(repo_path, repo_name, policy):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError as exc:
                stats["failures"].append({"repo": repo_name, "path": rel_path, "reason": str(exc)})
                continue
            if not text:
                stats["skipped_documents"].append({"repo": repo_name, "path": rel_path, "reason": "empty"})
                continue
            visibility, visibility_source = visibility_resolver.resolve(repo_name, rel_path)
            if visibility not in VISIBILITIES:
                visibility = "REVIEW_REQUIRED"
                visibility_source = "invalid-visibility-fail-closed"
            if visibility == "REVIEW_REQUIRED" and not policy.include_review_required:
                stats["skipped_documents"].append({"repo": repo_name, "path": rel_path, "reason": "review-required"})
                continue
            if visibility == "INTERNAL" and not policy.include_internal_index:
                stats["skipped_documents"].append({"repo": repo_name, "path": rel_path, "reason": "internal-excluded"})
                continue
            route = None  # docs-site route (front matter slug); lets promotion check the page is actually published
            if repo_name == "omnibioai-docs" and visibility_resolver.in_public_build_input(rel_path):
                slug = front_matter(text).get("slug")
                if isinstance(slug, str) and slug.strip():
                    route = normalize_route(slug)
            content_hash = sha256_text(text)
            doc_id = document_id(repo_name, rel_path, revision, content_hash)
            docs.append({
                "document_id": doc_id,
                "repo": repo_name,
                "repository": repo_name,
                "repo_path": str(repo_path),
                "relative_path": rel_path,
                "bundle": bundle_for(rel_path),
                "source_revision": revision,
                "content_hash": content_hash,
                "title": title_from_markdown(text, Path(rel_path).stem),
                "document_type": document_type_for(repo_name, rel_path),
                "visibility": visibility,
                "visibility_source": visibility_source,
                "route": route,
                "verification_state": verification_state_for(repo_name, rel_path),
                "source_authority": authority_for(repo_name, rel_path),
                "content_state": content_state_for(rel_path),
                "generated_state": "GENERATED" if repo_name == "omnibioai-docs" and "/generated/" in rel_path else "SOURCE",
                "text": text,
            })
    return docs, stats


def chunks_for_document(doc: dict[str, Any], build_id: str, embedding_model: str) -> list[dict[str, Any]]:
    raw_chunks = chunk_text(doc["text"])
    records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(raw_chunks):
        chash = sha256_text(chunk)
        cid = chunk_id(doc["document_id"], idx, chash)
        heading = chunk.splitlines()[0].strip() if chunk.lstrip().startswith("#") else None
        meta = {k: v for k, v in doc.items() if k not in {"text", "repo_path"}}
        meta.update({
            "chunk_id": cid,
            "chunk_hash": chash,
            "chunk_index": idx,
            "chunk_count": len(raw_chunks),
            "heading": heading,
            "text": chunk,
            "source": f"{doc['repo']}:{doc['relative_path']}@{doc['source_revision']}",
            "citation": {
                "repository": doc["repo"],
                "relative_path": doc["relative_path"],
                "source_revision": doc["source_revision"],
                "document_id": doc["document_id"],
                "chunk_id": cid,
            },
            "ingestion_build_id": build_id,
            "ingestion_timestamp": utc_now_iso(),
            "embedding_model": embedding_model,
            "metadata_schema_version": SCHEMA_VERSION,
        })
        records.append(meta)
    return records


def build_manifest(build_id: str, metadata: list[dict[str, Any]], discovery_stats: dict[str, Any], *,
                   embedding_provider: str, embedding_model: str, embedding_identity: str,
                   embedding_dimension: int, vector_backend: str = "FAISS IndexFlatIP") -> dict[str, Any]:
    visibility_counts = {v: 0 for v in sorted(VISIBILITIES)}
    for meta in metadata:
        visibility_counts[meta.get("visibility", "REVIEW_REQUIRED")] = visibility_counts.get(meta.get("visibility", "REVIEW_REQUIRED"), 0) + 1
    repos = sorted({m["repo"] for m in metadata})
    return {
        "schema": "devhub.index-manifest.v1",
        "build_id": build_id,
        "build_timestamp": utc_now_iso(),
        "metadata_schema_version": SCHEMA_VERSION,
        "source_selection_policy_version": SOURCE_POLICY_VERSION,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_identity": embedding_identity,
        "embedding_dimension": embedding_dimension,
        "vector_backend": vector_backend,
        "configured_for_ingestion": discovery_stats.get("configured_repositories", []),
        "discovered": discovery_stats.get("repositories_discovered", []),
        "parsed": len({m["document_id"] for m in metadata}),
        "chunked": len(metadata),
        "embedded": 0,
        "indexed": 0,
        "retrievable": 0,
        "evaluated": False,
        "repositories": repos,
        "repository_revisions": {repo: next(m["source_revision"] for m in metadata if m["repo"] == repo) for repo in repos},
        "selected_document_count": len({m["document_id"] for m in metadata}),
        "selected_chunk_count": len(metadata),
        "visibility_counts": visibility_counts,
        "skipped_documents": discovery_stats.get("skipped_documents", []),
        "ingestion_failures": discovery_stats.get("failures", []),
        "artifact_hashes": {},
        "evaluation_status": "NOT_EVALUATED",
        "promotion_status": "CANDIDATE",
        "build_status": "CLEAN",
        "missing_repositories": discovery_stats.get("missing_repositories", []),
    }

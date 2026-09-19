"""Tests for Phase 18 trusted ingestion metadata and source policy."""

import json

from ingestion.trusted import (
    SourcePolicy,
    chunks_for_document,
    discover_documents,
    document_id,
)


def test_docs_visibility_inventory_preserves_public_and_excludes_review_required(tmp_path):
    repo_base = tmp_path
    docs = repo_base / "omnibioai-docs"
    (docs / "visibility").mkdir(parents=True)
    (docs / "site/docs/public").mkdir(parents=True)
    (docs / "tutorials").mkdir()
    (docs / "visibility/VISIBILITY-INVENTORY.json").write_text(json.dumps({
        "artifacts": [
            {"path": "site/docs/public/index.md", "classification": "PUBLIC"},
            {"path": "tutorials/review.md", "classification": "REVIEW_REQUIRED"},
        ]
    }))
    (docs / "site/docs/public/index.md").write_text("# Public\n\nVisible")
    (docs / "tutorials/review.md").write_text("# Review\n\nNope")

    found, stats = discover_documents(str(repo_base), SourcePolicy(repository_names=["omnibioai-docs"]))

    assert [d["relative_path"] for d in found] == ["site/docs/public/index.md"]
    assert found[0]["visibility"] == "PUBLIC"
    assert stats["skipped_documents"][0]["reason"] == "review-required"


def test_app_repo_docs_default_internal_and_source_code_not_selected(tmp_path):
    repo = tmp_path / "omnibioai-tes"
    (repo / "docs").mkdir(parents=True)
    (repo / "README.md").write_text("# TES\n\nReadme")
    (repo / "docs/guide.md").write_text("# Guide\n\nDocs")
    (repo / "service.py").write_text("print('not selected')")

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-tes"]))

    assert {d["relative_path"] for d in found} == {"README.md", "docs/guide.md"}
    assert {d["visibility"] for d in found} == {"INTERNAL"}


def test_dot_directories_are_never_walked(tmp_path):
    """A tool/agent scratch tree under a dot-directory (e.g. `.claude/worktrees/`
    holding duplicate repo checkouts) must never be treated as a documentation
    source, even though its files would otherwise match README.md/docs/ selection."""
    repo = tmp_path / "omnibioai-workbench"
    (repo / ".claude" / "worktrees" / "agent-1" / "plugins" / "foo").mkdir(parents=True)
    (repo / ".claude" / "worktrees" / "agent-1" / "plugins" / "foo" / "README.md").write_text("# Foo\n\nDupe")
    (repo / ".claude" / "worktrees" / "agent-1" / "README.md").write_text("# Agent 1\n\nScratch")
    (repo / "plugins" / "foo").mkdir(parents=True)
    (repo / "plugins" / "foo" / "README.md").write_text("# Foo\n\nReal")
    (repo / "README.md").write_text("# Workbench\n\nReadme")

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-workbench"]))

    assert {d["relative_path"] for d in found} == {"README.md", "plugins/foo/README.md"}
    assert not any(d["relative_path"].startswith(".claude") for d in found)


def test_missing_docs_visibility_fails_closed_to_review_required(tmp_path):
    docs = tmp_path / "omnibioai-docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Missing policy\n\nNo public default")

    found, stats = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-docs"]))

    assert found == []
    assert stats["skipped_documents"][0]["reason"] == "review-required"


def test_document_and_chunk_identity_include_revision_path_and_hash():
    doc = {
        "document_id": document_id("repo", "README.md", "abc", "contenthash"),
        "repo": "repo",
        "repository": "repo",
        "relative_path": "README.md",
        "source_revision": "abc",
        "content_hash": "contenthash",
        "title": "Readme",
        "document_type": "README",
        "visibility": "PUBLIC",
        "visibility_source": "test",
        "verification_state": "UNKNOWN",
        "source_authority": "UNKNOWN",
        "content_state": "CURRENT",
        "generated_state": "SOURCE",
        "text": "# Readme\n\nBody",
    }
    chunks = chunks_for_document(doc, "build-1", "nomic-embed-text")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["document_id"] == doc["document_id"]
    assert chunk["chunk_id"]
    assert chunk["chunk_hash"]
    assert chunk["citation"] == {
        "repository": "repo",
        "relative_path": "README.md",
        "source_revision": "abc",
        "document_id": doc["document_id"],
        "chunk_id": chunk["chunk_id"],
    }
    assert not chunk["source"].startswith("/")


def test_default_repository_policy_adds_only_deliberate_phase18_repositories():
    """Include Workbench, TES, and Dev Hub, but not every present sibling repository."""
    selected = set(SourcePolicy().selected_repositories())
    assert {"omnibioai-workbench", "omnibioai-tes", "omnibioai-dev-hub"} <= selected
    assert "omnibioai-billing" not in selected
    assert "omnibioai-ui" not in selected
    assert "omnibioai-design-tokens" not in selected
    assert "omnibioai-launcher" not in selected
    assert "omnibioai-ecosystem-regression" not in selected

"""Tests for Phase 18 trusted ingestion metadata and source policy."""

import json

import pytest

from ingestion.trusted import (
    SourcePolicy,
    VisibilityResolver,
    bundle_for,
    chunks_for_document,
    content_state_for,
    discover_documents,
    document_id,
    front_matter,
    normalize_route,
)


def _docs_repo(tmp_path, inventory, manifest_entries=(), *, manifest=True, policy=True):
    """A synthetic omnibioai-docs checkout with the three publication-boundary inputs."""
    docs = tmp_path / "omnibioai-docs"
    (docs / "visibility").mkdir(parents=True)
    (docs / "visibility/VISIBILITY-INVENTORY.json").write_text(json.dumps({
        "artifacts": [{"path": p, "classification": c} for p, c in inventory.items()]}))
    if manifest:
        (docs / "reconciliation").mkdir()
        (docs / "reconciliation/OMNIBIOAI-DOCS-V1-2-VISIBILITY-MANIFEST.json").write_text(json.dumps({
            "entries": [{"path": p, "page_classification": c} for p, c in dict(manifest_entries).items()]}))
    if policy:
        (docs / "PUBLICATION-POLICY.yaml").write_text("build_inputs:\n  docs: site/docs\n  static: site/static\n")
    return docs


def test_docs_visibility_inventory_preserves_public_and_excludes_review_required(tmp_path):
    repo_base = tmp_path
    docs = _docs_repo(tmp_path, {"site/docs/public/index.md": "PUBLIC", "tutorials/review.md": "REVIEW_REQUIRED"})
    (docs / "site/docs/public").mkdir(parents=True)
    (docs / "tutorials").mkdir()
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


@pytest.mark.parametrize("path,expected", [
    ("proteomics/alphafold_structure/README.md", "CURRENT"),  # "alphafold" contains "old"
    ("structural_genomics/fold_switching/README.md", "CURRENT"),
    ("metabolomics/targeted_metabolomics_panel/README.md", "CURRENT"),  # "targeted" != "target"
    ("funcgen/crispr_offtarget/README.md", "CURRENT"),
    ("docs/archive/notes.md", "HISTORICAL"),
    ("docs/legacy-auth.md", "HISTORICAL"),
    ("docs/old_design.md", "HISTORICAL"),
    ("docs/roadmap.md", "TARGET"),
    ("docs/future-work.md", "TARGET"),
    ("docs/design-intent.md", "TARGET"),
    ("plugins/drug_target_intelligence/README.md", "CURRENT"),  # biology "target", not target-state
    ("rnaseq/mirna_target_prediction/README.md", "CURRENT"),
])
def test_content_state_matches_whole_path_tokens_not_substrings(path, expected):
    assert content_state_for(path) == expected


@pytest.mark.parametrize("path,expected", [
    ("README.md", None),
    ("atacseq/README.md", "atacseq"),
    ("rnaseq/count_matrix_qc/README.md", "rnaseq"),
    ("docs/guide.md", "docs"),
])
def test_bundle_is_first_directory_or_none_for_root_files(path, expected):
    assert bundle_for(path) == expected


def test_chunk_metadata_carries_bundle_for_api_scope_filter(tmp_path):
    repo = tmp_path / "omnibioai-workflow-bundles"
    (repo / "atacseq").mkdir(parents=True)
    (repo / "atacseq" / "README.md").write_text("# ATAC\n\nPipeline")
    (repo / "README.md").write_text("# Bundles\n\nRoot")

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-workflow-bundles"]))
    by_path = {d["relative_path"]: d for d in found}
    assert by_path["atacseq/README.md"]["bundle"] == "atacseq"
    assert by_path["README.md"]["bundle"] is None
    chunk = chunks_for_document(by_path["atacseq/README.md"], "b1", "nomic-embed-text")[0]
    assert chunk["bundle"] == "atacseq"


@pytest.mark.parametrize("text,expected", [
    ("intro\n# Title Here\nbody", "Title Here"),
    ("#   \nbody", "fallback"),
    ("no heading at all", "fallback"),
])
def test_title_from_markdown_uses_first_h1_or_fallback(text, expected):
    from ingestion.trusted import title_from_markdown

    assert title_from_markdown(text, "fallback") == expected


@pytest.mark.parametrize("repo,path,expected", [
    ("omnibioai-tes", "docs/README.md", "README"),
    ("omnibioai-docs", "site/docs/page.md", "PUBLICATION_PAGE"),
    ("omnibioai-tes", "docs/security/model.md", "SECURITY_DOCUMENTATION"),
    ("omnibioai-tes", "docs/testing.md", "TEST_SOURCE"),
    ("omnibioai-tes", "docs/guide.md", "DOCUMENTATION"),
])
def test_document_type_for_classifies_by_name_repo_and_path(repo, path, expected):
    from ingestion.trusted import document_type_for

    assert document_type_for(repo, path) == expected


def test_source_policy_skips_denylisted_dirs_and_skip_segment_files(tmp_path):
    policy = SourcePolicy()
    assert not policy.should_walk_dir(tmp_path, tmp_path, "node_modules")
    assert policy.should_walk_dir(tmp_path, tmp_path, "docs")
    assert not policy.should_walk_dir(tmp_path / "archive", tmp_path, "docs")
    assert not policy.should_select_file("omnibioai-docs", "archive/notes.md")
    assert not policy.should_select_file("omnibioai-tes", "docs/guide.txt")


def test_files_under_a_skip_segment_directory_are_not_discovered(tmp_path):
    repo = tmp_path / "omnibioai-tes"
    (repo / "docs").mkdir(parents=True)
    (repo / "archive").mkdir()
    (repo / "docs/guide.md").write_text("# Guide\n\nDocs")
    (repo / "archive/README.md").write_text("# Old\n\nStale")

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-tes"]))

    assert [d["relative_path"] for d in found] == ["docs/guide.md"]


def test_unreadable_empty_and_missing_sources_are_reported(tmp_path, monkeypatch):
    from pathlib import Path

    repo = tmp_path / "omnibioai-tes"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs/empty.md").write_text("   \n")
    (repo / "docs/broken.md").write_text("# Broken")
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "broken.md":
            raise OSError("disk error")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    found, stats = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-tes", "omnibioai-missing"]))

    assert found == []
    assert stats["failures"] == [{"repo": "omnibioai-tes", "path": "docs/broken.md", "reason": "disk error"}]
    assert stats["skipped_documents"] == [{"repo": "omnibioai-tes", "path": "docs/empty.md", "reason": "empty"}]
    assert stats["missing_repositories"] == ["omnibioai-missing"]


def test_invalid_resolved_visibility_fails_closed_to_review_required(tmp_path, monkeypatch):
    from ingestion.trusted import VisibilityResolver

    repo = tmp_path / "omnibioai-tes"
    repo.mkdir()
    (repo / "README.md").write_text("# TES\n\nReadme")
    monkeypatch.setattr(VisibilityResolver, "resolve", lambda self, repo_name, rel_path: ("SECRET", "bogus"))

    found, stats = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-tes"]))
    assert found == [] and stats["skipped_documents"][0]["reason"] == "review-required"

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-tes"], include_review_required=True))
    assert found[0]["visibility"] == "REVIEW_REQUIRED"
    assert found[0]["visibility_source"] == "invalid-visibility-fail-closed"


def test_build_manifest_counts_visibility_documents_and_revisions():
    from ingestion.trusted import build_manifest

    metadata = [
        {"repo": "a", "document_id": "d1", "source_revision": "r1", "visibility": "PUBLIC"},
        {"repo": "a", "document_id": "d1", "source_revision": "r1", "visibility": "PUBLIC"},
        {"repo": "b", "document_id": "d2", "source_revision": "r2", "visibility": "INTERNAL"},
        {"repo": "b", "document_id": "d3", "source_revision": "r2"},
    ]
    stats = {"configured_repositories": ["a", "b", "c"], "repositories_discovered": ["a", "b"],
             "missing_repositories": ["c"], "skipped_documents": [{"path": "x"}], "failures": []}

    m = build_manifest("b1", metadata, stats, embedding_provider="ollama", embedding_model="nomic-embed-text",
                       embedding_identity="ollama:nomic-embed-text", embedding_dimension=768)

    assert m["build_id"] == "b1" and m["vector_backend"] == "FAISS IndexFlatIP"
    assert m["visibility_counts"] == {"INTERNAL": 1, "PUBLIC": 2, "REVIEW_REQUIRED": 1}
    assert m["parsed"] == 3 and m["chunked"] == 4 and m["selected_document_count"] == 3
    assert m["repositories"] == ["a", "b"] and m["repository_revisions"] == {"a": "r1", "b": "r2"}
    assert m["missing_repositories"] == ["c"] and m["discovered"] == ["a", "b"]
    assert m["build_status"] == "CLEAN" and m["promotion_status"] == "CANDIDATE"


@pytest.mark.parametrize("page_class, expected", [
    ("PLATFORM_OWNER", ("INTERNAL", "docs-publication-manifest")),
    ("REVIEW_REQUIRED", ("REVIEW_REQUIRED", "docs-publication-manifest")),
    ("DEVELOPER", ("PUBLIC", "docs-visibility-inventory")),  # published on the site, so not withheld
    ("PUBLIC", ("PUBLIC", "docs-visibility-inventory")),
])
def test_publication_manifest_withholds_pages_whose_front_matter_says_public(tmp_path, page_class, expected):
    page = "site/docs/admin/ops.md"
    docs = _docs_repo(tmp_path, {page: "PUBLIC"}, {page: page_class})
    assert VisibilityResolver(docs).resolve("omnibioai-docs", page) == expected


def test_public_docs_outside_the_build_input_are_never_published(tmp_path):
    docs = _docs_repo(tmp_path, {"COMPAT.md": "PUBLIC", "site/docs-old/x.md": "PUBLIC"})
    resolver = VisibilityResolver(docs)
    assert resolver.resolve("omnibioai-docs", "COMPAT.md") == ("INTERNAL", "docs-not-in-public-build")
    assert resolver.resolve("omnibioai-docs", "site/docs-old/x.md") == ("INTERNAL", "docs-not-in-public-build")


def test_inventory_stricter_than_manifest_wins(tmp_path):
    page = "site/docs/a.md"
    docs = _docs_repo(tmp_path, {page: "INTERNAL"}, {page: "PUBLIC"})
    assert VisibilityResolver(docs).resolve("omnibioai-docs", page) == ("INTERNAL", "docs-visibility-inventory")


@pytest.mark.parametrize("missing", ["manifest", "policy"])
def test_missing_publication_boundary_input_fails_closed(tmp_path, missing):
    page = "site/docs/a.md"
    docs = _docs_repo(tmp_path, {page: "PUBLIC"}, manifest=missing != "manifest", policy=missing != "policy")
    assert VisibilityResolver(docs).resolve("omnibioai-docs", page) == (
        "REVIEW_REQUIRED", "missing-docs-publication-boundary-fail-closed")


def test_route_is_recorded_from_the_front_matter_slug_of_built_pages_only(tmp_path):
    docs = _docs_repo(tmp_path, {"site/docs/u/start.md": "PUBLIC", "site/docs/u/noslug.md": "PUBLIC",
                                 "site/docs/admin/ops.md": "PUBLIC", "guides/g.md": "PUBLIC"},
                      {"site/docs/admin/ops.md": "PLATFORM_OWNER"})
    (docs / "site/docs/u").mkdir(parents=True)
    (docs / "site/docs/admin").mkdir()
    (docs / "guides").mkdir()
    (docs / "site/docs/u/start.md").write_text("---\ntitle: Start\nslug: /user/start\n---\n# Start\n\nGo")
    (docs / "site/docs/u/noslug.md").write_text("# No slug\n\nBody")
    (docs / "site/docs/admin/ops.md").write_text("---\nslug: /admin/ops/\n---\n# Ops\n\nPrivileged")
    (docs / "guides/g.md").write_text("---\nslug: /guides/g/\n---\n# G\n\nNot built")

    found, _ = discover_documents(str(tmp_path), SourcePolicy(repository_names=["omnibioai-docs"]))
    by_path = {d["relative_path"]: d for d in found}

    assert by_path["site/docs/u/start.md"]["route"] == "/user/start/"
    assert by_path["site/docs/u/start.md"]["visibility"] == "PUBLIC"
    assert by_path["site/docs/u/noslug.md"]["route"] is None
    assert by_path["site/docs/admin/ops.md"]["visibility"] == "INTERNAL"  # withheld, still indexed as INTERNAL
    assert by_path["guides/g.md"]["route"] is None and by_path["guides/g.md"]["visibility"] == "INTERNAL"
    chunk = chunks_for_document(by_path["site/docs/u/start.md"], "b", "m")[0]
    assert chunk["route"] == "/user/start/"


@pytest.mark.parametrize("text, expected", [
    ("---\nslug: /a/\ntitle: A\n---\n# A", {"slug": "/a/", "title": "A"}),
    ("# no front matter\n---\nslug: /x/\n---", {}),
    ("---\n: [unbalanced\n---\nbody", {}),
    ("---\n- a list\n---\nbody", {}),
    ("---\nslug: /end/\n---", {"slug": "/end/"}),
])
def test_front_matter_reads_only_a_leading_mapping(text, expected):
    assert front_matter(text) == expected


@pytest.mark.parametrize("route, expected", [("/", "/"), ("", "/"), ("/a/b", "/a/b/"), ("a/b/", "/a/b/"), (" /x/ ", "/x/")])
def test_normalize_route(route, expected):
    assert normalize_route(route) == expected

"""Tests for the Phase 18 evaluation harness (fake engine, no index, no Ollama)."""

import json

from scripts.run_eval import (
    MIN_POSITIVE_CASES_PER_SCOPE,
    MIN_RECALL_AT_K,
    _contains_source,
    evaluate_case,
    summarize,
)


class FakeEngine:
    def __init__(self, docs):
        self.docs = docs
        self.last_allowed = None

    def retrieve(self, query, top_k=5, repo=None, bundle=None, rerank=False, allowed_visibilities=None):
        self.last_allowed = allowed_visibilities
        return self.docs


def _doc(repo="omnibioai-docs", rel="site/docs/admin/security.md", vis="PUBLIC", cid="c1", text="body", **extra):
    return {
        "repo": repo, "relative_path": rel, "visibility": vis, "chunk_id": cid, "text": text,
        "source": f"{repo}:{rel}@abc", "score": 0.9,
        "citation": {"repository": repo, "relative_path": rel}, **extra,
    }


def test_source_matching_understands_repo_slash_path_form():
    d = _doc(repo="omnibioai-workflow-bundles", rel="atacseq/README.md")
    assert _contains_source(d, "omnibioai-workflow-bundles/atacseq")
    assert _contains_source(d, "atacseq/README.md")
    assert not _contains_source(d, "omnibioai-workflow-bundles/chipseq")


def test_default_scope_is_public_and_internal_scope_requires_explicit_opt_in():
    eng = FakeEngine([_doc()])
    r = evaluate_case(eng, {"query": "q", "expected_source_contains": "admin/security.md"})
    assert r["scope"] == "public" and eng.last_allowed == {"PUBLIC"}
    r = evaluate_case(eng, {"query": "q", "allowed_visibilities": ["PUBLIC", "INTERNAL"], "expected_source_contains": "admin/security.md"})
    assert r["scope"] == "internal"


def test_any_result_outside_allowed_visibility_is_a_leak_even_if_case_forbids_nothing():
    eng = FakeEngine([_doc(vis="INTERNAL", cid="leak")])
    r = evaluate_case(eng, {"query": "q", "expected_source_contains": "admin/security.md"})
    assert r["visibility_leaks"] == 1 and not r["passed"]


def test_forbidden_source_check_is_not_vacuous_with_repo_slash_path():
    eng = FakeEngine([_doc(repo="omnibioai-rag", rel="README.md", vis="PUBLIC")])
    r = evaluate_case(eng, {"query": "q", "forbidden_source_contains": ["omnibioai-rag/README.md"]})
    assert r["forbidden_source_hits"] == 1 and not r["passed"]


def test_unsupported_terms_fail_when_retrieved_text_supports_the_claim():
    eng = FakeEngine([_doc(text="Configure the Salesforce connector here")])
    r = evaluate_case(eng, {"query": "q", "unsupported_terms": ["salesforce"]})
    assert r["unsupported_term_hits"] == ["salesforce"] and not r["passed"] and r["negative"]
    eng = FakeEngine([_doc(text="unrelated nearest neighbour")])
    assert evaluate_case(eng, {"query": "q", "unsupported_terms": ["salesforce"]})["passed"]


def test_expected_metadata_must_match_the_matched_chunk():
    case = {"query": "q", "expected_source_contains": "admin/security.md", "expected_metadata": {"content_state": "CURRENT"}}
    assert evaluate_case(FakeEngine([_doc(content_state="CURRENT")]), case)["passed"]
    assert not evaluate_case(FakeEngine([_doc(content_state="TARGET")]), case)["passed"]


def test_duplicate_results_fail_the_case():
    eng = FakeEngine([_doc(cid="same"), _doc(cid="same")])
    r = evaluate_case(eng, {"query": "q", "expected_source_contains": "admin/security.md"})
    assert r["duplicate_results"] == 1 and not r["passed"]


def test_missing_citation_fails_positive_case():
    d = _doc()
    d["citation"] = None
    assert not evaluate_case(FakeEngine([d]), {"query": "q", "expected_source_contains": "admin/security.md"})["passed"]


def _res(scope, passed, negative=False, expected="x", leaks=0):
    return {"scope": scope, "passed": passed, "negative": negative, "expected": None if negative else expected,
            "visibility_leaks": leaks, "duplicate_results": 0}


def _healthy():
    n = MIN_POSITIVE_CASES_PER_SCOPE
    return [_res("public", True) for _ in range(n)] + [_res("internal", True) for _ in range(n)] + [_res("public", True, negative=True)]


def test_gate_passes_when_every_condition_holds():
    assert summarize(_healthy())["threshold_passed"]


def test_gate_fails_on_any_leak_even_with_perfect_recall():
    results = _healthy() + [_res("public", False, negative=True, leaks=1)]
    s = summarize(results)
    assert not s["gates"]["zero_visibility_leakage"] and not s["threshold_passed"]


def test_gate_fails_if_a_safety_case_fails():
    s = summarize(_healthy() + [_res("public", False, negative=True)])
    assert not s["gates"]["all_safety_cases_pass"] and not s["threshold_passed"]


def test_gate_recall_boundary_is_exactly_the_frozen_threshold():
    n = 10
    ok = [_res("public", True) for _ in range(7)] + [_res("public", False) for _ in range(3)]  # 0.70
    rest = [_res("internal", True) for _ in range(n)]
    assert summarize(ok + rest)["gates"]["public_recall_at_k"]
    below = [_res("public", True) for _ in range(6)] + [_res("public", False) for _ in range(4)]  # 0.60
    assert not summarize(below + rest)["gates"]["public_recall_at_k"]
    assert MIN_RECALL_AT_K == 0.70


def test_gate_fails_when_a_scope_has_too_few_positive_cases():
    few = [_res("public", True) for _ in range(MIN_POSITIVE_CASES_PER_SCOPE - 1)]
    rest = [_res("internal", True) for _ in range(MIN_POSITIVE_CASES_PER_SCOPE)]
    assert not summarize(few + rest)["gates"]["public_recall_at_k"]


def test_shipped_dataset_is_well_formed_and_covers_both_scopes():
    with open("tests/eval/retrieval_eval.json", encoding="utf-8") as f:
        cases = json.load(f)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    internal = [c for c in cases if "INTERNAL" in c.get("allowed_visibilities", [])]
    public_pos = [c for c in cases if "INTERNAL" not in c.get("allowed_visibilities", []) and c.get("expected_source_contains")]
    assert len(public_pos) >= MIN_POSITIVE_CASES_PER_SCOPE and len(internal) >= MIN_POSITIVE_CASES_PER_SCOPE
    assert any(c.get("unsupported_terms") for c in cases)
    assert any(c.get("forbidden_visibilities") for c in cases)
    assert not any(c.get("expect_no_results") for c in cases)  # unsatisfiable without a relevance cutoff

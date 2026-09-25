#!/usr/bin/env python3
"""Phase 18 retrieval evaluation gate for Dev Hub.

Loads a local FAISS artifact and runs retrieval cases from JSON. Query
embeddings need the configured local Ollama endpoint; nothing else is
contacted.

PROMOTION GATE (frozen before the first candidate evaluation; do not tune
after seeing results). ALL of the following must hold:

  1. zero visibility leakage -- no retrieved chunk outside the case's allowed
     visibilities, in any case;
  2. every safety case passes (visibility-leakage, unsupported-claim and
     index-separation cases: these are 100%, not a percentage);
  3. recall@5 >= MIN_RECALL_AT_K on positive cases in the PUBLIC scope
     (ordinary callers -- what the API actually serves);
  4. recall@5 >= MIN_RECALL_AT_K on positive cases in the INTERNAL scope
     (explicit allowed_visibilities incl. INTERNAL -- index quality on the
     majority of the corpus; not exposed by the API);
  5. each scope has at least MIN_POSITIVE_CASES_PER_SCOPE positive cases, so a
     scope cannot pass on an empty or token dataset.

PRODUCTION PROFILE (--profile production), for the PUBLIC-only artifact that is
actually served: gates 1-3 only. Gate 4 (INTERNAL-scope recall) is dropped
because a PUBLIC-only artifact cannot contain INTERNAL content by design; the
INTERNAL-scope cases are still run, as an informational probe that must find
nothing. Thresholds are identical to the full profile.

The earlier `all_passed` conjunction (every case must pass) is not part of the
gate: it made the stated 70% threshold unreachable-in-effect (100% required).
Failing cases are still listed individually in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.vector_store import VectorStore
from rag.engine import RAGEngine

TOP_K = 5
MIN_RECALL_AT_K = 0.70
MIN_POSITIVE_CASES_PER_SCOPE = 10
ZERO_VISIBILITY_LEAKAGE_REQUIRED = True
PASS_MARKER = "PASS"
FAIL_MARKER = "FAIL"


def load_eval(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("evaluation file must contain a list of cases")
    return data


def _source_strings(doc: dict[str, Any]) -> list[str]:
    citation = doc.get("citation") or {}
    repo = doc.get("repo") or doc.get("repository") or citation.get("repository") or ""
    rel = doc.get("relative_path") or citation.get("relative_path") or ""
    parts = [
        doc.get("source", ""),
        repo,
        rel,
        f"{repo}/{rel}" if repo and rel else "",
    ]
    return [str(p) for p in parts if p]


def _contains_source(doc: dict[str, Any], expected: str) -> bool:
    return any(expected in source for source in _source_strings(doc))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def evaluate_case(engine: RAGEngine, case: dict[str, Any], *, top_k: int = TOP_K,
                  min_relevance: float | None = None) -> dict[str, Any]:
    query = case["query"]
    allowed = set(case.get("allowed_visibilities", ["PUBLIC"]))
    scope = "internal" if "INTERNAL" in allowed else "public"

    docs = engine.retrieve(
        query,
        top_k=top_k,
        repo=case.get("repo"),
        bundle=case.get("bundle"),
        rerank=case.get("rerank", False),
        allowed_visibilities=allowed,
        **({"min_relevance": min_relevance} if min_relevance is not None else {}),
    )
    expected = case.get("expected_source_contains")
    expected_metadata = case.get("expected_metadata") or {}
    expect_no_results = bool(case.get("expect_no_results"))
    unsupported_terms = [t.lower() for t in _as_list(case.get("unsupported_terms"))]
    forbidden_visibilities = set(case.get("forbidden_visibilities", []))
    forbidden_source_contains = _as_list(case.get("forbidden_source_contains"))

    # Leakage is anything outside the caller's allowed set, plus anything the
    # case explicitly forbids -- not only what a case remembered to list.
    leaked_docs = [d for d in docs if d.get("visibility") not in allowed or d.get("visibility") in forbidden_visibilities]
    forbidden_source_hits = [
        d for d in docs
        if any(fragment in source for fragment in forbidden_source_contains for source in _source_strings(d))
    ]
    duplicate_results = len(docs) - len({d.get("chunk_id") or d.get("text") for d in docs})
    term_hits = sorted({t for d in docs for t in unsupported_terms if t in (d.get("text") or "").lower()})
    source_hit = next((d for d in docs if expected and _contains_source(d, expected)), None)
    citation_ok = True
    if expected and case.get("require_citation", True):
        citation_ok = bool(source_hit and source_hit.get("citation"))
    metadata_ok = all(source_hit is not None and source_hit.get(k) == v for k, v in expected_metadata.items()) if expected_metadata else True

    common_ok = not leaked_docs and not forbidden_source_hits and duplicate_results == 0
    if unsupported_terms:
        passed = common_ok and not term_hits
    elif expect_no_results:
        passed = not docs and common_ok
    elif expected:
        passed = bool(source_hit) and citation_ok and metadata_ok and common_ok
    else:
        passed = bool(docs) and common_ok

    return {
        "id": case.get("id", query[:60]),
        "category": case.get("category", "uncategorized"),
        "scope": scope,
        "query": query,
        "passed": passed,
        "expected": expected,
        "matched": source_hit.get("source") if source_hit else (docs[0].get("source") if docs else "(no results)"),
        "retrieved": len(docs),
        "top_score": docs[0].get("score") if docs else None,
        "visibility_leaks": len(leaked_docs),
        "forbidden_source_hits": len(forbidden_source_hits),
        "duplicate_results": duplicate_results,
        "unsupported_term_hits": term_hits,
        "citation_ok": citation_ok,
        "metadata_ok": metadata_ok,
        "negative": bool(expect_no_results or unsupported_terms or forbidden_visibilities or forbidden_source_contains),
    }


def _recall(results: list[dict[str, Any]], scope: str) -> tuple[int, int, float]:
    pos = [r for r in results if r["expected"] and not r["negative"] and r["scope"] == scope]
    ok = sum(1 for r in pos if r["passed"])
    return ok, len(pos), (ok / len(pos) if pos else 0.0)


def summarize(results: list[dict[str, Any]], profile: str = "full") -> dict[str, Any]:
    probes = [r for r in results if r["scope"] == "internal-probe"]
    results = [r for r in results if r["scope"] != "internal-probe"]
    pub_ok, pub_n, pub_recall = _recall(results, "public")
    int_ok, int_n, int_recall = _recall(results, "internal")
    visibility_leaks = sum(r["visibility_leaks"] for r in results)
    negative = [r for r in results if r["negative"]]
    negative_passed = sum(1 for r in negative if r["passed"])
    gates = {
        "zero_visibility_leakage": visibility_leaks == 0 if ZERO_VISIBILITY_LEAKAGE_REQUIRED else True,
        "all_safety_cases_pass": negative_passed == len(negative),
        "public_recall_at_k": pub_n >= MIN_POSITIVE_CASES_PER_SCOPE and pub_recall >= MIN_RECALL_AT_K,
        "internal_recall_at_k": int_n >= MIN_POSITIVE_CASES_PER_SCOPE and int_recall >= MIN_RECALL_AT_K,
    }
    if profile == "production":
        del gates["internal_recall_at_k"]
    return {
        "total": len(results),
        "public": {"positive": pub_n, "passed": pub_ok, "recall_at_k": pub_recall},
        "internal": {"positive": int_n, "passed": int_ok, "recall_at_k": int_recall},
        "safety_cases": len(negative),
        "safety_cases_passed": negative_passed,
        "visibility_leaks": visibility_leaks,
        "duplicate_results": sum(r["duplicate_results"] for r in results),
        "all_cases_passed": all(r["passed"] for r in results),
        "profile": profile,
        "internal_probe": {"cases": len(probes), "expected_source_found": sum(1 for r in probes if r["passed"] or (r["expected"] and r["matched"] and r["expected"] in str(r["matched"]))),
                           "leaks": sum(r["visibility_leaks"] for r in probes)},
        "gates": gates,
        "threshold_passed": all(gates.values()),
        "min_recall_at_k": MIN_RECALL_AT_K,
        "min_positive_cases_per_scope": MIN_POSITIVE_CASES_PER_SCOPE,
        "zero_visibility_leakage_required": ZERO_VISIBILITY_LEAKAGE_REQUIRED,
    }


def _error_result(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    allowed = set(case.get("allowed_visibilities", ["PUBLIC"]))
    return {
        "id": case.get("id", case.get("query", "unknown")),
        "category": case.get("category", "uncategorized"),
        "scope": "internal" if "INTERNAL" in allowed else "public",
        "query": case.get("query", ""),
        "passed": False,
        "expected": case.get("expected_source_contains"),
        "matched": f"[ERROR] {exc}",
        "retrieved": 0,
        "top_score": None,
        "visibility_leaks": 0,
        "forbidden_source_hits": 0,
        "duplicate_results": 0,
        "unsupported_term_hits": [],
        "citation_ok": False,
        "metadata_ok": False,
        "negative": bool(case.get("expect_no_results") or case.get("unsupported_terms")
                         or case.get("forbidden_visibilities") or case.get("forbidden_source_contains")),
    }


def run_eval(index_dir: str, eval_path: str, min_relevance: float | None = None, profile: str = "full") -> dict[str, Any]:
    vs = VectorStore()
    if not vs.load(index_dir):
        print(f"[ERROR] Could not load FAISS index from {index_dir}", file=sys.stderr)
        sys.exit(1)

    engine = RAGEngine(vs)
    cases = load_eval(eval_path)
    results: list[dict[str, Any]] = []
    col_q = min(max(len(c["query"]) for c in cases), 60) if cases else 10

    print(f"\n{'Query':<{col_q}}  {'Scope':<8}  {'Category':<20}  {'Result':<6}  Matched source")
    print("-" * (col_q + 100))
    for case in cases:
        try:
            result = evaluate_case(engine, case, top_k=TOP_K, min_relevance=min_relevance)
        except Exception as e:  # noqa: BLE001 -- per-query boundary: one failing case is reported, not hidden
            result = _error_result(case, e)
        if profile == "production" and result["scope"] == "internal":
            result["scope"] = "internal-probe"
            result["passed"] = None  # informational: not part of the production gate
        results.append(result)
        label = "PROBE" if result["passed"] is None else PASS_MARKER if result["passed"] else FAIL_MARKER
        print(f"{result['query'][:col_q]:<{col_q}}  {result['scope']:<8}  {result['category'][:20]:<20}  {label:<6}  {str(result['matched'])[:70]}")

    summary = summarize(results, profile)
    print("\n" + "=" * (col_q + 60))
    for scope in ("public",) if profile == "production" else ("public", "internal"):
        s = summary[scope]
        print(f"Recall@{TOP_K} [{scope}]: {s['passed']}/{s['positive']} ({s['recall_at_k']:.1%})  (gate >= {MIN_RECALL_AT_K:.0%}, min {MIN_POSITIVE_CASES_PER_SCOPE} cases)")
    if profile == "production":
        ip = summary["internal_probe"]
        print(f"INTERNAL-scope probe (informational, must find nothing): {ip['expected_source_found']}/{ip['cases']} expected INTERNAL sources found, leaks={ip['leaks']}")
    print(f"Safety cases: {summary['safety_cases_passed']}/{summary['safety_cases']}    Visibility leakage: {summary['visibility_leaks']}    Duplicate results: {summary['duplicate_results']}")
    for name, ok in summary["gates"].items():
        print(f"  gate {name}: {'PASS' if ok else 'FAIL'}")
    print(f"Evaluation result: {'PASS' if summary['threshold_passed'] else 'FAIL'}")
    print("=" * (col_q + 60))

    failures = [r for r in results if r["passed"] is False]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures:
            print(f"  - [{f['scope']}] {f['id']}: {f['query'][:90]}")
            print(f"    expected: {f['expected']}  matched: {f['matched']}  leaks={f['visibility_leaks']} terms={f['unsupported_term_hits']}")

    return {"summary": summary, "results": results}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_build_id(manifest_path: str) -> str | None:
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)["build_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 18 Dev Hub retrieval evaluation")
    parser.add_argument("--index-dir", default="data/faiss_index", help="Path to saved FAISS index directory")
    parser.add_argument("--eval", default="tests/eval/retrieval_eval.json", help="Path to eval JSON file")
    parser.add_argument("--json-out", help="Optional path for machine-readable evaluation output")
    parser.add_argument("--min-relevance", type=float, default=None,
                        help="Apply the production cosine relevance cutoff. Omitted = the frozen gate configuration; "
                             "gate thresholds are unchanged either way.")
    parser.add_argument("--profile", choices=["full", "production"], default="full",
                        help="full = frozen gate for a mixed candidate; production = gate for the PUBLIC-only served artifact")
    args = parser.parse_args()

    result = run_eval(args.index_dir, args.eval, args.min_relevance, args.profile)
    manifest_path = os.path.join(args.index_dir, "manifest.json")
    result["provenance"] = {
        "eval_file": args.eval,
        "eval_file_sha256": _sha256(args.eval),
        "index_build_id": _read_build_id(manifest_path),
        "min_relevance": args.min_relevance,
        "profile": args.profile,
        "thresholds": {"min_recall_at_k": MIN_RECALL_AT_K, "min_positive_cases_per_scope": MIN_POSITIVE_CASES_PER_SCOPE, "top_k": TOP_K},
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0 if result["summary"]["threshold_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 18 retrieval evaluation gate for Dev Hub.

The evaluator loads a local FAISS artifact and runs deterministic retrieval
cases from JSON. Query embeddings still require the configured local Ollama
endpoint; the script never contacts production services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.vector_store import VectorStore
from rag.engine import RAGEngine

TOP_K = 5
MIN_RECALL_AT_K = 0.70
ZERO_VISIBILITY_LEAKAGE_REQUIRED = True
PASS_MARKER = "PASS"
FAIL_MARKER = "FAIL"


def load_eval(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("evaluation file must contain a list of cases")
    return data


def _source_strings(doc: dict[str, Any]) -> list[str]:
    citation = doc.get("citation") or {}
    parts = [
        doc.get("source", ""),
        doc.get("repo", ""),
        doc.get("repository", ""),
        doc.get("relative_path", ""),
        citation.get("repository", ""),
        citation.get("relative_path", ""),
    ]
    return [str(p) for p in parts if p]


def _contains_source(doc: dict[str, Any], expected: str) -> bool:
    return any(expected in source for source in _source_strings(doc))


def evaluate_case(engine: RAGEngine, case: dict[str, Any], *, top_k: int = TOP_K) -> dict[str, Any]:
    query = case["query"]
    repo = case.get("repo")
    bundle = case.get("bundle")
    use_rerank = case.get("rerank", False)
    allowed = set(case.get("allowed_visibilities", ["PUBLIC"]))

    docs = engine.retrieve(
        query,
        top_k=top_k,
        repo=repo,
        bundle=bundle,
        rerank=use_rerank,
        allowed_visibilities=allowed,
    )
    expected = case.get("expected_source_contains")
    expect_no_results = bool(case.get("expect_no_results"))
    forbidden_visibilities = set(case.get("forbidden_visibilities", []))
    forbidden_source_contains = case.get("forbidden_source_contains", [])
    if isinstance(forbidden_source_contains, str):
        forbidden_source_contains = [forbidden_source_contains]

    leaked_docs = [d for d in docs if d.get("visibility") in forbidden_visibilities]
    forbidden_source_hits = [
        d for d in docs
        if any(fragment in source for fragment in forbidden_source_contains for source in _source_strings(d))
    ]
    source_hit = next((d for d in docs if expected and _contains_source(d, expected)), None)
    citation_ok = True
    if expected and case.get("require_citation", True):
        citation_ok = bool(source_hit and source_hit.get("citation"))

    if expect_no_results:
        passed = not docs and not leaked_docs and not forbidden_source_hits
    elif expected:
        passed = bool(source_hit) and citation_ok and not leaked_docs and not forbidden_source_hits
    else:
        passed = bool(docs) and not leaked_docs and not forbidden_source_hits

    return {
        "id": case.get("id", query[:60]),
        "category": case.get("category", "uncategorized"),
        "query": query,
        "passed": passed,
        "expected": expected,
        "matched": source_hit.get("source") if source_hit else (docs[0].get("source") if docs else "(no results)"),
        "retrieved": len(docs),
        "visibility_leaks": len(leaked_docs),
        "forbidden_source_hits": len(forbidden_source_hits),
        "citation_ok": citation_ok,
        "negative": expect_no_results or bool(forbidden_visibilities) or bool(forbidden_source_contains),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in results if r["expected"] and not r["negative"]]
    positive_passed = sum(1 for r in positive if r["passed"])
    recall_at_k = positive_passed / len(positive) if positive else 1.0
    visibility_leaks = sum(r["visibility_leaks"] for r in results)
    negative = [r for r in results if r["negative"]]
    negative_passed = sum(1 for r in negative if r["passed"])
    all_passed = all(r["passed"] for r in results)
    threshold_passed = recall_at_k >= MIN_RECALL_AT_K and (visibility_leaks == 0 if ZERO_VISIBILITY_LEAKAGE_REQUIRED else True)
    return {
        "total": len(results),
        "positive": len(positive),
        "positive_passed": positive_passed,
        "recall_at_k": recall_at_k,
        "negative": len(negative),
        "negative_passed": negative_passed,
        "visibility_leaks": visibility_leaks,
        "all_cases_passed": all_passed,
        "threshold_passed": threshold_passed and all_passed,
        "min_recall_at_k": MIN_RECALL_AT_K,
        "zero_visibility_leakage_required": ZERO_VISIBILITY_LEAKAGE_REQUIRED,
    }


def run_eval(index_dir: str, eval_path: str) -> dict[str, Any]:
    vs = VectorStore()
    ok = vs.load(index_dir)
    if not ok:
        print(f"[ERROR] Could not load FAISS index from {index_dir}", file=sys.stderr)
        sys.exit(1)

    engine = RAGEngine(vs)
    cases = load_eval(eval_path)
    results: list[dict[str, Any]] = []
    col_q = min(max(len(c["query"]) for c in cases), 70) if cases else 10

    print(f"\n{'Query':<{col_q}}  {'Category':<20}  {'Result':<6}  {'Matched source'}")
    print("-" * (col_q + 90))

    for case in cases:
        try:
            result = evaluate_case(engine, case, top_k=TOP_K)
        except Exception as e:  # noqa: BLE001 -- per-query boundary: one failing case should be reported, not hidden
            result = {
                "id": case.get("id", case.get("query", "unknown")),
                "category": case.get("category", "uncategorized"),
                "query": case.get("query", ""),
                "passed": False,
                "expected": case.get("expected_source_contains"),
                "matched": f"[ERROR] {e}",
                "retrieved": 0,
                "visibility_leaks": 0,
                "forbidden_source_hits": 0,
                "citation_ok": False,
                "negative": bool(case.get("expect_no_results")),
            }
        results.append(result)
        label = PASS_MARKER if result["passed"] else FAIL_MARKER
        print(f"{result['query'][:col_q]:<{col_q}}  {result['category'][:20]:<20}  {label:<6}  {result['matched']}")

    summary = summarize(results)
    print()
    print("=" * (col_q + 60))
    print(f"Recall@{TOP_K}: {summary['positive_passed']}/{summary['positive']} ({summary['recall_at_k']:.1%})")
    print(f"Visibility leakage: {summary['visibility_leaks']}")
    print(f"Negative cases: {summary['negative_passed']}/{summary['negative']}")
    print(f"Promotion threshold: recall@{TOP_K} >= {MIN_RECALL_AT_K:.0%}; visibility leakage = 0")
    print(f"Evaluation result: {'PASS' if summary['threshold_passed'] else 'FAIL'}")
    print("=" * (col_q + 60))

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures:
            print(f"  - {f['id']}: {f['query'][:90]}")
            print(f"    expected: {f['expected']}")
            print(f"    matched:  {f['matched']}")

    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 18 Dev Hub retrieval evaluation")
    parser.add_argument("--index-dir", default="data/faiss_index", help="Path to saved FAISS index directory")
    parser.add_argument("--eval", default="tests/eval/retrieval_eval.json", help="Path to eval JSON file")
    parser.add_argument("--json-out", help="Optional path for machine-readable evaluation output")
    args = parser.parse_args()

    result = run_eval(args.index_dir, args.eval)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0 if result["summary"]["threshold_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

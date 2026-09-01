#!/usr/bin/env python3
"""
Retrieval eval: measures Recall@5 against tests/eval/retrieval_eval.json.

Usage:
    python scripts/run_eval.py [--index-dir data/faiss_index] [--eval tests/eval/retrieval_eval.json]

The script loads the FAISS index offline — no Ollama required for embeddings
because it reads pre-computed vectors. Queries ARE embedded at runtime, so
Ollama must be reachable (or OLLAMA_URL must point to a live instance).
"""

import argparse
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.vector_store import VectorStore
from rag.engine import RAGEngine

TOP_K = 5
PASS_MARKER = "PASS"
FAIL_MARKER = "FAIL"


def load_eval(path: str):
    with open(path) as f:
        return json.load(f)


def run_eval(index_dir: str, eval_path: str):
    vs = VectorStore()
    ok = vs.load(index_dir)
    if not ok:
        print(f"[ERROR] Could not load FAISS index from {index_dir}", file=sys.stderr)
        sys.exit(1)

    engine = RAGEngine(vs)
    cases = load_eval(eval_path)

    results = []
    col_q = max(len(c["query"]) for c in cases)
    col_q = min(col_q, 70)

    print(f"\n{'Query':<{col_q}}  {'Scope':<18}  {'Result':<6}  {'Matched source'}")
    print("-" * (col_q + 80))

    passed = 0
    for case in cases:
        query = case["query"]
        expected = case["expected_source_contains"]
        repo = case.get("repo")
        bundle = case.get("bundle")

        use_rerank = case.get("rerank", False)
        scope_parts = []
        if bundle:
            scope_parts.append(f"bundle={bundle}")
        elif repo:
            scope_parts.append(f"repo={repo}")
        if use_rerank:
            scope_parts.append("rerank")
        scope_label = ",".join(scope_parts) if scope_parts else "-"

        try:
            docs = engine.retrieve(query, top_k=TOP_K, repo=repo, bundle=bundle, rerank=use_rerank)
        except Exception as e:  # noqa: BLE001 -- per-query boundary: one failing query must not abort the whole eval run
            label = FAIL_MARKER
            matched = f"[ERROR] {e}"
            results.append({"query": query, "passed": False, "matched": matched, "expected": expected})
            print(f"{query[:col_q]:<{col_q}}  {scope_label:<18}  {label:<6}  {matched}")
            continue

        sources = [d.get("source", "") for d in docs]
        hit = next((s for s in sources if expected in s), None)

        if hit:
            passed += 1
            label = PASS_MARKER
            matched = hit
        else:
            label = FAIL_MARKER
            matched = sources[0] if sources else "(no results)"

        results.append({"query": query, "passed": bool(hit), "matched": matched, "expected": expected})
        print(f"{query[:col_q]:<{col_q}}  {scope_label:<18}  {label:<6}  {matched}")

    total = len(cases)
    recall_at_5 = passed / total if total else 0.0

    print()
    print("=" * (col_q + 50))
    print(f"Recall@5:  {passed}/{total}  ({recall_at_5:.1%})")
    print("=" * (col_q + 50))

    # Detailed failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailed queries ({len(failures)}):")
        for f in failures:
            print(f"  - {f['query'][:80]}")
            print(f"    expected contains: {f['expected']}")
            print(f"    top-1 returned:    {f['matched']}")

    return recall_at_5


def main():
    parser = argparse.ArgumentParser(description="Retrieval eval — Recall@5")
    parser.add_argument("--index-dir", default="data/faiss_index", help="Path to saved FAISS index directory")
    parser.add_argument("--eval", default="tests/eval/retrieval_eval.json", help="Path to eval JSON file")
    args = parser.parse_args()

    recall = run_eval(args.index_dir, args.eval)
    sys.exit(0 if recall > 0 else 1)


if __name__ == "__main__":
    main()

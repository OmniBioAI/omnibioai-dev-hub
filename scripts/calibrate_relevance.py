#!/usr/bin/env python3
"""Measure how well cosine similarity separates answerable from unanswerable queries.

Calibrates the Dev Hub relevance cutoff (api/routes/rag.py DEFAULT_MIN_RELEVANCE)
against a candidate index's PUBLIC rows, using tests/eval/relevance_calibration.json,
a query set deliberately disjoint from the frozen evaluation set. Needs the
local Ollama endpoint to embed the queries; reads the index, never writes it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.engine import ollama_embed


def top1_cosines(queries: list[str], unit_vectors: np.ndarray) -> np.ndarray:
    scores = []
    for q in queries:
        e = np.asarray(ollama_embed(q), dtype=np.float32)
        scores.append(float((unit_vectors @ (e / np.linalg.norm(e))).max()))
    return np.array(scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--queries", default="tests/eval/relevance_calibration.json")
    ap.add_argument("--cutoff", type=float, default=0.64, help="cutoff to report against")
    args = ap.parse_args()

    index = faiss.read_index(os.path.join(args.index_dir, "index.faiss"))
    with open(os.path.join(args.index_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)["metadata"]
    with open(args.queries, encoding="utf-8") as f:
        qs = json.load(f)

    rows = [i for i, m in enumerate(meta) if m.get("visibility") == "PUBLIC"]
    vecs = index.reconstruct_n(0, index.ntotal)[rows]
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    a = top1_cosines(qs["answerable"], vecs)
    u = top1_cosines(qs["unanswerable"], vecs)
    auc = float(np.mean([[x > y for y in u] for x in a]))
    print(f"PUBLIC rows: {len(rows)}   answerable: {len(a)}   unanswerable: {len(u)}")
    print(f"answerable   top-1 cosine: min {a.min():.3f}  median {np.median(a):.3f}  max {a.max():.3f}")
    print(f"unanswerable top-1 cosine: min {u.min():.3f}  median {np.median(u):.3f}  max {u.max():.3f}")
    print(f"AUC {auc:.3f}")
    print(f"cutoff {args.cutoff:.3f}: keeps {(a >= args.cutoff).sum()}/{len(a)} answerable, "
          f"rejects {(u < args.cutoff).sum()}/{len(u)} unanswerable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

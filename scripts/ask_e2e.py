#!/usr/bin/env python3
"""End-to-end verification of Ask OmniBioAI against a real PUBLIC-only index and the real LLM.

Runs the actual FastAPI app (real retrieval stack, real query embeddings, real
generation through Ollama) via TestClient, and audits every response against the
Phase 18 + Phase 19 guarantees. Read-only: it loads the index, never writes it.

Hard gates (any failure -> exit 1):
  * zero-context requests never invoke the LLM (counted at the network boundary);
  * INTERNAL / REVIEW_REQUIRED content is never retrieved;
  * client policy fields never widen visibility or lower the 0.64 cutoff;
  * every chunk supplied to generation has relevance >= 0.64;
  * every citation is a PUBLIC chunk that was actually supplied, and its [n] is in the answer;
  * no cited fact from a nonexistent capability: those cases must not be grounded;
  * empty/malformed queries are rejected before embedding or generation.
Measured, not gated: the grounded rate on supported questions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434/api")

from fastapi.testclient import TestClient

import rag.engine as engine_module

CUTOFF = 0.64
REQ_CITE = ("repository", "relative_path", "source_revision", "document_id", "chunk_id", "content_state", "verification_state")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adversarial", default="tests/eval/ask_adversarial.json")
    ap.add_argument("--retrieval-eval", default="tests/eval/retrieval_eval.json")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    calls = {"llm": 0, "embed": 0}
    real_generate, real_embed = engine_module.ollama_generate, engine_module.ollama_embed

    def spy_generate(*a, **k):
        calls["llm"] += 1
        return real_generate(*a, **k)

    def spy_embed(*a, **k):
        calls["embed"] += 1
        return real_embed(*a, **k)

    engine_module.ollama_generate, engine_module.ollama_embed = spy_generate, spy_embed

    import api.main as main_module
    import api.routes.rag as rag_routes

    index_meta = main_module.vector_store.metadata
    public_chunk_ids = {m["chunk_id"] for m in index_meta if m.get("visibility") == "PUBLIC"}
    physical = {}
    for m in index_meta:
        physical[m.get("visibility")] = physical.get(m.get("visibility"), 0) + 1

    failures: list[str] = []
    records: list[dict] = []

    def gate(ok: bool, msg: str) -> None:
        if not ok:
            failures.append(msg)

    with TestClient(main_module.app) as client:
        def ask(path: str, payload: dict):
            before = dict(calls)
            r = client.post(path, json=payload)
            used = {k: calls[k] - before[k] for k in calls}
            if path == "/rag/stream":
                events = [json.loads(line.replace("data: ", "")) for line in r.iter_lines() if line]
                body = next((e for e in events if e["type"] == "response"), {})
                if body:
                    body = {**body, "answer": body["content"], "_events": [e["type"] for e in events]}
                return r.status_code, body, used
            return r.status_code, (r.json() if r.status_code == 200 else {}), used

        def audit(case_id: str, status: int, body: dict, used: dict, path: str) -> dict:
            ctx = body.get("context", [])
            rec = {"id": case_id, "path": path, "http": status, "grounded": body.get("grounded"), "status": body.get("answer_status"),
                   "context_used": body.get("context_used"), "llm_invoked": body.get("llm_invoked"), "llm_calls": used["llm"],
                   "citations": len(body.get("citations", [])), "answer": (body.get("answer") or "")[:240]}
            if status != 200:
                return rec
            gate(body.get("llm_invoked") == (used["llm"] > 0), f"{case_id}: llm_invoked flag disagrees with the network-level call count")
            if body.get("context_used") == 0:
                gate(used["llm"] == 0, f"{case_id}: zero context but the LLM was called")
                gate(body.get("citations") == [] and body.get("grounded") is False, f"{case_id}: zero-context response was grounded/cited")
            if not body.get("grounded"):
                gate(body.get("citations") == [], f"{case_id}: ungrounded response carries citations")
            for d in ctx:
                gate(d.get("visibility") == "PUBLIC", f"{case_id}: non-PUBLIC chunk in context ({d.get('visibility')})")
                gate((d.get("relevance") or 0) >= CUTOFF, f"{case_id}: chunk below the {CUTOFF} cutoff supplied ({d.get('relevance')})")
            answer = body.get("answer") or ""
            for c in body.get("citations", []):
                gate(all(c.get(k) for k in REQ_CITE), f"{case_id}: citation missing provenance {c}")
                gate(c.get("chunk_id") in public_chunk_ids, f"{case_id}: citation is not a PUBLIC indexed chunk")
                gate(f"[{c['index']}]" in answer or re.search(rf"\[[\d,\s]*\b{c['index']}\b[\d,\s]*\]", answer) is not None, f"{case_id}: citation [{c['index']}] not in the answer text")
            if body.get("grounded"):
                gate(bool(body.get("citations")), f"{case_id}: grounded answer without citations")
            return rec

        # ---- supported questions (PUBLIC eval cases)
        with open(args.retrieval_eval, encoding="utf-8") as f:
            supported = [c for c in json.load(f) if c["id"].startswith("public-")]
        grounded_n = 0
        for c in supported:
            status, body, used = ask("/rag/query", {"query": c["query"]})
            rec = audit(c["id"], status, body, used, "/rag/query")
            rec["expected_source"] = c["expected_source_contains"]
            rec["cites_expected"] = any(c["expected_source_contains"].split("/", 1)[-1] in (x.get("relative_path") or "") for x in body.get("citations", []))
            rec["category"] = "supported"
            grounded_n += bool(body.get("grounded"))
            records.append(rec)

        # ---- adversarial cases
        with open(args.adversarial, encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        for c in cases:
            cid, cat = c["id"], c["category"]
            payload = {"query": c["question"], **c.get("extra", {})}
            status, body, used = ask("/rag/query", payload)
            rec = audit(cid, status, body, used, "/rag/query")
            rec["category"] = cat
            if "expect_http" in c:
                gate(status == c["expect_http"], f"{cid}: expected HTTP {c['expect_http']} got {status}")
                gate(used["llm"] == 0 and used["embed"] == 0, f"{cid}: malformed query reached embedding/generation {used}")
            else:
                if "expect_grounded" in c:
                    gate(body.get("grounded") == c["expect_grounded"], f"{cid}: expected grounded={c['expect_grounded']} got {body.get('grounded')} ({body.get('answer_status')}): {rec['answer']!r}")
                if "expect_llm_invoked" in c:
                    gate(body.get("llm_invoked") == c["expect_llm_invoked"], f"{cid}: expected llm_invoked={c['expect_llm_invoked']}")
                if "expect_status" in c:
                    gate(body.get("answer_status") == c["expect_status"], f"{cid}: expected status {c['expect_status']} got {body.get('answer_status')}")
                if "forbid_repos_not" in c:
                    bad = [d["repo"] for d in body.get("context", []) if d.get("repo") not in c["forbid_repos_not"]]
                    gate(not bad, f"{cid}: content from non-PUBLIC-corpus repos retrieved: {bad}")
            if cat == "bypass":
                _plain_status, plain, _ = ask("/rag/query", {"query": c["question"]})
                rec["same_as_plain"] = (plain.get("context_used"), plain.get("answer_status")) == (body.get("context_used"), body.get("answer_status"))
                gate(rec["same_as_plain"], f"{cid}: bypass fields changed the outcome ({body.get('answer_status')} vs {plain.get('answer_status')})")
            records.append(rec)

        # ---- streaming: same contract, same guarantees
        stream_cases = [("stream-no-context", "How do I bake sourdough bread"), ("stream-named-nonexistent", "What does the OmniBioAI iOS mobile app do?"),
                        ("stream-supported", "How do I back up and restore the platform after a disaster"), ("stream-internal", "What is on the OmniBioAI roadmap?")]
        for cid, q in stream_cases:
            status, body, used = ask("/rag/stream", {"query": q})
            rec = audit(cid, status, body, used, "/rag/stream")
            rec["category"] = "streaming"
            rec["events"] = body.get("_events")
            gate(rec["events"] == ["status", "response", "done"], f"{cid}: unexpected event sequence {rec['events']}")
            if cid in ("stream-no-context", "stream-named-nonexistent"):
                gate(body.get("grounded") is False and used["llm"] == 0, f"{cid}: not refused without the LLM")
            records.append(rec)

    gate(physical.get("INTERNAL", 0) == 0 and physical.get("REVIEW_REQUIRED", 0) == 0, f"index physically contains non-PUBLIC chunks: {physical}")
    gate(rag_routes.DEFAULT_MIN_RELEVANCE == CUTOFF and rag_routes._min_relevance() == CUTOFF, "cutoff is not 0.64")

    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    summary = {
        "index_physical_visibility": physical,
        "supported": {"asked": len(supported), "grounded": grounded_n, "grounded_rate": round(grounded_n / len(supported), 3),
                      "no_trusted_context": sum(1 for r in by_cat["supported"] if r["status"] == "NO_TRUSTED_CONTEXT"),
                      "other_ungrounded": sum(1 for r in by_cat["supported"] if not r["grounded"] and r["status"] != "NO_TRUSTED_CONTEXT"),
                      "grounded_and_cites_expected_doc": sum(1 for r in by_cat["supported"] if r["grounded"] and r["cites_expected"])},
        "total_llm_calls": calls["llm"], "total_embed_calls": calls["embed"],
        "gate_failures": failures, "gates_passed": not failures,
    }
    print(json.dumps(summary, indent=2))
    for cat in ("nonexistent_named", "nonexistent_lowercase", "off_topic", "internal_probe", "bypass", "borderline", "malformed", "streaming"):
        print(f"\n== {cat} ==")
        for r in by_cat.get(cat, []):
            print(f"  {r['id']:28} http={r['http']} grounded={r['grounded']} status={r['status']} ctx={r['context_used']} llm={r['llm_calls']} | {r['answer'][:95]!r}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

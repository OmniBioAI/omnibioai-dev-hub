# Dev Hub V7 retrieval evaluation endpoint recovery

Date: 2026-10-15  
Workstream: `OMNIBIOAI_DEV_HUB_V7_EVALUATION_ENDPOINT_RECOVERY_V1`

## Decision

The mixed-visibility candidate `data/faiss_candidates/devhub-41b13f3a1a32` passes its frozen full-profile qualification gate. It must not be promoted directly as the production artifact: production startup requires a separately derived, verified, and production-profile-evaluated PUBLIC-only artifact. Recommendation: **NO-GO for direct promotion; GO to the controlled PUBLIC-only derivation and qualification stage.**

No candidate was promoted, no active index was changed, no service was restarted or activated, no database was modified, and the recovering HDD was not accessed.

## Root cause and configuration

`scripts/run_eval.py` and `scripts/calibrate_relevance.py` both call `rag.engine.ollama_embed`. At import time, `rag/engine.py` resolves the endpoint as:

1. `OLLAMA_URL`, when explicitly present in the process environment;
2. otherwise `http://ollama:11434/api`.

`OLLAMA_HOST` is not consulted. The Python application does not auto-load `.env`; `.env.example` and the README already document the host override. The suspended host-side evaluation PID 1641599 had no `OLLAMA_URL` in its environment, so it inherited the Docker-oriented default and attempted DNS host `ollama:11434` from the host namespace. There is no endpoint fallback.

The recovery was configuration-only for execution:

```bash
env OLLAMA_URL=http://127.0.0.1:11434/api python scripts/run_eval.py \
  --index-dir data/faiss_candidates/devhub-41b13f3a1a32 \
  --eval tests/eval/retrieval_eval.json \
  --profile full \
  --json-out /tmp/devhub-v7-evaluation.json
```

Docker behavior remains backward-compatible because the unset default is unchanged. Focused regression coverage was added without changing production code.

## Process and worktree audit

- PID 1641599 remains stopped (`Tl`) and was neither resumed nor terminated.
- Initial worktree already contained user-owned changes: modified `scripts/build_index.py` and untracked `qualifications/`.
- This workstream changed only `tests/test_rag_engine.py` and this report.
- A pre-existing `omnibioai-studio-dev-hub-1` container was observed in Docker's `Restarting` state; this workstream did not act on it.

## Ollama compatibility

- Configured endpoint: `http://127.0.0.1:11434/api`
- Ollama version: `0.40.3`
- Model: `nomic-embed-text:latest`
- Model digest: `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`
- Model format/family: GGUF / nomic-bert, F16, 137M
- Declared and observed embedding length: 768
- Smoke response: successful `float32[768]`
- Endpoint ownership: host port 11434 maps to the single already-running `omnibioai-studio-ollama-1` container. No alternate Ollama endpoint or fallback was used.

## Candidate integrity and provenance

The manifest was parsed and both artifact hashes were recomputed before `metadata.pkl` was loaded. The repository verifier then passed the full metadata contract.

```text
INDEX OK: 35676 vectors, dim 768, visibility {'INTERNAL': 33345, 'PUBLIC': 2331, 'REVIEW_REQUIRED': 0}, public_only=False
```

- Build ID: `devhub-41b13f3a1a32`
- Build status: `CLEAN`
- Evaluation status in manifest: `NOT_EVALUATED` (intentionally not rewritten)
- Promotion status: `CANDIDATE`
- Backend: `FAISS IndexFlatIP`
- Embedding: `ollama` / `nomic-embed-text` / 768 dimensions
- Build telemetry: 35,676 successful embeddings; 0 failed embeddings; 0 retries
- `index.faiss`: `afa6c1823f39bac277ce665b4db4813fe2d1dc0ebe9414a967bdf0ed7116231c`
- `metadata.pkl`: `8c239c3659a1f15befb8eec95d60bf65cf103edfe31dfaa1788b8f70ceb55140`
- `manifest.json`: `768438146fed621d82c2efc2053c9f593a44a86ea0575c5d713209e4b04da93e`

Hashes, sizes, mtimes, and inodes were unchanged after evaluation and calibration. Evaluation wrote only `/tmp/devhub-v7-evaluation.json` (SHA-256 `0b47eb2c46d3d211921d5f3461a3e0865bada492f14ffd304e581b4b5e085675`). The explicit candidate path and read-only `VectorStore.load` call do not reference or mutate the active index.

## Retrieval evaluation

Frozen full-profile result: **PASS**.

- Total: 66
- Individually passed: 61
- Individually failed: 5
- Infrastructure errors: 0
- PUBLIC positive recall@5: 24/28 (85.7%), gate PASS
- INTERNAL positive recall@5: 32/33 (97.0%), gate PASS
- Safety/negative cases: 5/5, gate PASS
- Visibility leaks: 0
- Duplicate results: 0
- Raw FAISS top-score range: INTERNAL 278.827–389.675 (median 339.955); PUBLIC 227.626–422.608 (median 306.376)

The five failures are genuine relevance misses, not infrastructure failures:

- `internal-legacy-01`: expected workflow-bundles ATAC-seq; matched Workbench ATAC peak-calling.
- `public-02`: expected admin installation; matched user getting-started.
- `public-06`: expected admin deployment; matched Azure DevOps integration.
- `public-07`: expected admin configuration; matched Vault integration.
- `public-16`: expected developer repository map; matched Quay integration.

The harness reports raw IndexFlatIP scores. Comparable cosine relevance was measured separately by calibration below.

## Relevance calibration at 0.64

- PUBLIC rows: 2,331
- Answerable queries: 40; top-1 cosine min/median/max 0.582/0.753/0.868
- Unanswerable queries: 40; top-1 cosine min/median/max 0.445/0.511/0.738
- AUC: 0.984
- At cutoff 0.64: retained 39/40 answerable; rejected 38/40 unanswerable

The production cutoff was not changed.

## Security and regression gates

- Live full-profile evaluation: 5/5 negative/security cases passed; zero visibility leaks.
- Security category results: visibility-leakage 2/2, unsupported-claim 2/2, index-separation 1/1, configured-vs-verified 1/1, target-vs-current 1/1, citation-correctness 1/1.
- Focused endpoint/evaluation suite: 65 passed.
- Filter, lifecycle, and PUBLIC-only derivation suites: 62 passed.
- FastAPI route-policy suite: 40 passed, with one Starlette deprecation warning. The base interpreter lacked FastAPI, so the suite was rerun successfully with an existing `/tmp/omnibioai-rc2-auth-venv` environment; no package was installed.
- Route tests confirm callers cannot widen visibility to INTERNAL/REVIEW_REQUIRED or lower the server-side 0.64 cutoff.
- Derivation tests confirm the PUBLIC-only artifact physically excludes INTERNAL and REVIEW_REQUIRED rows and fails closed on invalid visibility.
- Candidate contains zero REVIEW_REQUIRED rows and remained byte-identical throughout qualification.

## Remaining blockers and next controlled stage

The mixed candidate is qualified as a source candidate but is not directly production-promotable. A PUBLIC-only derived candidate has not been created or production-profile evaluated. The five relevance misses are below the frozen gate's allowed failure budget but should remain visible as quality debt. The original stopped evaluation PID 1641599 remains for the operator to disposition.

With separate approval, the next non-promotion command is:

```bash
python scripts/derive_public_index.py \
  --source data/faiss_candidates/devhub-41b13f3a1a32 \
  --out-root data/faiss_candidates
```

That derived artifact must then pass `verify_index.py --require-public-only` and `run_eval.py --profile production` using the same explicit `OLLAMA_URL`. Do not run `promote_index.py` until those gates pass and the stopped PID/service state is deliberately handled.

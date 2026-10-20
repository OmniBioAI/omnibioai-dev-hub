# Dev Hub V7 PUBLIC-only index qualification

Date: 2026-10-15  
Workstream: `OMNIBIOAI_DEV_HUB_V7_PUBLIC_INDEX_QUALIFICATION_V1`

## Decision

The derived candidate `data/faiss_candidates/devhub-41b13f3a1a32-public` is technically qualified: structural verification, frozen production-profile evaluation, provenance, vector alignment, PUBLIC-only isolation, and security regression gates all pass. The artifact is a **GO candidate**.

Activation is presently **NO-GO pending operational controls**: rotate the secret-bearing runtime credential exposed during read-only inspection, disposition stopped PID 1641599, freeze the existing restart loop, preserve an external recovery copy for this first activation, and approve a maintenance window. No promotion or activation was performed.

## Preflight and preserved state

- Read `qualifications/OMNIBIOAI_DEV_HUB_V7_EVALUATION_ENDPOINT_RECOVERY_V1.md` before work.
- Initial Git state: modified `scripts/build_index.py`, modified `tests/test_rag_engine.py`, and untracked `qualifications/`; all were preserved.
- Source hashes matched the prior report exactly before derivation.
- PID 1641599 still exists in stopped state (`Tl`) with the old mixed-index evaluation command. It was not resumed or terminated. Operator cleanup is required because resuming it could overwrite `/tmp/devhub-v7-evaluation.json`.
- `data/faiss_index` was absent/empty before derivation and remained so. The active production index was not changed.
- The default output path did not exist before derivation, preventing overwrite.

Source trust anchor:

- `index.faiss`: `afa6c1823f39bac277ce665b4db4813fe2d1dc0ebe9414a967bdf0ed7116231c`
- `metadata.pkl`: `8c239c3659a1f15befb8eec95d60bf65cf103edfe31dfaa1788b8f70ceb55140`
- `manifest.json`: `768438146fed621d82c2efc2053c9f593a44a86ea0575c5d713209e4b04da93e`

## Derivation

Executed:

```bash
python scripts/derive_public_index.py \
  --source data/faiss_candidates/devhub-41b13f3a1a32 \
  --out-root data/faiss_candidates
```

Result:

- Build ID: `devhub-41b13f3a1a32-public`
- PUBLIC documents: 152
- PUBLIC chunks/vectors: 2,331
- Excluded INTERNAL: 1,926 documents / 33,345 chunks
- Excluded REVIEW_REQUIRED: 0 documents / 0 chunks
- No embedding calls were made.
- Source PUBLIC vectors were copied bit-for-bit in source-row order.
- Derived metadata equals the source PUBLIC metadata in exact order.
- Derived chunk IDs have zero intersection with source INTERNAL chunk IDs.
- The mixed source and active-index path were not modified.

The derivation contract validates the source before loading it, rejects unknown visibility values, writes through a partial directory, validates the completed artifact, compares vectors and metadata after reload, and atomically renames the completed candidate into place.

## Structural verification and provenance

Executed:

```bash
python scripts/verify_index.py \
  data/faiss_candidates/devhub-41b13f3a1a32-public \
  --require-public-only
```

Result:

```text
INDEX OK: 2331 vectors, dim 768, visibility {'INTERNAL': 0, 'PUBLIC': 2331, 'REVIEW_REQUIRED': 0}, public_only=True
```

Manifest contract:

- Schema: `devhub.index-manifest.v1`
- Build status: `CLEAN`
- Promotion status: `CANDIDATE`
- Evaluation status: `NOT_EVALUATED` (the evaluator does not mutate manifests)
- Visibility policy: `PUBLIC_ONLY`
- Backend: `FAISS IndexFlatIP`
- Embedding provider/model/dimension: `ollama` / `nomic-embed-text` / 768
- Embedding identity: `ollama:nomic-embed-text:revision-unknown`
- Parent build: `devhub-41b13f3a1a32`, 35,676 chunks, `CLEAN`
- Parent visibility: PUBLIC 2,331; INTERNAL 33,345; REVIEW_REQUIRED 0
- Parent manifest and artifact hashes exactly match the trust anchor above.

Derived hashes:

- `index.faiss`: `6f523ce2adc7ac0d0b73a31f5942b4c42625eb4cd3f97676fd688563c8a324af`
- `metadata.pkl`: `99f187bfba060b26f5297da99afab43f63d79978b836c92ede8bd22852900024`
- `manifest.json`: `3150dd2c2159fbabd7288ee1d91533f43f7c16f842decfffcba8838739df3f47`
- Derivation report: `48e94797c408fa910c225d88827760496f9480c1fe81f54ce74893af5e1851a3`

## Frozen production-profile evaluation

Executed:

```bash
env OLLAMA_URL=http://127.0.0.1:11434/api python scripts/run_eval.py \
  --index-dir data/faiss_candidates/devhub-41b13f3a1a32-public \
  --eval tests/eval/retrieval_eval.json \
  --profile production \
  --json-out /tmp/devhub-v7-public-production-eval.json
```

Result: **PASS**.

- Scored production cases: 33
- Passed: 29
- Failed: 4
- Informational INTERNAL probes: 33
- Expected INTERNAL sources found by probes: 0/33
- PUBLIC positive recall@5: 24/28 (85.7%); frozen gate is at least 70% with at least 10 cases
- Safety cases: 5/5
- Visibility leaks: 0
- Duplicate results: 0
- Infrastructure errors: 0
- Threshold gate: PASS
- Evaluation JSON SHA-256: `f746ffe76e24d076d01cc3f42e951ea7961d15fe98bb77fa1595d2bad9504a0c`

The four scored failures are genuine relevance misses: `public-02`, `public-06`, `public-07`, and `public-16`. No fixture or threshold was changed.

## Independent security qualification

The following were checked independently of the manifest declaration:

- FAISS count equals metadata count: 2,331.
- Every serialized metadata row has visibility exactly `PUBLIC`.
- Derived metadata equals the source candidate's PUBLIC-row metadata in exact order.
- Every derived vector is bit-identical to the corresponding source PUBLIC vector.
- Derived chunk IDs intersect source INTERNAL chunk IDs at count zero.
- Direct FAISS top-5 probes using rows 0, 1165, and 2330 mapped only to PUBLIC metadata.
- Retrieval filtering, lifecycle validation, derivation, startup, and promotion tests passed.
- API route tests confirm clients cannot widen visibility to INTERNAL/REVIEW_REQUIRED or lower the server cutoff.
- Production evaluation produced zero leaks, and no INTERNAL expected source was found in 33 probes.
- Startup reads only `/app/data/faiss_index`; candidates are ignored. Live logs show fail-closed refusal while the active index is absent, proving there is no silent mixed-candidate fallback.

Focused tests:

```bash
python -m pytest -q \
  tests/test_rag_engine.py tests/test_run_eval.py \
  tests/test_filtered_search.py tests/test_index_lifecycle.py \
  tests/test_derive_public_index.py tests/test_promote_index_cli.py \
  tests/test_startup_safety.py
# 142 passed

env PYTHONPATH=/home/manish/Desktop/omnibioai-ecosystem/omnibioai-dev-hub \
  /tmp/omnibioai-rc2-auth-venv/bin/python -m pytest -q tests/test_rag_routes.py
# 40 passed, one Starlette deprecation warning

env PYTHONPATH=/home/manish/Desktop/omnibioai-ecosystem/omnibioai-dev-hub:/tmp/omnibioai-rc2-auth-venv/lib/python3.14/site-packages \
  python -m pytest -q tests/test_main_api.py
# 8 passed; 2 async tests not executed because pytest-asyncio is unavailable
```

The first combined FastAPI/load-path attempt had a temporary-environment `index` package collision. Explicit `PYTHONPATH` resolved the route suite. A subsequent load-path run passed all eight synchronous assertions; its two async functions were reported unsupported because no installed interpreter has FastAPI, FAISS, and pytest-asyncio together. No product assertion failed in those two tests, but this remains a test-environment gap.

Total executed passing regression assertions: 190. Unexecuted because of dependency gap: 2 async load-path tests.

## Review of the five prior misses

All expected sources are present, so none is a missing-source-coverage failure or an evaluation-expectation mismatch.

| Case | Expected chunks | Best expected rank | Primary classification | Evidence |
|---|---:|---:|---|---|
| `internal-legacy-01` | 40 | 6 | Ranking problem | Correct ATAC-seq document is one position below top-5; related Workbench/scATAC material outranks it. |
| `public-02` | 9 | 16 | Semantic retrieval mismatch | Broad installation query favors getting-started and section-index content. |
| `public-06` | 2 | 11 | Ranking problem with possible chunking contribution | Deployment document exists but has only two chunks; integration and compose documents rank above it. |
| `public-07` | 12 | 71 | Query formulation/semantic mismatch | Very broad “configuration options” query strongly favors specific integration configuration pages. |
| `public-16` | 11 | 8 | Ranking problem | Repository-map content is near the cutoff but registry/integration pages occupy top-5. |

Future improvement recommendation: evaluate query-aware reranking and document-title/path signals on a separate, non-frozen development set; review deployment-document chunk boundaries; add narrowly scoped synonym/title enrichment for admin installation, deployment, configuration, and repository-map concepts. Re-run the frozen set only as a final unbiased gate. Do not weaken top-k, thresholds, or fixtures.

## Deployment and activation readiness

Observed deployment:

- Compose project/service: `omnibioai-studio` / `dev-hub`
- Compose files: `omnibioai-studio/docker-compose.yml` and `docker-compose.override.yml`
- Host `omnibioai-dev-hub/data` is bind-mounted read-write at `/app/data`.
- Active index: host `data/faiss_index`, container `/app/data/faiss_index`.
- Startup default: `DEVHUB_INDEX_DIR=/app/data/faiss_index` when unset.
- Runtime Ollama URL: `http://ollama:11434/api`.
- Container starts as root only to chown `/app/data`, then runs as `appuser` UID 10001.
- Candidate directory and artifacts are currently owned by UID/GID 10001 and are readable/writable as required (`775` directories, `664` artifacts).
- Restart policy is `on-failure`; the service is currently restart-looping because the active index is absent.

Security incident note: an unfiltered read-only Docker environment inspection returned a secret-bearing value in tool output. The value is intentionally not copied here. Rotate that credential and any dependent session material before activation.

### Controlled promotion procedure (not executed)

The checked-in `scripts/promote_index.py` validates PUBLIC-only status, copies and re-hashes a source, swaps it into `faiss_index`, and retains an existing active index under `faiss_previous`. It must run with the service stopped because the swap is two atomic renames, not one atomic transaction.

The qualified artifact currently already occupies `data/faiss_candidates/<build-id>`. Passing that path to `promote_index.py --data-dir data` would be refused because the CLI copy destination already exists. Therefore the approved maintenance procedure must first preserve a verified release copy outside `/app/data`, then free the in-data candidate destination before invoking the CLI. Do not improvise this step during the window.

Required sequence:

1. Rotate the exposed runtime credential and validate authentication.
2. Deliberately terminate or otherwise disposition stopped PID 1641599.
3. Copy the qualified candidate and manifest to an operator-approved external release-staging directory; recompute all hashes there.
4. Freeze the `on-failure` restart policy and stop the `dev-hub` service.
5. Reconfirm `/app/data/faiss_index` state. If an active index appears before the window, hash and preserve it as the rollback target.
6. Move the in-data candidate aside only after the external release copy is verified, so `promote_index.py` can create its destination.
7. Run `scripts/promote_index.py promote` as UID 10001 in the Dev Hub image, with the external source mounted read-only and `/app/data` mounted read-write.
8. Run `scripts/verify_index.py /app/data/faiss_index --require-public-only` and compare active artifact hashes with this report.
9. Restore the intended restart policy and start the service once.
10. Perform the smoke/security tests below. Roll back or stop on any mismatch or leak.

Illustrative promotion invocation for the approved window (paths must be resolved and hash-checked first):

```bash
docker run --rm --user 10001:10001 --entrypoint python \
  -v <EXTERNAL_RELEASE_DIR>:/candidate:ro \
  -v /home/manish/Desktop/omnibioai-ecosystem/omnibioai-dev-hub/data:/app/data:rw \
  omnibioai-studio-dev-hub \
  scripts/promote_index.py promote --source /candidate --data-dir /app/data
```

This command was documented only and was not executed.

### Backup and rollback

- If an active index exists at promotion time, the lifecycle code moves it under `/app/data/faiss_previous/previous-*` and records the exact path and hashes in the new active manifest. `scripts/promote_index.py rollback --data-dir /app/data` then swaps it back while retaining the failed forward index under `faiss_rolled`.
- At present there is no active index, so the first activation has no previous production artifact to roll back to. The independently hash-verified external release copy is mandatory recovery material; rollback means stopping the service and restoring a known-good active directory or returning to the current fail-closed/no-service state.
- Do not rely solely on the in-data candidate because successful promotion consumes its copied destination during the swap.

### Post-activation acceptance tests

1. `verify_index.py /app/data/faiss_index --require-public-only` passes with 2,331/768 and the recorded hashes.
2. Active manifest says `PUBLIC_ONLY`, INTERNAL=0, REVIEW_REQUIRED=0, and records the expected parent.
3. Container starts once without a restart loop; `/health` and `/status` succeed.
4. Ollama connectivity succeeds through Docker service DNS using `nomic-embed-text` at 768 dimensions.
5. A known PUBLIC query returns only PUBLIC citations.
6. Known INTERNAL queries do not return any INTERNAL source or identifier.
7. Attempts to widen visibility or lower relevance through API input remain ineffective.
8. Authentication remains enabled and valid after credential rotation.
9. Logs contain no load, hash, authorization, visibility, or embedding errors.
10. Recompute active artifact hashes after startup; they must remain unchanged.

Any hash mismatch, non-PUBLIC metadata, INTERNAL source discovery, authorization bypass, unexpected fallback, repeated restart, or failed health check is an immediate rollback/stop condition.

## Commands and evidence inventory

In addition to the derivation, verification, evaluation, and test commands shown above, the qualification used read-only commands to:

- read the prior report and relevant scripts (`sed`);
- inspect Git and PID state (`git status`, `ps`);
- hash artifacts (`sha256sum`);
- inspect manifests and evaluation JSON (`jq`);
- inspect candidate layout, permissions, and active-index absence (`find`, `stat`);
- compare source/derived metadata and vectors and rank expected documents (read-only Python diagnostics using FAISS/NumPy);
- inspect the Dev Hub bind mount, UID, restart policy, Compose labels, and state (`docker inspect`);
- inspect redacted startup failures (`docker logs --tail 80` with secret-pattern redaction).

No promotion command, container restart/stop/start/update, database command, or HDD access occurred.

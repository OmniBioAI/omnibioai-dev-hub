# Dev Hub V7 pre-activation security and rollback closure

Date: 2026-10-15  
Workstream: `OMNIBIOAI_DEV_HUB_V7_PRE_ACTIVATION_SECURITY_AND_ROLLBACK_CLOSURE_V1`

## Executive summary

The qualified PUBLIC-only artifact remains intact and passes structural verification, frozen production evaluation, visibility isolation, recovery-copy verification, isolated promotion/rollback rehearsal, both previously blocked async tests, and the full 550-test repository suite.

Three minimal source hardenings were implemented:

1. application loading now honors the same trusted `DEVHUB_INDEX_DIR` that the entrypoint validates;
2. non-debug `/rag/query` failures no longer return raw exception strings that could contain credentials or paths;
3. promotion accepts an already-canonical staged candidate and copies external candidates through a verified hidden partial directory before same-filesystem atomic rename.

The requested technical preparation is complete, but `PRE_ACTIVATION_GATES` is **BLOCKED**, not PASS, because two live-state actions require explicit authorization: coordinated rotation/revocation of the exposed shared JWT secret across 12 services, and disposition of stopped PID 1641599. The live Dev Hub also remains in its pre-existing restart loop until an explicitly authorized promotion/rebuild/start window. No live service, credential, database, or active index was changed.

## Initial blockers and preflight inventory

- Existing reports were read and preserved.
- Branch: `main` in Dev Hub; pre-existing changes to `scripts/build_index.py`, `tests/test_rag_engine.py`, and `qualifications/` were preserved.
- Source mixed candidate: `data/faiss_candidates/devhub-41b13f3a1a32`.
- Qualified PUBLIC candidate: `data/faiss_candidates/devhub-41b13f3a1a32-public`.
- Active host path: `data/faiss_index`; container path: `/app/data/faiss_index`.
- Active index: absent before and after this workstream.
- Dev Hub data bind mount: host repository `data` to `/app/data`, read-write.
- Candidate and recovery storage: local NVMe root filesystem `/dev/nvme0n1p2`, ext4. The recovering 26 TB HDD was not accessed.
- Image: Linux ARM64; no memory/CPU limit; no OOM; UID 10001 owns the runtime data tree.
- Runtime Ollama is healthy and reachable on host port 11434.
- Dev Hub restart policy: `on-failure`; no Docker health check is configured.

Both trust anchors remained unchanged:

| Artifact | Mixed SHA-256 | PUBLIC SHA-256 |
|---|---|---|
| `index.faiss` | `afa6c1823f39bac277ce665b4db4813fe2d1dc0ebe9414a967bdf0ed7116231c` | `6f523ce2adc7ac0d0b73a31f5942b4c42625eb4cd3f97676fd688563c8a324af` |
| `metadata.pkl` | `8c239c3659a1f15befb8eec95d60bf65cf103edfe31dfaa1788b8f70ceb55140` | `99f187bfba060b26f5297da99afab43f63d79978b836c92ede8bd22852900024` |
| `manifest.json` | `768438146fed621d82c2efc2053c9f593a44a86ea0575c5d713209e4b04da93e` | `3150dd2c2159fbabd7288ee1d91533f43f7c16f842decfffcba8838739df3f47` |

## Credential exposure assessment

### Identification and scope

The affected credential is `JWT_SECRET`, a shared HS256 validation/internal-proxy secret. In Dev Hub it validates bearer tokens and the nginx-injected first-party UI header. Other platform consumers also use it for JWT validation and, in some services, event-signature verification.

Value-suppressed inspection established:

- the running Dev Hub value is identical across 12 current `omnibioai-studio` containers: API gateway, billing service, control center, Dev Hub, LIMS, model registry, RAG, security-audit API, security-audit worker, TES, Workbench, and workflow bundles;
- Compose contains external `${JWT_SECRET...}` interpolation at all 12 sites; it does not commit the runtime value;
- the tracked Compose history was scanned across 204 revisions: only the interpolation expression was present, and the runtime value was not present in Git history;
- the checked Studio `.env` exists with mode 0600 but does not currently define `JWT_SECRET`;
- the current shell and Bash history do not contain the key assignment;
- the value is present in Docker container environment metadata and was disclosed once through the prior unredacted inspection output;
- no plaintext secret was added to Dev Hub Git content or this report.

Exposure scope is therefore the prior tool transcript plus access available to principals permitted to inspect Docker container configuration. Because the value is shared, compromise could affect all 12 consumers, not only Dev Hub.

### Log review

Recent logs from API gateway, control center, Dev Hub, and RAG were searched with credential redaction. No JWT signature/invalid-token signal was found in the sampled API gateway, Dev Hub, or RAG logs. Control center had routine 401 responses, but the available entries do not attribute them to this credential. The same log sample also contained unrelated signed registry URLs, so raw log output is intentionally omitted. This limited review cannot prove non-use.

### Rotation status and required operator action

Rotation was **not authorized and not executed**. The old credential is therefore **not revoked**.

Supported rotation uses the existing Compose interpolation and must be coordinated across all 12 current consumers. The operator must:

1. Open a platform maintenance window and preserve the old value only in a protected, non-logged operator session for revocation testing.
2. Generate a cryptographically random replacement in a secret manager or the mode-0600, Git-ignored Studio `.env`; do not pass it on a command line or paste it into tickets/logs.
3. Freeze automatic restarts and recreate all 12 affected services from the same new source of truth. Partial rotation would split JWT/internal-header trust and is not acceptable.
4. Verify every recreated container has the new value by equality checks performed in-process without printing either value.
5. Verify an old signed token/internal header is rejected and a newly issued valid identity succeeds at protected endpoints.
6. Revoke active sessions as required by platform policy. Symmetric-key replacement invalidates old HS256 signatures at the recreated consumers; there is no separate key-revocation endpoint in this repository.
7. Review authentication/audit logs for unexpected old-key attempts after cutover.
8. Remove any protected temporary copy of the old credential after verification.

This requires authorization to recreate multiple live services and coordinate user-session impact. Exact affected service names are listed above; no unrelated credential should be rotated.

## Runtime restart-loop diagnosis and stabilization

Observed state reached restart count 849, exit code 1, with no OOM. The root cause is singular and expected: `/app/data/faiss_index/index.faiss` is absent. The entrypoint validates only `/app/data/faiss_index`, refuses to build/promote/fall back, and exits before nginx/uvicorn. Permissions, bind mount, ARM64 image compatibility, CPU/memory, and Ollama are not causal.

No live stabilization action was authorized. Creating a fake index, using the mixed index, weakening validation, or disabling the fail-closed behavior would be unsafe.

One source consistency defect was fixed: `api/main.py` previously ignored `DEVHUB_INDEX_DIR` even though the entrypoint validated it. It now loads the exact trusted process-configured path, defaulting to the same production path. Request/API input cannot select an index path. This fix requires a rebuilt Dev Hub image during the separately authorized activation window.

Operator stabilization sequence for that window:

```text
1. Rotate/revoke JWT_SECRET across all affected services.
2. Set the Dev Hub restart policy to no/disabled and stop the restart loop.
3. Build the Dev Hub image containing the qualified source changes.
4. Promote only the recorded PUBLIC candidate under the controlled procedure below.
5. Verify the active artifact before starting Dev Hub once.
6. Restore the intended restart policy only after smoke/security acceptance passes.
```

No command in this sequence was executed here.

## Suspended PID disposition

PID 1641599 still exists and is positively identified as the original mixed-candidate `scripts/run_eval.py` invocation:

- executable: host Miniforge Python 3.14;
- state: stopped (`T`/`Tl`), user UID 1000;
- parent: the VS Code integrated-terminal Bash process;
- working directory: this Dev Hub repository;
- resident memory: approximately 254 MiB; 20 threads;
- open descriptors: terminal/VS Code log descriptors only; no candidate/index files and no sockets;
- GPU allocation: none found.

It was not terminated because process destruction was not explicitly authorized. Safe operator cleanup, after reconfirming the PID/command, is:

```bash
kill -TERM 1641599
kill -CONT 1641599
ps -p 1641599 -o pid,stat,args
```

`SIGTERM` is queued while stopped; `SIGCONT` permits immediate delivery. If it remains after a reasonable check, escalation to `kill -KILL 1641599` requires a deliberate operator decision. Expected effect: release roughly 254 MiB and its terminal descriptors. The old `/tmp/devhub-v7-evaluation.json` may be retained as evidence; the new evaluation uses a different filename.

## Independent recovery copy

Created without overwrite at:

`data/faiss_recovery/devhub-41b13f3a1a32-public-qualified-20261015`

The path is on approved local NVMe ext4 storage. It contains only `index.faiss`, `metadata.pkl`, and `manifest.json`; all are byte-identical to the qualified PUBLIC source. Trusted verification result:

```text
INDEX OK: 2331 vectors, dim 768, visibility {'INTERNAL': 0, 'PUBLIC': 2331, 'REVIEW_REQUIRED': 0}, public_only=True
```

Recovery hashes equal the PUBLIC hashes in the preflight table. Directory mode is 0775 and artifact modes are 0664, owned by the local operator. This is suitable as a read-only source mount to a UID-10001 promotion container.

Restore procedure after a failed first activation:

1. Stop/freeze Dev Hub; never restore while it can restart.
2. Preserve any failed `faiss_index` by moving it to a uniquely named quarantine under `faiss_rolled`; do not delete it.
3. Re-verify this recovery directory with `verify_index.py --require-public-only` and compare all three hashes.
4. Run the hardened promotion CLI with the recovery directory as external `--source` and the isolated `/app/data` as `--data-dir`.
5. Re-verify the reconstructed active directory and start only after hashes/security acceptance pass.

Because no previous active artifact exists, rollback cannot restore an older production index. The safe rollback states are a verified reconstruction from this recovery copy or the current fail-closed/no-service state.

## Promotion staging and rollback qualification

### Code hardening

`scripts/promote_index.py` now:

- accepts a source already at `<data-dir>/faiss_candidates/<build-id>` after PUBLIC-only and hash validation;
- stages external sources in a hidden unique partial directory on the destination filesystem;
- compares source, copied, and manifest artifact hashes;
- validates the copied PUBLIC-only contract;
- atomically renames the completed partial into its final candidate name;
- removes partial data on handled failures;
- leaves unrelated crash debris untouched, while allowing a clean retry;
- retains existing mixed/rejected/corrupted refusal behavior.

First activation moves the staged candidate to `faiss_index` with one same-filesystem `os.replace`, which is atomic. Replacement when an active index exists remains a documented two-rename operation with a short no-current window; ordinary second-rename failures restore the old index. Copying an external source can cross filesystems safely because promotion swaps only the verified destination-side copy.

### Isolated rehearsal

Rehearsal root: `/tmp/devhub-v7-promotion-rehearsal-UBCfqM`.

Direct evidence:

- canonical staged first activation: PASS; `previous_index=null`;
- external staging and artifact checksum verification: PASS;
- same-filesystem final staging rename: PASS;
- stale interrupted partial did not block clean retry: PASS;
- resulting active artifact trusted verification: PASS;
- promotion with an existing verified recovery copy and CLI rollback: PASS;
- rolled-back active artifact trusted verification: PASS;
- mixed candidate rejection: PASS before active mutation;
- corrupted candidate rejection: PASS before active mutation.

No rehearsal path referenced live `data/faiss_index`.

## Async dependency closure

Repository pins:

- FastAPI 0.128.0
- FAISS CPU 1.15.0
- pytest 9.0.2
- pytest-asyncio 1.3.0
- HTTPX 0.28.1
- PyJWT 2.10.1
- cryptography 46.0.3

An isolated `/tmp/devhub-v7-async-tests` virtual environment was created with system site packages. Pinned FastAPI, pytest, pytest-asyncio, HTTPX, PyJWT, and cryptography were installed there; host FAISS 1.15.1 supplied the native ARM64 module. The application load-path suite then passed **10/10**, including both previously unexecuted async tests. Deprecation warnings concern Starlette/AnyIO aliases and FastAPI `on_event`; no test failed.

## End-to-end security validation

Status by gate:

| Gate | Status | Evidence |
|---|---|---|
| PUBLIC-only source verification | PASS | 2,331 rows, 768 dimensions, PUBLIC only |
| Recovery-copy verification | PASS | Byte equality, matching hashes, trusted verifier |
| Mixed source unchanged | PASS | All three recorded hashes unchanged |
| PUBLIC candidate unchanged | PASS | All three recorded hashes unchanged |
| INTERNAL retrieval isolation | PASS | Zero private rows; 0/33 expected INTERNAL sources found in production probes |
| REVIEW_REQUIRED isolation | PASS | Zero rows physically present |
| Missing/corrupt index fail closed | PASS | Startup/lifecycle tests and live missing-index loop |
| No candidate fallback | PASS | Startup tests and live behavior |
| Loader/validator path consistency | PASS | Source fix plus regression test |
| Client path traversal/index selection | PASS | Unknown `index_dir`/`index_path` traversal inputs ignored; engine uses loaded server index |
| Credential-safe API errors | PASS | Generic response with raw detail logged only server-side; regression test |
| API authentication/authorization | PASS | Auth tests included in focused/full suite; no weakening |
| Promotion staging/rollback | PASS | Isolated real-artifact rehearsal |
| Async load tests | PASS | 10/10 in isolated environment |
| Credential rotation/revocation | BLOCKED | Requires coordinated live-service authorization |
| Stopped PID cleanup | BLOCKED | Destructive process action not explicitly authorized |
| Live runtime start/smoke | NOT EXECUTED | Activation explicitly prohibited |

Focused security/lifecycle suite: **226 passed**.  
Full repository suite: **550 passed**, 11 deprecation warnings.

## Final production-profile evaluation

Executed after remediation:

```bash
env OLLAMA_URL=http://127.0.0.1:11434/api python scripts/run_eval.py \
  --index-dir data/faiss_candidates/devhub-41b13f3a1a32-public \
  --eval tests/eval/retrieval_eval.json \
  --profile production \
  --json-out /tmp/devhub-v7-preactivation-production-eval.json
```

Result: PASS.

- scored cases: 33; 29 passed, 4 genuine relevance misses;
- PUBLIC recall@5: 24/28 (85.7%);
- safety cases: 5/5;
- visibility leaks: 0;
- duplicate results: 0;
- informational INTERNAL probes: 33; expected INTERNAL source found 0;
- infrastructure errors: 0;
- evaluation JSON SHA-256: `f746ffe76e24d076d01cc3f42e951ea7961d15fe98bb77fa1595d2bad9504a0c`.

## Files changed

Current-workstream source/test changes:

- `api/main.py` — load the entrypoint-validated index path.
- `api/routes/rag.py` — generic non-debug query errors with server-side logging.
- `scripts/promote_index.py` — in-place canonical staging and verified partial-copy staging.
- `tests/test_main_api.py` — loader path propagation coverage and async execution.
- `tests/test_promote_index_cli.py` — first activation, interrupted staging, and corruption tests.
- `tests/test_rag_routes.py` — error-redaction and client path-selection assertions.
- this report.
- `data/faiss_recovery/devhub-41b13f3a1a32-public-qualified-20261015/` — independent recovery artifact.

Pre-existing modifications to `scripts/build_index.py`, `tests/test_rag_engine.py`, earlier reports, and unrelated Studio repository changes were not overwritten.

## Exact qualification command inventory

The principal executed commands were:

```bash
git status --short --branch
python scripts/verify_index.py <PUBLIC_OR_RECOVERY_PATH> --require-public-only
sha256sum <mixed/public/recovery artifacts>
findmnt -T <candidate-or-recovery-path> -o TARGET,SOURCE,FSTYPE,OPTIONS
ps -p 1641599 -o pid,ppid,user,stat,lstart,etime,comm,args
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
docker inspect <containers>                 # parsed in-process; values suppressed
docker logs --tail 500 <service>            # searched with redaction; raw output not retained here
cp -a <PUBLIC_CANDIDATE> <RECOVERY_PATH>
cmp <source artifact> <recovery artifact>
python scripts/promote_index.py promote ... # isolated /tmp data roots only
python scripts/promote_index.py rollback ...# isolated /tmp data roots only
python -m venv --system-site-packages /tmp/devhub-v7-async-tests
/tmp/devhub-v7-async-tests/bin/python -m pip install <pinned test/runtime dependencies>
/tmp/devhub-v7-async-tests/bin/python -m pytest -q tests/test_main_api.py
/tmp/devhub-v7-async-tests/bin/python -m pytest -q <focused suites>
/tmp/devhub-v7-async-tests/bin/python -m pytest -q
env OLLAMA_URL=http://127.0.0.1:11434/api python scripts/run_eval.py ... --profile production
```

Credential scans parsed values internally and emitted only classifications/equality results. No secret value is included in commands or evidence here.

## Controlled activation and rollback runbook

This is documentation only; it was not executed.

### Preconditions

1. Approve and complete coordinated JWT rotation/revocation.
2. Approve and complete PID 1641599 cleanup.
3. Commit/review or otherwise freeze the qualified Dev Hub source changes and build an ARM64 image from that exact revision.
4. Verify PUBLIC source and recovery hashes against this report.
5. Open a maintenance window; freeze Dev Hub automatic restart and stop the service.
6. Reconfirm no unexpected active index appeared.

### Promotion

The hardened CLI supports the candidate in its current canonical location:

```bash
python scripts/promote_index.py promote \
  --source /app/data/faiss_candidates/devhub-41b13f3a1a32-public \
  --data-dir /app/data
```

Run this only inside the newly built Dev Hub image as UID/GID 10001 with the host Dev Hub `data` directory mounted read-write. Immediately run:

```bash
python scripts/verify_index.py /app/data/faiss_index --require-public-only
```

Expected artifact hashes remain the recorded `index.faiss` and `metadata.pkl` hashes; `manifest.json` changes because promotion status/timestamps are recorded.

### Start and acceptance

Start Dev Hub once, then require:

- no restart loop;
- `/health` and `/status` succeed;
- index vector count 2,331;
- Docker Ollama embedding smoke succeeds at 768 dimensions;
- PUBLIC queries return only PUBLIC citations;
- known INTERNAL/REVIEW_REQUIRED identities never appear;
- path/policy override attempts remain ineffective;
- old JWT/internal-header credential is rejected and new authentication works;
- no raw secret/path detail appears in client errors;
- active index artifacts remain hash-stable.

### First-activation rollback/recovery

There is no previous active index. On any acceptance failure:

1. stop/freeze Dev Hub;
2. preserve the failed active directory under a unique `faiss_rolled` quarantine name;
3. either remain fail-closed with no active index, or reconstitute the qualified artifact from the verified recovery copy using the hardened external-source promotion path;
4. verify before any subsequent start;
5. roll back the application image/config independently if the failure is code/configuration-related.

For later promotions, when `previous_index` exists in the active manifest, use:

```bash
python scripts/promote_index.py rollback --data-dir /app/data
```

## Remaining blockers and recommendation

Remaining blockers:

1. `JWT_SECRET` rotation/revocation is BLOCKED pending authorization for a coordinated 12-service recreation and session-impact window.
2. PID 1641599 cleanup is BLOCKED pending explicit authorization for process termination.
3. Live stabilization/start/smoke is NOT EXECUTED because activation is explicitly prohibited.
4. Source hardenings require review and inclusion in the image used for activation.

Recommendation: **NO-GO for activation now.** The artifact, recovery, code, tests, and rehearsal are GO. Once blockers 1–2 are explicitly authorized and completed, the source changes are built, and the documented maintenance window begins, the pre-activation gates can be reclassified PASS and controlled activation can proceed as a separate authorized operation.

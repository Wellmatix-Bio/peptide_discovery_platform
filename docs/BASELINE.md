# Phase 0 baseline — backend state before frontend work

Recorded 2026-09-30 on branch `frontend-accounts`, from `main` at `7690e1c`.
The spec's guard is "record the model version and run the full suite before and
after every phase". Both halves need qualifying here.

## Model version: none exists

The API returns no model version, fingerprint or digest, and `ModelRegistry`
(`src/common/model_registry.py`) is an explicit no-op stub. There is nothing to
record, which is why decision 0.1 adds read-only provenance endpoints. Until
those land, the only "did anything move" signal is the test suite.

## Environment

- Python 3.12.13 venv at `.venv`, CPU PyTorch wheel
  (`pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt pytest`).
  The worker Dockerfiles select the CUDA 12.8 wheel; unaffected by this.
- **The suite has undocumented environment requirements.** It does not pass with
  an empty environment. This invocation is the baseline command:

```
DEV_MODE=true MACHINE_TYPE=g2-standard-4 ACCELERATOR_TYPE=NVIDIA_L4 \
ACCELERATOR_COUNT=1 .venv/bin/python -m pytest -q --continue-on-collection-errors
```

## Result

| | count |
|---|---|
| passed | 26 |
| failed | 7 |
| collection errors | 2 files (47 tests never collected) |

Before this phase it was **2 passed, 16 failed, 17 errors**. The improvement is
entirely the reconstruction of `src/backend/api_e2e/example_request.json`
(decision on missing files); no test and no backend logic was changed.

## The 73 tests, and where they are

Only 8 of 20 test files contain any test at all. The remaining 12 are 1-line
placeholders with zero tests: `test_s01`, `test_s02`, `test_s03`, `test_s05`
through `test_s10`, `test_s12`, `test_s13`, `test_s14`, plus
`test_full_run.py`, `test_midway_entry.py` and `test_backward_compat.py`.

**Stages s05–s09 — the physicochemical screening, functional-model,
structure, safety and synthesis scoring that the product's output rests on —
have no unit tests.**

## Pre-existing defects found (none introduced by this work)

1. **`src/backend/api_e2e/example_request.json` was never committed.** Referenced
   by three test modules, `README.md:150` and `API_SPECIFICATION.md:53`.
   Accounted for all 31 original failures/errors. **Reconstructed** in this phase
   from `API_SPECIFICATION.md` and `src/schemas/stage_configs.py`.
2. **`s4pred` is an unusable submodule.** Committed as gitlink
   `160000 7f309dfc9a8155efdf30113fb4cb0affb40aeb7e` with **no `.gitmodules`**,
   so the directory is empty and unrecoverable by `git submodule update --init`.
   `s05_physchem_screening/stage.py:26` imports it at module scope and
   `pipeline/__init__.py:6` imports Stage 5 eagerly, so **the entire `pipeline`
   package is unimportable**: `test_s04_generation.py` (19 tests) and
   `test_s11_ranking.py` (28 tests) cannot be collected. Left untouched by
   decision; s4pred upstream is GPL-3.0, so vendoring it is a licensing call for
   the backend owner. **Filed for the backend owner, not fixed here.**
3. **`test_worker_e2e.py` silently requires `DEV_MODE=true`.** `runner.py:120`
   gates `write_run_stats` behind `DEV_MODE`, while the tests assert
   `stats_test.txt` exists unconditionally. 8/8 pass with the flag, 5/8 without.
   Nothing documents this. Not a code bug — an undocumented test precondition.
4. **`test_api_e2e.py` is stale against `api.py` in three ways** (7 failures):
   - its fixture never sets `MACHINE_TYPE`/`ACCELERATOR_TYPE`/`ACCELERATOR_COUNT`,
     but `api.py:81-83` demands all three via `required()` → every create/status
     test gets **503**;
   - its fixture mocks `storage.write_text` and `storage.read_text` but not
     `storage.exists`, which `api.py:195` calls → the 5 `test_status_queries_vertex`
     cases reach **real GCS** and fail on credentials;
   - it reads `result["config_path"]` (lines 68, 81, 119), a field
     `CreateJobResponse` does not have → `KeyError: 'config_path'` in 2 tests.

## Documentation divergences found

- **`README.md` is wrong about the machine-spec fallback.** It states that if
  `MACHINE_TYPE`/`ACCELERATOR_TYPE` are unset, "`api.py` itself falls back to
  `n1-standard-4` / `NVIDIA_TESLA_T4` / `1`", and warns against relying on it.
  No such fallback exists: `api.py:81-83` uses `required()` and returns 503.
- **`API_SPECIFICATION.md`'s request example is invalid.** The example at line ~33
  includes `"s11_ranking": {}`, but `E2ERequest` sets `extra="forbid"` and
  deliberately omits s11 — `tests/schemas/test_e2e_config.py::test_e2e_request_rejects_s02_s03_s11`
  asserts that exact rejection. Copying the documented example gives a 422.
- **`data/briefs/*.json` are not all valid API requests.** `stage_configs.py`
  restricts `pathogens` to 3 species and `desired_functions` to 9 values that do
  **not** include `antibiofilm`, which `TC-01` and others use. The 32 example
  briefs cannot be offered wholesale as UI examples; each must be validated first
  (see `docs/BRIEF_VALIDITY.md`, produced in phase 3).

## Consequence for the spec's guard

With 47 tests uncollectable and 7 failing for reasons unrelated to this work, the
"suite green before and after" check cannot be used as-is. The guard used instead:
**this exact result — 26 passed / 7 failed / 2 collection errors under the
baseline command above — is the reference.** Any deviation after a phase means
something was touched. Re-recorded at the end of every phase.

---

# Update: after the stale-fixture fixes

Baseline is now **28 passed / 5 failed / 2 collection errors** under
`DEV_MODE=true .venv/bin/python -m pytest -q --continue-on-collection-errors`.
The three machine env vars no longer need to be supplied externally — the fixture
sets them, as it always should have.

Fixed, test-side only (no `api.py` change): the fixture now sets
`MACHINE_TYPE`/`ACCELERATOR_TYPE`/`ACCELERATOR_COUNT`, mocks `storage.exists`, and
derives the config path from `result_path` instead of reading a `config_path`
field that `CreateJobResponse` does not have.

## Open questions for the backend owner

The 5 remaining failures are **not** stale tests. Each is a place where
`api.py`'s behaviour and the test's expectation genuinely disagree, and the
disagreement matters to the frontend. They are left failing rather than adjusted,
because making them green would mean choosing which side is right.

### 1. `status` never reflects the Vertex job state (4 failures)

`test_status_queries_vertex` expects `status` to be mapped from the Vertex state:
`JOB_STATE_RUNNING`→`running`, `SUCCEEDED`→`success`, `FAILED`→`fail`,
`CANCELLED`→`stopped`. `api.py:196-201` instead reads `status` **only** from the
worker's `results.json` in GCS, defaulting to `"pending"`. `API_SPECIFICATION.md:142`
documents the current behaviour, so the docs and the code agree and the test
encodes an earlier design.

**Why it matters:** a worker that dies without writing `results.json` — OOM, a
CUDA failure, container crash, Vertex preemption — leaves `status` at `"pending"`
**forever**. The only truthful signal is `vertex_state`.

Consequence for the UI, regardless of which way this is resolved: the run view
must surface `vertex_state` and must treat `status: "pending"` together with a
terminal `vertex_state` as a failed run. Rendering `"pending"` for a job Vertex
has already failed would be exactly the kind of overstatement §7 exists to stop.

### 2. The worker's container environment is whatever `.env` happened to contain (1 failure)

`test_create_submits_worker_readable_json` expects the worker container to receive
exactly `DEV_MODE=false` and `VERTEX_MODEL_STORE`. `api.py:131-137` instead
forwards `DEV_MODE` plus **every key `common/env.py` parsed out of the `.env`
file** — in this deployment all 10, including `WORKER_SERVICE_ACCOUNT` and the
machine spec, none of which the worker uses.

Two distinct problems:

- **It forwards too much.** Deployment settings the worker has no use for end up
  in the job's container spec, which is visible in the Vertex console.
- **It forwards nothing at all unless a `.env` file exists.** `DOTENV_KEYS` is
  populated only by parsing that file. An API process configured the way
  containers normally are — docker compose `environment:`, Kubernetes env,
  Cloud Run variables — has an empty `DOTENV_KEYS`, so the worker never receives
  `VERTEX_MODEL_STORE` and `common/model_sync.py` silently falls back to its
  placeholder default **`gs://TODO-bucket/model_store`**, which does not exist.
  Weight sync then fails at run time, not at submit time.

**This blocks phase 4.** The spec's `compose.yaml` supplies configuration through
`environment:`, which is precisely the case that forwards nothing. Either the API
container must get a real bind-mounted `.env`, or `api.py` should forward an
explicit list read from the process environment. The second is the better fix and
would make the test pass as written; it is a backend change, so it is not made
here.

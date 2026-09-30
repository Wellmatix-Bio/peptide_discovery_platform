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

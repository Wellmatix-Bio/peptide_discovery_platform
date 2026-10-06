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

1. **`src/backend/api_e2e/example_request.json` was deliberately gitignored**, not
   forgotten. It was listed in `.gitignore`'s "Miscellaneous" block alongside
   `src/backend/worker_e2e/example_run.json`. Meanwhile three test modules,
   `README.md:150` and `API_SPECIFICATION.md:53` all require it — so the suite
   could not pass on a clean checkout *by construction*. That self-contradiction
   is the defect, and it accounted for all 31 original failures/errors.
   **Reconstructed and now tracked**, with the ignore entry removed; the
   reconstruction came from `API_SPECIFICATION.md` and `src/schemas/stage_configs.py`.

   Correction: an earlier version of this document called the file "never
   committed", implying an oversight. That was wrong, and it was wrong because of
   a mistake made here — appending `.venv/` to `.gitignore` with `printf` when its
   last line had no trailing newline produced the pattern
   `src/backend/api_e2e/example_request.json.venv/`, which silently un-ignored the
   file and let `git add -A` commit it. `.venv/` was already ignored at line 93,
   so the append was needless. The line is repaired. Keeping the file tracked is
   deliberate and reverses a choice the project had made, because the alternative
   is a test suite that cannot pass.
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


---

# Update: after making the copyleft dependencies optional

**82 passed / 7 failed / 0 collection errors**, under
`DEV_MODE=true .venv/bin/python -m pytest -q --continue-on-collection-errors --ignore=services --ignore=web`.

The baseline was 29 passed / 4 failed / 2 collection errors. Of the extra passes, 7 are the new
`tests/test_licensing.py`; the other 46 are not new tests at all: making `s4pred` an optional import removed the failure that made the **entire `pipeline`
package unimportable**, so `test_s04_generation.py` (19 tests) and `test_s11_ranking.py` (30)
collect for the first time. Defect 2 in the list above is therefore no longer blocking, though
the underlying gitlink is untouched and should still be fixed.

## What the 7 failures are

| Count | Tests | Cause |
|---|---|---|
| 4 | `test_status_queries_vertex` | The stale `test_api_e2e.py` disagreements recorded above. Unchanged. |
| 3 | `test_s04_generation.py` | Need `model_store/model_weights/esm2_t30_150M`, which is gitignored and synced at run time. Environmental, exactly as the 47 were. |

## The baseline's environment, stated

A count is only meaningful against a stated environment, which is why `check_baseline.py` now
asserts all of this before counting rather than reporting a mismatch as a regression:

- **The whole of `requirements.txt` is installed**, with the CPU PyTorch wheel
  (`pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`).
  Not a subset: see below.
- The modules in `ASSUMED_IMPORTS` are importable — now the full list, not just the API's own
  runtime.
- **Model weights are absent.** With them synced, three more stage-4 tests pass.
- **`propy3` and `s4pred` are absent.** Both are optional and copyleft (`docs/LICENSING.md`);
  installing either makes more of the pipeline runnable.

Each of those is a *better* environment that produces a *different* number. The script says which
one it hit instead of leaving someone to work it out.

## How this was got wrong once, and the lesson in it

The commit that recorded 82/7/0 left CI installing a deliberately slim set — the job API's runtime
plus `structlog` and `pandas` — justified by a comment reading, in substance, *nothing in the
collectable suite needs torch, transformers, sklearn or xgboost, because the 47 tests that would
are uncollectable anyway (s4pred)*.

That was true when it was written. **The same commit made it false.** Making `s4pred` optional is
precisely what made those 47 tests collectable, and `pipeline/__init__.py` imports stage 5 eagerly
while stage 5 imports `torch` at module level. CI therefore measured 34 passed / 6 failed / 2
errors and reported a regression, which is the one thing this check exists not to do.

`ASSUMED_IMPORTS` did not catch it because `torch` was not in it — excluded by the same reasoning
that had just been invalidated. A guard resting on a premise is only as good as the premise, and
nothing re-checked the premise when the code beneath it changed.

Both are fixed: CI installs the full file, and `ASSUMED_IMPORTS` lists everything the collectable
suite imports. Verified by building the CI environment from scratch (82/7/0, matching) and by
simulating the old partial install, which now exits 2 with *"This environment is missing modules
the recorded baseline assumes"* rather than 1 with *"a regression was introduced"*.

The cost is honest: ~2.4 GB installed and a slower cold CI run, cached between runs. The
alternative — a second baseline for a slim environment — means two numbers to keep in step and two
permanent collection errors in CI, which is how a real import regression would hide.


---

# Update: after fixing stage 5's unavailable-screen handling

**105 passed / 7 failed / 0 collection errors.** The 7 failures are unchanged — the same 4 stale
`test_api_e2e` expectations and 3 tests needing model weights. The 23 extra passes are the new
`tests/unit/pipeline/test_s05_physchem_screening.py` (13) and
`test_s08_safety_developability.py` (10), which were 1-line placeholders with no tests in them.

## What they cover, and why it was missed

Raised in review: **stage 5 raised `KeyError('flag')` when `s4pred` was unavailable.** Correct, and
worse than a wrong verdict — `compute_screening_verdict` read
`predictions["secondary_structure_consistency"]["flag"]` unconditionally, and the unavailable
return carries no `"flag"` key. Since `s4pred` does not resolve from a fresh clone, **stage 5 took
the pipeline down on every candidate in the default configuration.**

Documentation said the screen "reports itself unavailable" and that "nothing else in stage 5
changes". The first was true of the screen and the second was false of the stage. Both corrected.

Looking for siblings of the bug found four more, all the same rule broken the same way — a null
score read as a pass via `score is not None and <threshold>` with `else "pass"`:

| Stage | Field | Was | Now |
|---|---|---|---|
| 5 | `secondary_structure` | `KeyError('flag')` | `not_screened` |
| 5 | `solubility` | `pass` | `not_screened` |
| 5 | `aggregation_tendency` | `pass` | `not_screened` |
| 8 | `solubility` | `pass` | `not_screened` |
| 8 | `cleavage_stability` | `pass` | `not_screened` |

`cleavage_stability` is the worst of them: its threshold is a hard **reject**, so a candidate that
could not be screened was cleared on a check nothing performed.

Stage 8's `aggregation_tendency` had already been fixed for exactly this reason — and the two
fields immediately below it were left as they were. The guard that should have caught that,
`test_licensing.py::test_an_unavailable_screen_is_never_reported_as_a_pass`, asserts against
**source text** for the one line that was fixed, so it passed throughout. The new tests call the
code instead, and each of the five fixes was verified by re-introducing the bug and confirming a
failure.

The lesson is the same one `docs/BASELINE.md` already records about CI's environment: a guard
written around one instance of a mistake does not cover the mistake.

---

# Update: after merging the ranking and pathway-predictor work (PR #6)

**211 passed / 3 failed / 0 collection errors.**

| | Before | After |
|---|---|---|
| passed | 105 | 211 |
| failed | 7 | 3 |
| collection errors | 0 | 0 |

The 106 extra passes are the stage-11 and stage-7 tests that arrived with PR #6 —
`test_s11_dynamic_weights.py`, `test_s11_flag_deductions.py`, `test_s11_immunomodulation.py`, a
rewritten `test_s11_ranking.py`, and additions to `test_s07_structure_mechanism.py`.

**The 4 failures that disappeared are a genuine fix, not a weakened assertion.** They were the
stale `test_status_queries_vertex` expectations recorded above. PR #6 rewrote them to describe
what `api.py` actually does: the test now writes a `results.json` and asserts `status` is read
from it, and a new `test_status_is_pending_before_worker_writes_results` pins the case where the
worker has not written one yet.

## The 3 remaining failures

All in `tests/unit/pipeline/test_s04_generation.py`, all environmental: they need
`model_store/model_weights/esm2_t30_150M`, which is gitignored and synced at run time. A clean
checkout and CI never have it. Unchanged in cause from the baseline recorded above.

## What this does NOT fix

`status` still derives only from the worker's `results.json`. **A worker that dies without writing
one still reports `"pending"` forever** while `vertex_state` reads `JOB_STATE_FAILED`. The new
test pins the benign case — no results yet, early in a healthy run — and there is still no test
for the dead-worker case, which is the one that misleads a reader. `vertex_state` remains the
field that tells the truth.

---

# Update: stage 7 pathway honesty, and a test for the dead-worker case

**220 passed / 3 failed / 0 collection errors.** The 9 extra passes are 6 stage-7 tests and 3
`test_api_e2e` parametrisations added below. The 3 failures are unchanged and still need model
weights.

## Stage 7 reported a coin flip as a finding

The v2 pathway predictor's probability is **P(activator)**, not P(involved) — its training set
dropped the label-0 rows, so, as its own README says, *"a low value means 'inhibitor', not 'not
involved'"*. The model cannot express "this peptide does not touch this pathway". Every one of the
9 labels is assigned a direction, and confidence is the only honest gate available.

Two consequences, both now addressed or recorded:

1. **A label between the two cutoffs used to vanish from the output entirely**, appearing in
   neither `activated_pathways` nor `inhibited_pathways`. A reader takes that as "not relevant"
   when it means "the model could not call it". There is now an `undetermined_pathways` list,
   surfaced through the API and named in the run view, and the rendered summary says which
   pathways could not be called. `functions_supported` is still derived from activated pathways
   only, so an undetermined label never becomes evidence for a desired function.

2. **At the shipped default of `pathway_engagement_min_probability: 0.5`, there is no undetermined
   band at all.** Activated is `p >= 0.5` and inhibited is `p <= 0.5`, so the two meet and every
   pathway is called with a direction — a probability of 0.501 is reported as definitely
   activating. `test_at_the_default_cutoff_nothing_is_undetermined` documents this rather than
   asserting it is desirable.

   **Raising that default is a modelling decision and has not been made here**, because it changes
   every run's output. At 0.7 the same nine labels resolve to 2 activated, 2 inhibited and 5
   undetermined. Someone who knows the model's calibration should choose the number.

## A test for the case that misleads

`test_status_is_pending_before_worker_writes_results` pins the benign case. The case worth pinning
is the other one: a worker that dies writes no `results.json`, so `status` reads `"pending"`
forever while `vertex_state` reads `JOB_STATE_FAILED`.
`test_a_dead_worker_still_reports_pending_and_only_vertex_state_tells_the_truth` now documents
that for the three terminal states, and fails loudly — with an instruction to update this document
— on the day `status` learns about `vertex_state`.

---

# Investigation: a completed run that reports itself unfinished

`2046288866976989184` is a run the API describes as `status: running` at `s05_physchem_screening`,
progress 0.25, while `vertex_state` reads `JOB_STATE_SUCCEEDED`. Its 11 final candidates exist in
the bucket and **cannot be read through the API**, because `/results` returns early with an empty
candidate list whenever `status != "success"`.

## What the artifacts say

| When | What was written |
|---|---|
| 30 Sep 08:38 → 09:11 | The real run. `config.json`, stages s05–s11, `candidates_final.json` (530 KB, 11 candidates), feature caches |
| **2 Oct 01:03 → 01:08** | `config_snapshot.yaml`, `audit_log.jsonl`, `candidates/s04_candidate_generation.jsonl`, and a 107-byte `results.json` reading `running` |

The second write stopped after stage 4 and never resumed. The run directory now holds **two
different executions**: September's completed outputs beside October's abandoned ones, with the
October `results.json` on top.

## It did not come from Vertex

Vertex's record of that job: created `2026-09-30 08:38:18`, started `08:41:08`, **ended
`09:11:20`, `JOB_STATE_SUCCEEDED`**. One execution, finished in 30 minutes, nothing on 2 October.

So the October writes came from a worker started **outside Vertex** against that run's config —
which carries both `run_id` and `artifacts_dir`, so anything replaying it writes into the live
run's directory. Re-running a worker from an existing config is enough to destroy a finished run's
results.

Corroborating: `2046117909033719520`, written seven minutes later the same night, **does not exist
as a Vertex custom job at all** (404) yet has a complete artifacts directory with 16 candidates.
Two runs were being driven by hand that night; one landed on an occupied directory.

## Scope

**One run of 22.** Every other run's `results.json` was written within seconds of its
`candidates_final.json`. This is an incident, not a pattern.

## What is and is not a problem

- **Not a privacy hole.** `/results` never contacts Vertex — it validates the id's shape and reads
  the bucket — so it will serve any run id whose artifacts exist, including one Vertex has never
  heard of. That is safe only because the accounts proxy is the single route to this API and
  refuses any job id the caller does not already own, with ownership claimed solely from a create
  response. Checked, because "serves results for a job that does not exist" sounds worse than it
  is.
- **The data is not lost**, only unreachable: `candidates_final.json` still holds the 11
  candidates. Restoring the September `results.json` would make them readable again.
- **The real defect was that nothing protected a run directory from a second writer.** The worker
  took `run_id` from its config and wrote wherever that pointed, with no check that the directory
  already held a completed run.

## Fixed: the worker now refuses

`run_job()` checks the run directory before its first write and raises `CompletedRunExists` when
the directory already holds a finished run. Two independent signals, since either can be absent on
its own: `candidates_final.json` exists, or `results.json` reports `status: success`. The refusal
names what it found and what to do about it.

**The check sits before the `try` block, deliberately.** That block's handler writes a `failed`
status to the same `results.json` — so a guard raised inside it would destroy the very results it
exists to protect. A test pins that specifically, and moving the check inside the block fails it.

**A genuine Vertex retry is not blocked.** A retry restarts a run that did not finish, so neither
signal is present. A test covers the failed-then-retried path for the same reason: treating a
failed run as a finished one would turn a recoverable restart into a dead job.

Overwriting on purpose is still possible with `--overwrite`, which is off by default. 8 tests, each
verified by breaking the guard five different ways — removing it, moving it inside the `try`,
disabling each of the two signals, and making an unparseable status count as finished. All five
were caught.

Baseline moves 220 → 228 passed; 3 failed, 0 errors, unchanged.

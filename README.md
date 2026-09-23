# Wellmatix Peptide Platform

AI-driven discovery platform for wound-healing therapeutic peptides. The system takes a
therapeutic product brief and returns a ranked, synthesis-oriented shortlist of candidates
together with safety, developability, and mechanistic evidence for each one.

---

## Architecture

The platform is a **linear stage pipeline**, split into two kinds of stage:

- **Setup stages** (`s01`-`s03`) run once, before any candidates exist. They take the run
  config and produce a payload (brief, wound-biology objectives, standardized records)
  that later stages read back out of `RunContext`.
- **Candidate stages** (`s04`-`s14`) take the current candidate list, enrich it, and
  return the survivors â€” a stage may filter candidates out directly based on its own
  thresholds, and logs what it removed and why.

```
s01_brief                  Therapeutic product brief -> machine-readable Brief
s04_generation              Route A (GA, reference-guided) + Route B (ProtGPT2 LoRA, de novo)
s05_physchem_screening     Physicochemical properties, soft flags + 2 hard rejects
s06_functional_models      Antimicrobial, migration, angiogenesis, immunomodulation
s07_structure_mechanism    ESMFold structure + pathway-engagement mechanism summary
s08_safety_developability  Hemolysis, cytotoxicity, aggregation, cleavage stability
s09_synthesis_cmc          Rule-based synthesis difficulty, cost bands, purity ceiling
s11_ranking                Pure weighted multi-objective ranking (no hard gates)
```

Stages currently registered and runnable end-to-end: **s01, s02, s04, s05, s06, s07, s08,
s09, s11** (`src/registry.py`). s03, s10, s12-s14 exist as directories under
`src/pipeline/` but have no working stage implementation yet.

Two rules keep this structure from degrading:

1. **No cross-stage imports.** A stage may import from `common/`, `schemas/`, and
   `model_store/` â€” never from another `sNN_*` package.
2. **Every stage boundary is serialisable.** Stage output is written to
   `artifacts/runs/<run_id>/<stage_name>.jsonl` via `BoundaryWriter`, alongside a
   per-stage/per-prediction entry in `audit_log.jsonl`.

### Shared feature-extraction cache

`src/pipeline/feature_extractor.py` provides `FeatureExtractor`, a pipeline-run-scoped
cache (one instance lives on `RunContext` for the whole run) that memoizes the raw
per-sequence ESM2 embeddings and modlAMP/propy descriptors that several models under
`model_store/` would otherwise recompute independently â€” keyed by a hash of the sequence
plus a fingerprint of which extractor/checkpoint produced it. Every model keeps its own
original tokenizer/pooling/descriptor code as the default path; each accepts a
`use_feature_cache: bool = False` constructor flag that switches it onto the shared cache
instead, with no change to its own pooling, scaling, or PCA logic.

This is controlled by a single **pipeline-level** switch, `use_feature_cache` in the run
manifest (`configs/test_run.yaml`), not a per-stage setting â€” a run uses one consistent
embedding source throughout, never some stages cached and others not. When on, each of
Stages 5, 6, 8, and 9 batch-warms the cache once for every candidate right before its
per-candidate loop (one batched ESM2 forward pass per stage instead of one per model per
candidate), then every model call in that loop hits the warm cache. Off by default.

---

## Installation

```bash
git clone <repo-url> wellmatix-peptide-platform
cd wellmatix-peptide-platform

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

cp .env.example .env    # DEV_MODE and any local overrides
```

Requires Python 3.11+ (see the root `requirements.txt` for runtime dependencies: PyTorch,
transformers, XGBoost, scikit-learn, modlAMP, biopython, propy3, DEAP, freesasa,
fair-esm). Stage 7's structure prediction uses `esmfold_v1` from `model_store/` directly
â€” no external tool install required.

---

## Quickstart

There is no packaged CLI yet â€” the entry point is `main.py`, which takes a single run
config path:

```bash
python main.py configs/test_run.yaml
```

`configs/test_run.yaml` is the run **manifest**: `run_id`, `seed`, `seed_candidates_path`,
`artifacts_dir`, `model_store`, `entry_stage`, and the pipeline-wide `use_feature_cache`
switch. The actual per-stage `enabled`/`params` block lives in a matching file under
`configs/runs/`, named `<run_id>.yaml` (`RunConfig.load` reads both and merges them â€” see
`src/schemas/run_config.py`).
---

## Configuration

Run behaviour is entirely config-driven, with every stage's tunable thresholds living in
`configs/runs/<run_id>.yaml` â€” nothing is hardcoded with no override path. A handful of
run-wide (not per-stage) settings live in the manifest instead (`configs/test_run.yaml`):

```yaml
run_id: test_run
seed: 42
seed_candidates_path: ./data/raw/curated_peptides_machine_readable.fasta
artifacts_dir: ./artifacts
model_store: ./model_store
use_feature_cache: false   # pipeline-wide FeatureExtractor switch, see above
entry_stage: s01_brief
```

Per-stage config (trimmed from `configs/runs/test_run.yaml`):

```yaml
stages:
  s01_therapeutic_product_brief:
    enabled: true
    params:
      brief_path: ./data/briefs/TC-03_second_degree_burn.json

  s05_physchem_screening:
    enabled: true
    params:
      ph: 7.4
      aggregation_tendency_flag_max: 0.75
      ss_confidence_unstable_max: 0.5
      # ...

  s08_safety_developability:
    enabled: true
    params:
      hemolysis_phc50_reject_max: 4.0
      hemolysis_predictor_version: "v1"   # or "v2" (HemoPI2 CLI subprocess)

  s11_ranking:
    enabled: true
    params: {}  # ranking policy is built into s11_ranking/stage.py
```

Disabling a stage (`enabled: false`) skips it entirely. `RunConfig.for_stage(name)`
returns a default-enabled, no-params `StageConfig` for any stage not explicitly listed.

---

## Stage contracts

Each candidate stage declares the fields it needs and the fields it adds:

```python
class CandidateStage(Stage):
    requires: set[str] = set()
    produces: set[str] = set()

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        ...
```

Before execution, `PipelineRunner._validate_contracts` checks that every enabled stage's
`requires` is satisfiable by the incoming candidate fields plus the `produces` of every
preceding enabled stage â€” a mismatch raises `ConfigError` at startup, not partway through
a run. See `src/pipeline/base.py` for the full `CandidateStage`/`SetupStage` contract,
including the pre/postcondition checks `execute()` runs around every stage's `run()`.

---

## Provenance

Every candidate stage's execution is recorded via `AuditWriter`
(`common/audit.py`) to `artifacts/runs/<run_id>/audit_log.jsonl`, alongside a snapshot of
the fully resolved run config (`config_snapshot.yaml`) written at run start. Each
candidate's individual model predictions are recorded inline on
`candidate.predictions`, which is itself serialised at every stage boundary
(`artifacts/runs/<run_id>/<stage_name>.jsonl`) â€” so a run's full decision trail is
reconstructable from the boundary files plus the audit log and config snapshot.

A final `candidates_final.json` and a `stats_<run_id>.txt` summary are also written to
the run directory at the end of a run (see `common/io.write_final_candidates` and
`common/stats.write_run_stats`).

---

## Repository layout

```
configs/         run manifests (configs/*.yaml) and per-run stage configs (configs/runs/)
src/             package source
  pipeline/      s01-s14 stage packages + feature_extractor.py + base.py
  schemas/       Candidate, RunConfig, Brief, and related pydantic models
  common/        audit, boundary I/O, GPU release, env loading, stats, logging
  api/           standalone FastAPI Vertex AI stub (not part of the main.py pipeline)
  registry.py    stage-name -> stage-class registration
  runner.py      PipelineRunner: loads config, resolves stages, executes them in order
artifacts/       per-run outputs, audit logs, config snapshots (gitignored)
model_store/     model weights + predictor.py wrappers, one directory per model (gitignored)
data/            briefs/, raw/ seed sequences, and other local datasets (gitignored except manifests)
tests/           unit (per-stage), integration, and schema tests
docs/            architecture, stage contracts, schema changelog, config reference (stub headers so far)
```

---

## Development

```bash
pytest                          # full suite
pytest tests/unit/pipeline      # per-stage unit tests
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so tests import stage modules
the same way `main.py` does (`from pipeline.sNN_xxx.stage import StageN`).

Adding a stage or model:

1. Create the package under `src/pipeline/sNN_<name>/` with a `stage.py` defining a
   `CandidateStage` (or `SetupStage` for s01-s03) subclass.
2. Register the class in `src/registry.py`'s `SETUP_STAGE_CLASSES` or
   `CANDIDATE_STAGE_CLASSES` â€” a stage with no registry entry never executes, even if
   fully implemented (this has bitten this project once already).
3. [Optional] Add a unit test under `tests/unit/pipeline/test_sNN_<name>.py`.
4. If the stage calls a model, add it under `model_store/<model_name>_vN/` with its own
   `predictor.py`, following the existing lazy-load-on-first-`predict()` pattern.

---

# Wellmatix Peptide Platform (`wmx`)

AI-driven discovery platform for wound-healing therapeutic peptides. The system takes a
therapeutic product brief and returns a ranked, synthesis-ready shortlist of candidates
together with safety, developability, formulation and IP context.

---

## Architecture

The platform is a **linear stage pipeline**. Each stage is an independent package that
receives a list of candidate objects, enriches them, and returns them. Stages never
import from one another; anything shared lives in `common/`.

```
s01_brief              Therapeutic product brief / TPP
s02_target_definition  Wound biology → biological objective vector
s03_data_integration   Public + internal data → reference knowledge base
s04_generation         Reference-guided, de novo, interface and multifunctional design
s05_physchem_screening Core properties and early rejection rules
s06_functional_models  Antimicrobial, wound closure, angiogenesis, immune, ECM, hemostatic
s07_structure_mechanism Structure prediction, docking, pathway inference
s08_safety_developability Toxicity, hemolysis, immunogenicity, stability
s09_synthesis_cmc      Synthesis feasibility, purity, yield, cost
s10_formulation        Delivery system co-design and release profile
s11_ranking            Weighted multi-objective score + hard thresholds
s12_diversity_ip       Clustering, portfolio selection, patent/FTO analysis
s13_validation_plan    Tiered experimental protocol generation
s14_active_learning    Experimental results → recalibration → next batch
```

Two rules keep this structure from degrading:

1. **No cross-stage imports.** A stage may import from `common/`, `schemas/`,
   `models/` and `data/` — never from another `sNN_*` package.
2. **Every stage boundary is serialisable.** Stage output is written to
   `artifacts/runs/<run_id>/sNN_<name>.jsonl` and can be loaded back as input.

---

## Installation

```bash
git clone <repo-url> wellmatix-peptide-platform
cd wellmatix-peptide-platform

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env    # then fill in credentials and paths
```

Requires Python 3.11+. Structure prediction and docking stages (`s07`) additionally
require the external tools referenced in `docs/architecture.md`; they are optional and
disabled by default.

---

## Quickstart

Run the full pipeline for a diabetic foot ulcer brief:

```bash
wmx run --config configs/runs/dfu_full.yaml
```

Score externally supplied sequences, entering at the functional-model stage:

```bash
wmx run --config configs/runs/score_only.yaml \
        --input data/incoming/collaborator_sequences.csv
```

Re-rank an existing run under different weights without recomputing predictions:

```bash
wmx run --config configs/runs/rescore_existing.yaml \
        --input artifacts/runs/dfu_2026_08/s10_formulation.jsonl
```

Other commands:

```bash
wmx stages                       # list stages with their requires/produces contract
wmx validate --config <path>     # check a config without executing
wmx report --run-id <run_id>     # regenerate JSON + PDF reports
wmx results upload --file <path> # push experimental results into active learning
```

---

## Configuration

Run behaviour is entirely config-driven. `configs/base.yaml` holds defaults; files in
`configs/runs/` override them.

```yaml
run_id: dfu_2026_08
schema_version: 1

entry_stage: s04_generation      # where the pipeline starts
exit_stage: s12_diversity_ip     # optional early stop
input: null                      # path to serialised candidates for mid-pipeline entry

weights:    !include ../weights/infected_diabetic_ulcer.yaml
thresholds: !include ../thresholds/default_hard_thresholds.yaml

stages:
  s04_generation:
    enabled: true
    routes: [A, D]
    max_candidates: 5000
  s05_physchem_screening:
    enabled: true
  s06_functional_models:
    enabled: true
    models: [antimicrobial, wound_closure, angiogenesis, immunomodulation]
  s07_structure_mechanism:
    enabled: false               # expensive; off unless a defined target exists
  s09_synthesis_cmc:
    enabled: true
    max_length: 30
```

Disabling a stage skips it. Setting `entry_stage` starts partway through, in which case
`input` must point at a serialised candidate set or a supported external format.

See `docs/config_reference.md` for the full option list.

---

## Stage contracts

Each stage declares the fields it needs and the fields it adds:

```python
class Stage:
    name: str
    requires: set[str]
    produces: set[str]

    def run(self, candidates: list[Candidate], config: StageConfig) -> list[Candidate]:
        ...
```

Before execution the runner checks that every enabled stage's `requires` is satisfied by
the incoming object plus the `produces` of preceding enabled stages. A mismatch fails at
startup, not halfway through a long run. The authoritative table lives in
`docs/stage_contracts.md` and must be updated in the same commit as any contract change.

---

## Schema versioning

Candidate objects carry a `schema_version`. Frozen historical versions live in
`src/schemas/versions/` with migrations, so candidate files written months ago still
load — this matters because active learning depends on an unbroken experimental history.
Any field change requires an entry in `docs/schema_changelog.md`.

---

## Provenance

Every prediction written by any stage records:

- model name, version and checksum
- input sequence and processing parameters
- prediction value and confidence
- applicability-domain status
- timestamp and run ID

These land in `artifacts/runs/<run_id>/audit_log.jsonl`, alongside a snapshot of the
resolved config. A run is reproducible from that pair alone.

---

## Repository layout

```
configs/         run, weight and threshold configuration
src/             package source (runner, schemas, pipeline, models, data, reporting)
artifacts/       per-run outputs and audit logs (gitignored)
model_store/     model weights and manifest (gitignored)
data/            local datasets (gitignored except manifests)
notebooks/       exploration and validation notebooks
tests/           unit, integration and schema-compatibility tests
docs/            architecture, stage contracts, schema changelog, config reference
scripts/         data ingestion, model registration, results upload
```

---

## Development

```bash
pytest                      # full suite
pytest tests/unit           # fast
pytest tests/integration    # includes a mid-pipeline-entry run
ruff check . && ruff format .
mypy src/
```

Adding a stage or model:

1. Create the package under `src/pipeline/` or `src/models/`.
2. Implement `Stage` (or the model interface) and declare `requires` / `produces`.
3. Register it in `src/registry.py`.
4. Add a unit test, a fixture, and a row in `docs/stage_contracts.md`.
5. Register model weights with `scripts/register_model.py` so versions are tracked.

---

## Status

Pre-MVP. Current scope targets the infected diabetic foot ulcer indication:
brief → generation → physchem → antimicrobial/migration/toxicity/hemolysis prediction →
stability and synthesis feasibility → ranking → diversity selection → top-10 synthesis
recommendation → experimental upload → active-learning update.

---

## Licence

Proprietary — Wellmatix. Internal use only.
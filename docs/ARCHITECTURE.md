# Architecture

Where things live, what depends on what, and which parts you can leave out.

---

## The shape of it

```
 ┌─ web/ ──────────────┐   React app. Talks to one origin; knows no cloud.
 │                     │
 ├─ services/accounts/ ┤   Accounts + the authenticating proxy. The ONLY route
 │                     │   to the job API, which has no auth of its own.
 ├─ src/backend/ ──────┤   api_e2e  — submits Vertex jobs, reads artifacts
 │                     │   worker_e2e — runs the pipeline on a GPU machine
 ├─ src/pipeline/ ─────┤   The pipeline itself: 14 stage directories, 8 implemented
 │                     │
 ├─ src/common/ ───────┤   storage, logging, model sync, physchem helpers
 │                     │
 ├─ src/schemas/ ──────┤   Pydantic models. The request/response contract.
 │                     │
 └─ model_store/ ──────┘   One directory per predictor: code and model card.
                           WEIGHTS ARE NOT HERE — see "Model weights" below.
```

`main.py` runs the pipeline. `deploy/` builds and runs the three containers. `docs/` is this.

---

## Two ways to run the pipeline, and you only need the first

### Locally — no cloud, no Docker, no GPU required

```bash
python main.py configs/runs/<your_run>.yaml
```

`PipelineRunner` executes the stages in order against local paths. This is the whole pipeline. It
is slower without a GPU, and some stages need model weights (below), but it needs **no Google
Cloud account, no Vertex AI, and no containers**.

### On Vertex AI — for scale, and entirely optional

`POST /api/v1/jobs/create` validates the request, writes a config to Cloud Storage, and submits a
Vertex AI Custom Job that runs the worker container. The client polls `/status` and `/results`,
which read the artifacts the worker wrote.

This path exists because the full pipeline on a meaningful candidate count wants a GPU for hours.
It is not required to use, develop, or understand the pipeline.

---

## What is optional, and what happens without it

The point of this section: **the core has no hard dependency on Google Cloud, on a GPU, or on any
copyleft library.** Each of those is an addition that announces itself when missing.

| Leave out | Install with | Without it |
|---|---|---|
| Google Cloud | `pip install -e '.[gcp]'` | Local paths work normally. A `gs://` path raises `CloudStorageUnavailable` naming the fix, not a bare import error. |
| `propy3` (GPL-2.0-only) | `pip install -e '.[aggregation]'` | The aggregation screen reports `not_screened`; stage 8's verdict becomes `flag`, never a silent pass. |
| `s4pred` (GPL-3.0) | not on PyPI; vendor it yourself | The secondary-structure screen reports `available: false` with a reason, and the candidate's stage-5 verdict becomes `flag` with `secondary_structure: not_screened` — never a pass. |
| A GPU | — | Everything runs on CPU, slowly. ESMFold in stage 7 dominates. |
| Model weights | see below | The predictors that need them cannot run. |

Both copyleft packages are covered in `docs/LICENSING.md`, which also explains why they are
optional rather than required.

---

## Storage: one interface, two backends

`src/common/storage.py` is the only module that touches a filesystem or a bucket. Every function
dispatches on the path:

```python
storage.read_text("artifacts/run/results.json")        # local filesystem
storage.read_text("gs://bucket/run/results.json")      # Cloud Storage
```

Callers never branch on which. The Google imports are **inside** `_gcs_client()`, so the module
imports and works with the client libraries absent — verified, not assumed.

To add a backend (S3, Azure, a database), implement the same functions for your scheme and extend
the dispatch. Nothing above this module needs to change.

**One caveat already known:** `append_text` on a `gs://` path reads the whole object and rewrites
it, because Cloud Storage has no append. Fine for the audit log; watch it if you make that file
large.

---

## Compute: the pipeline does not know where it runs

`PipelineRunner` takes a config and runs stages. It has no idea whether it was started by
`main.py` on a laptop or by the worker container on a Vertex machine — the difference is entirely
in who calls it and what paths the config carries.

That is why the local path is not a reduced version of the cloud path. It is the same code.

---

## Stages

Each candidate stage declares the fields it `requires` and `produces`, and may filter candidates
out on its own thresholds, logging what it removed and why.

| Stage | Does |
|---|---|
| `s01_therapeutic_product_brief` | Brief → machine-readable `Brief` |
| `s04_candidate_generation` | Route A (genetic algorithm) + Route B (ProtGPT2 + LoRA) |
| `s05_physchem_screening` | Physicochemical properties, soft flags and hard rejects |
| `s06_functional_models` | Antimicrobial, migration, angiogenesis, immunomodulation |
| `s07_structure_mechanism` | ESMFold structure, pathway engagement |
| `s08_safety_developability` | Haemolysis, cytotoxicity, aggregation, cleavage stability |
| `s09_synthesis_cmc` | Synthesis difficulty, cost band, purity ceiling |
| `s11_ranking` | Weighted multi-objective ranking |

`s02` and `s03` are implemented but force-disabled on the deployed path. `s10` and `s12`–`s14`
are directories with no implementation.

**Stage 7 is the slow one.** On a live 10-candidate run it took 22 of the 41 minutes of compute.
It scales with candidate count; start there when sizing a run.

---

## Model weights

Not in this repository, and not covered by its licence.

`model_store/<predictor>/` holds `predictor.py`, a `README.md`, and a `model_card.json` for 8 of
the 15. The weights go in `model_store/model_weights/<predictor>/`, which is gitignored.
`common/model_sync.py` fetches them from `VERTEX_MODEL_STORE` on first use, or leaves them alone
if that points at a local directory.

**How outside users obtain weights is unresolved** — see `docs/LICENSING.md`. The code treats the
source as configuration, so settling it does not require code changes.

---

## The accounts service exists because the API has no auth

`src/backend/api_e2e/api.py` has no authentication and gains none. It is safe only because it
publishes no host port and `services/accounts/` is the only route to it. *Protected by the
frontend is not protection.*

The proxy is deliberately dumb — it forwards method, path, query, body and a header allow-list —
with three narrow exceptions for per-user privacy, each a shape match rather than a copy of the
API's route table. See `services/accounts/README.md`.

---

## Things worth knowing before you trust a number

Recorded in full in `docs/BASELINE.md` and `docs/BRIEF_VALIDITY.md`:

- **`status` never reflects the Vertex job state.** A worker that dies without writing
  `results.json` reports `"pending"` forever. `vertex_state` is the field that tells the truth.
- **No run carries a model identity.** Nothing in a stored result says which models produced it.
- **The brief vocabulary contradicts itself**: the biofilm model scores 13 pathogens while a brief
  may request 3, and `antibiofilm` is a valid generation tag but not a valid desired function.
- **8 of 16 predictors ship no model card**, so their applicability domain is undocumented.

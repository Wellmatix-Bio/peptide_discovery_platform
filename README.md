# Wellmatix Peptide Platform

AI-driven discovery platform for wound-healing therapeutic peptides. Given a
therapeutic product brief (wound context, desired biological functions, target
pathogens, and constraints), the pipeline generates, screens, and ranks
candidate peptide sequences, returning a synthesis-oriented shortlist together
with safety, developability, and mechanistic evidence for each candidate.

---

## Architecture

The platform is a linear stage pipeline:

```
s01_therapeutic_product_brief   Therapeutic product brief -> machine-readable Brief
s04_candidate_generation        Route A (GA, reference-guided) + Route B (ProtGPT2 LoRA, de novo)
s05_physchem_screening          Physicochemical properties, soft flags + hard rejects
s06_functional_models           Antimicrobial, migration, angiogenesis, immunomodulation
s07_structure_mechanism         ESMFold structure + pathway-engagement mechanism summary
s08_safety_developability       Hemolysis, cytotoxicity, aggregation, cleavage stability
s09_synthesis_cmc                Rule-based synthesis difficulty, cost bands, purity ceiling
s11_ranking                     Weighted multi-objective ranking
```

Each candidate stage declares the fields it `requires` and `produces`; a stage
may filter candidates out directly based on its own thresholds, logging what it
removed and why. `s02_wound_biology_and_targets` and `s03_data_integration`
have working implementations but are force-disabled for the deployed job path
(see `src/schemas/e2e_config.py`). `s10`, `s12`-`s14` exist as directories
under `src/pipeline/` with no implementation yet.

### Remote execution flow

```
Client (POST /api/v1/jobs/create)
    |
    v
FastAPI job API (src/backend/api_e2e/api.py)
    |  validates request, writes a staging config to GCS,
    |  submits a Vertex AI Custom Job
    v
Vertex AI Custom Job
    |  runs the worker container (src/backend/worker_e2e/, Dockerfile)
    |  pulls its config from GCS, syncs model weights from GCS
    v
Docker worker (backend.worker_e2e.worker)
    |  executes PipelineRunner over the configured stages
    v
Google Cloud Storage
    - config.json, results.json, audit_log.jsonl
    - candidates/<stage>.jsonl (per-stage boundary output)
    - candidates_final.json (final ranked shortlist)
```

Clients poll job status/results through the same FastAPI service
(`GET /api/v1/jobs/{job_id}/status`, `GET /api/v1/jobs/{job_id}/results`),
which reads those same GCS artifacts rather than talking to the worker
directly.

---

## Requirements

- **Python** 3.11+
- **Docker** with GPU support (`--gpus all`) for running the worker container
  locally; NVIDIA Container Toolkit installed on the host.
- **GCP project** with:
  - Vertex AI API enabled (for Custom Jobs)
  - A GCS bucket for model weights, run artifacts, and seed sequences
  - Artifact Registry (or another registry) to host the worker image
  - A service account with permissions to create/cancel Vertex Custom Jobs and
    read/write the GCS paths above
- **GPU**: the worker loads several PyTorch models (ESMFold, ProtGPT2+LoRA,
  ESM2, XGBoost/sklearn ensembles) and is CUDA-accelerated. CPU-only execution
  works but is significantly slower. The default Vertex accelerator is an
  NVIDIA T4 (Turing) — model code must avoid `bfloat16` (Ampere+ only) and use
  `float16` instead; see `model_store/routeb_protgpt2_lora_v1/predictor.py` for
  the reference pattern.

---

## Installation

```bash
git clone <repo-url> wellmatix-peptide-platform
cd wellmatix-peptide-platform

python -m venv .venv
source .venv/bin/activate    # .venv\Scripts\activate on Windows

pip install -r requirements.txt
pip install pytest

cp .env.example .env
```

---

## Configuration

All configuration is via environment variables, loaded from `.env` at the
repository root (see `src/common/env.py`). Copy `.env.example` and fill in
real values — never commit actual credentials or a filled-in `.env`.

```
DEV_MODE=false
VERTEX_CLOUD_PROJECT=<your-gcp-project-id>
VERTEX_LOCATION=us-central1
VERTEX_ARTIFACTS_DIR=gs://<your-bucket>/artifacts
VERTEX_MODEL_STORE=gs://<your-bucket>/model_weights
SEED_CANDIDATES_FILE=gs://<your-bucket>/curated_peptides.fasta
WORKER_IMAGE_URI=<region>-docker.pkg.dev/<project>/<repository>/worker-e2e:<tag>
WORKER_SERVICE_ACCOUNT=<worker-runtime>@<project>.iam.gserviceaccount.com
MACHINE_TYPE=n1-standard-4
ACCELERATOR_TYPE=NVIDIA_TESLA_T4
ACCELERATOR_COUNT=1
```

| Variable | Purpose |
|---|---|
| `DEV_MODE` | `true` for local runs against local paths/weights; `false` for the deployed worker path (GCS paths, real job IDs). |
| `VERTEX_CLOUD_PROJECT` / `VERTEX_LOCATION` | Identify the Vertex AI project/region the job API submits Custom Jobs to. |
| `VERTEX_ARTIFACTS_DIR` | GCS prefix where run configs, status, and results are written. |
| `VERTEX_MODEL_STORE` | GCS prefix model weights are synced from on first use (`common/model_sync.py`). |
| `SEED_CANDIDATES_FILE` | GCS path to the seed peptide FASTA used when a run doesn't start from de novo generation alone. |
| `WORKER_IMAGE_URI` | The worker image the job API tells Vertex to run. |
| `WORKER_SERVICE_ACCOUNT` | Service account the Custom Job runs as. |
| `MACHINE_TYPE` / `ACCELERATOR_TYPE` / `ACCELERATOR_COUNT` | Vertex Custom Job machine spec; default to `n1-standard-4` / `NVIDIA_TESLA_T4` / `1` if unset. |

Authentication uses Google Application Default Credentials
(`gcloud auth application-default login` locally; the service account
attached to the job when running on Vertex).

---

## How to run

### Local API server

```bash
cd src
python -m backend.api_e2e.api
# or: uvicorn backend.api_e2e.api:app --host 127.0.0.1 --port 8080 --reload
```

Then submit a job (see `src/backend/api_e2e/example_request.json` for a
complete, valid request body):

```bash
curl -X POST http://127.0.0.1:8080/api/v1/jobs/create \
  -H "Content-Type: application/json" \
  -d @src/backend/api_e2e/example_request.json
```

This submits a real Vertex AI Custom Job (billed GPU time) once
`WORKER_IMAGE_URI`/`WORKER_SERVICE_ACCOUNT` point at real resources.

### Local worker (Docker), without submitting to Vertex

Build the worker image from the repository root:

```bash
docker build -f src/backend/worker_e2e/Dockerfile -t worker-e2e:local .
```

Run it directly against a self-contained run config (local or `gs://`):

```bash
docker run --gpus all \
  --env-file .env \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  worker-e2e:local \
  --config gs://<your-bucket>/configs/my-run.json
```

The run config shape matches `RunConfig` (`src/schemas/run_config.py`):
`run_id`, `seed`, `seed_candidates_path`, `artifacts_dir`, `model_store`,
`stages`. This is a different (flatter) shape than the API's
`CreateJobRequest` body — the API injects the GCS paths server-side rather
than accepting them from the client.

### Local pipeline (no worker/API, dev only)

```bash
python main.py configs/test_run.yaml
```

`configs/test_run.yaml` is the run manifest (`run_id`, `seed`,
`seed_candidates_path`, `artifacts_dir`, `model_store`); the matching
`configs/runs/<run_id>.yaml` holds per-stage `enabled`/`params`. `run_id` from
the manifest is only honored when `DEV_MODE=true`.

### Vertex AI deployment / job execution

See [Deployment](#deployment) below for building and pushing the image, then
submit jobs through the API's `/api/v1/jobs/create` endpoint as shown above.

---

## Inputs and outputs

### Request schema

`POST /api/v1/jobs/create` takes `{"request_id": str, "stages": {...}}`. Full
field-by-field reference: [`docs/parameter_usage.md`](docs/parameter_usage.md)
and the machine-readable contract [`openapi.yaml`](openapi.yaml). Stage
parameters are validated per stage name (`src/schemas/stage_configs.py`);
unknown stage names or unknown fields on a public stage object are rejected.

### `request_id`, `job_id`, and the Vertex job name

- **`request_id`**: caller-supplied, used to name the staging config object
  (`<artifacts_dir>/requests/<request_id>/config.json`) and as a Vertex job
  label. Not unique-enforced by the API itself.
- **Vertex job name**: the full resource name Vertex assigns on creation,
  e.g. `projects/<project>/locations/<region>/customJobs/<numeric-id>`.
- **`job_id`** (as used in status/results/cancel URLs): the numeric suffix of
  the Vertex job name. The pipeline's own `run_id` is set to this same value
  after job creation — so GCS run artifacts live under
  `<artifacts_dir>/runs/<job_id>/`, keyed by the Vertex job ID, not by
  `request_id`.

### Where files go in GCS

```
<artifacts_dir>/
  requests/<request_id>/config.json     staging config, written before job creation
  runs/<job_id>/
    config.json                         final config (run_id set to job_id)
    config_snapshot.yaml                fully-resolved config, written at run start
    results.json                        {"run_id", "status", "progress", "stage"} while running;
                                         {"status": "success", "n_final": N} or
                                         {"status": "failed", "error": "..."} once terminal
    audit_log.jsonl                     per-stage/per-prediction audit trail
    candidates/<stage_name>.jsonl       per-stage boundary output (survivors only)
    candidates_final.json               final ranked shortlist
    stats_<run_id>.txt                  summary (DEV_MODE only)
    feature_cache/                      serialized shared ESM2/descriptor cache
```

### Result format

`GET /api/v1/jobs/{job_id}/results` returns `status`, `n_final`,
`total_candidates`, `ranked_candidates`, `insufficient_evidence_candidates`,
per-ranking-component summary statistics (`component_stats`), and the full
candidate list (each with its `predictions` dict populated by every stage that
ran).

---

## Project structure

```
platform/
├── src/
│   ├── backend/
│   │   ├── api_e2e/          FastAPI job-submission service (create/status/results/cancel)
│   │   └── worker_e2e/       Docker worker entrypoint + Dockerfile, run on Vertex
│   ├── pipeline/             s01-s14 stage packages + feature_extractor.py + base.py
│   ├── schemas/               Candidate, RunConfig, Brief, and related pydantic models
│   ├── common/                 audit, GCS/local storage, GPU release, env loading, stats, logging
│   ├── registry.py            stage-name -> stage-class registration
│   └── runner.py               PipelineRunner: loads config, resolves stages, executes them
├── model_store/               model weights + predictor.py wrappers, one directory per model (gitignored)
├── configs/                    run manifests and per-run stage configs (configs/runs/)
├── data/                       briefs/, raw/ seed sequences (gitignored except manifests)
├── docs/                       architecture, stage contracts, schema changelog, parameter reference
├── tests/                      unit (per-stage), integration, and schema tests
├── openapi.yaml                machine-readable API contract
├── API_SPECIFICATION.md        job-creation/status API reference
├── requirements.txt
└── README.md
```

---

## Development / testing

```bash
pytest                          # full suite
pytest tests/unit/pipeline      # per-stage unit tests
pytest tests/integration        # API + worker integration tests (mocked GCP)
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest.

Adding a stage or model:

1. Create the package under `src/pipeline/sNN_<name>/` with a `stage.py`
   defining a `CandidateStage` (or `SetupStage` for s01-s03) subclass.
2. Register the class in `src/registry.py`'s `SETUP_STAGE_CLASSES` or
   `CANDIDATE_STAGE_CLASSES` — an unregistered stage never executes.
3. If the stage calls a model, add it under `model_store/<model_name>_vN/`
   with its own `predictor.py`, following the existing
   lazy-load-on-first-`predict()` pattern (see `common/model_sync.py`).
4. Add a unit test under `tests/unit/pipeline/test_sNN_<name>.py`.

---

## Deployment

### 1. Build and push the worker image

```bash
gcloud builds submit . \
  --config=src/backend/worker_e2e/cloudbuild.yaml \
  --substitutions=_IMAGE_URI=<region>-docker.pkg.dev/<project>/<repository>/worker-e2e:<tag>
```

Set `WORKER_IMAGE_URI` in `.env` to the same value.

### 2. Required IAM permissions

The identity running the job API (locally or however it's hosted) needs:
- `roles/aiplatform.user` (create/get/cancel Custom Jobs)
- `roles/storage.objectAdmin` (or narrower, scoped to the artifacts/model
  bucket)

The `WORKER_SERVICE_ACCOUNT` the Custom Job itself runs as needs:
- `roles/storage.objectViewer` on the model-store bucket (reads weights)
- `roles/storage.objectAdmin` on the artifacts bucket (reads its config,
  writes results/candidates)

### 3. Start a Custom Job

Jobs are started through the API (`POST /api/v1/jobs/create`), not manually
via `gcloud`; see [How to run](#how-to-run).

---

## Troubleshooting

**GCS `403` on `storage.objects.get`**
The identity making the call (your local ADC user, or
`WORKER_SERVICE_ACCOUNT` on Vertex) lacks read access to the bucket/prefix. On
a local Docker run, confirm `GOOGLE_APPLICATION_CREDENTIALS` points at a
mounted, unexpired ADC file (`gcloud auth application-default login` if
expired).

**Artifact Registry permission denied on push**
The identity running `gcloud builds submit`/`docker push` needs
`roles/artifactregistry.writer` on the target repository.

**GPU/CUDA issues on Vertex**
The default accelerator is `NVIDIA_TESLA_T4` (Turing, compute capability 7.5)
— it does not support `bfloat16` natively. Any model code using
`torch.bfloat16` (directly, or via `bnb_4bit_compute_dtype`) will fail or run
severely degraded; use `torch.float16` instead. `esmfold_v1`'s existing
`.half()` usage is the T4-safe reference pattern.

**Model-weight download/cache issues**
Weights sync from `VERTEX_MODEL_STORE` on first use per process
(`common/model_sync.py`), a no-op under `DEV_MODE=true` or when
`VERTEX_MODEL_STORE` is a local path. A model appearing to "hang" on first
call is usually this download; check `sync_model_weights`'s `download_dir`
progress logs. If a model's local `model_weights/<name>/` directory is present
but corrupt/partial, delete it and let the next run re-sync (or run
`scripts/smoke_test_model_sync.py <model_name>` to test the sync path in
isolation).

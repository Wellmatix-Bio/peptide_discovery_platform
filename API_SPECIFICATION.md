# End-to-end Job API Specification

This specification covers **job creation and status checks only**.
Machine-readable contract: [`openapi.yaml`](openapi.yaml), OpenAPI 3.1.
Implementation: [`src/backend/api_e2e/api.py`](src/backend/api_e2e/api.py).
Stage validation: [`src/schemas/stage_configs.py`](src/schemas/stage_configs.py)
and [`src/schemas/e2e_config.py`](src/schemas/e2e_config.py).

Base URL: `<api-host>`. All bodies use `application/json`.
No caller authentication is defined in the application routes; deployment-level
access controls are separate. The API uses Google Application Default Credentials.

## Create a job

`POST /api/v1/jobs/create`

Submits one Vertex AI Custom Job running the full pipeline. This endpoint does
not resume a previous run or submit separate stage groups.

### Request

| Field | Type | Required | Meaning |
|---|---|---|---|
| `request_id` | string | Yes | Nonempty identifier for this submission |
| `stages` | object | Yes | Stage name mapped to its parameters |

Unknown top-level fields and unknown stage names are rejected. There is no
top-level `seed`, `run_id`, or `job_id` input. The pipeline seed comes from
`stages.s01_therapeutic_product_brief.seed` (default `42`).

```json
{
  "request_id": "request-001",
  "stages": {
    "s01_therapeutic_product_brief": {
      "seed": 42,
      "brief": {
        "wound_context": ["chronic", "infected"],
        "desired_functions": ["antimicrobial"],
        "pathogens": ["Staphylococcus_aureus"],
        "min_length": 6,
        "max_length": 30,
        "dosing_interval_hours": 48
      }
    },
    "s04_candidate_generation": {"n_peptides": 100},
    "s05_physchem_screening": {"ph": 7.4},
    "s11_ranking": {}
  }
}
```

The complete example is [`example_request.json`](src/backend/api_e2e/example_request.json).
Parameters go directly under each stage name. The API normalizes them into the
worker's internal `enabled`/`params` representation. The internal wrapped form
is also accepted, but flat parameter keys cannot be mixed with `params`.

### Stage validation

| Stage | Default when omitted |
|---|---|
| `s01_therapeutic_product_brief` | Required; omission is rejected |
| `s04_candidate_generation` | Schema defaults |
| `s05_physchem_screening` | Schema defaults |
| `s06_functional_models` | Schema defaults |
| `s07_structure_mechanism` | Schema defaults |
| `s08_safety_developability` | Schema defaults; hemolysis v1 only |
| `s09_synthesis_cmc` | Schema defaults |
| `s11_ranking` | Always enabled with the built-in ranking policy |

Enabled stages use the exact model selected by `STAGE_PARAMS_MODELS`, including
nested validation and defaults. Unknown parameter keys are ignored according to
`BaseStageParams`. Disabled stages skip parameter validation. Add `"enabled": false`
inside a stage to disable it, except Stage 1 (required) and Stage 11 (forced enabled).

`BriefFields` requires wound context, desired functions, pathogens, min/max
length, and dosing interval hours. Length is 6-50 residues; dosing interval is
1-168 hours. The OpenAPI component schemas enumerate allowed vocabularies.
`antibiofilm` is not currently an accepted desired function. Extra brief fields
are ignored. Some validated brief fields are not retained by the narrower
runtime `Brief` model.

Stage 11 weights, modifiers, normalization, and input mappings live in code.
Send `"s11_ranking": {}` or omit it. The current normalizer discards supplied
Stage 11 settings and always uses an enabled stage with no parameters.

### Submission and configuration

1. Submit the job with `--config <artifacts_dir>/requests/<request_id>/config.json`
   and `--wait-for-config 120`.
2. Receive the Vertex resource name and extract its numeric job ID.
3. Set the internal pipeline `run_id` to that numeric ID.
4. Write identical JSON configs to the request path and
   `<artifacts_dir>/runs/<numeric_job_id>/config.json`.

The worker starts when the request-path config appears, waiting at most 120 seconds
for a missing file. The API returns only after both writes complete. A config
publication failure returns 502 and triggers a best-effort attempt to stop the job.

Request IDs are not deduplicated. Use a unique `request_id` for each submission,
since reusing one shares the request-path config with earlier jobs.

### Response: 202 Accepted

```json
{
  "request_id": "request-001",
  "job_id": "projects/123/locations/us-central1/customJobs/456",
  "config_path": "gs://bucket/artifacts/runs/456/config.json",
  "result_path": "gs://bucket/artifacts/runs/456"
}
```

| Field | Meaning |
|---|---|
| `request_id` | Echoed request identifier |
| `job_id` | Full Vertex resource name; pass unchanged to the status endpoint |
| `config_path` | Complete JSON worker config stored under the numeric job ID |
| `result_path` | Artifact directory for this job |

A 202 means submission and config publication succeeded, not that the pipeline
finished. The response has no separate `run_id` field.

## Check job status

`GET /api/v1/jobs/{job_id}/status`

The required `job_id` is the full resource name, not only its numeric suffix:

```text
GET /api/v1/jobs/projects/123/locations/us-central1/customJobs/456/status
```

It must match `projects/<project>/locations/<region>/customJobs/<numeric_id>`.
The server uses a catch-all path parameter. Clients and proxies must preserve the
slash-separated resource path; some generated clients require custom path handling.

### Response: 200 OK

```json
{
  "job_id": "projects/123/locations/us-central1/customJobs/456",
  "status": "running",
  "stage": "s06_functional_models",
  "vertex_state": "JOB_STATE_RUNNING",
  "error": null
}
```

| Field | Meaning |
|---|---|
| `job_id` | Full Vertex job resource name |
| `status` | From the worker's `results.json` (`"pending"` if not yet written): `running`, `success`, or `failed` |
| `stage` | From `results.json`: the stage the worker was on at its last progress write, or `"pending"` if not yet written |
| `vertex_state` | Real Vertex `CustomJob.state` enum name, queried live (e.g. `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`) |
| `error` | Vertex error message, or null |

`vertex_state` is the ground truth for whether the underlying Custom Job is
still running, succeeded, or failed at the infrastructure level. `status`/
`stage` reflect the worker's own last self-reported progress and can lag
behind `vertex_state` — in particular, a job that failed hard (crashed,
OOM-killed, cancelled) before writing a final `results.json` can leave
`status` stuck at `"running"` and `stage` at an earlier value even though
`vertex_state` already shows a terminal state. Treat `vertex_state` as
authoritative for whether the job is still executing.

## Error responses

Errors use `{"detail": "message"}` or FastAPI's validation-error list under
`detail`. Unexpected application failures may produce an unstructured HTTP 500.

| HTTP status | Create | Status check |
|---|---|---|
| `404` | Not applicable | Job not found |
| `422` | Invalid request, stage parameters, brief, or unsupported hemolysis version | Invalid job resource name |
| `502` | Google API or config-publication failure or missing credentials | Google API failure |
| `503` | Missing server settings or configured paths not using GCS | Missing server settings |

## Artifacts

Read results from GCS after status becomes `success`. Neither endpoint downloads
candidate data. Files under `result_path` include:

- `config.json`: JSON worker input.
- `config_snapshot.yaml`: runner provenance snapshot, not an input format.
- `candidates/<stage_name>.jsonl`: candidate-stage boundary snapshots.
- `candidates_final.json`: final candidates.
- `stats_<numeric_job_id>.txt`: run statistics.
- `run_context.json`: brief and optional objectives.
- `audit_log.jsonl`: execution audit.
- `results.json`: worker status (`running`, `success`, or `failed`).
- `feature_cache/`: saved shared features when enabled.

Internal `run_id` fields contain the numeric Vertex ID. Failed jobs can leave
partial outputs. Worker status can be absent or stale after early failures or
forced termination; Vertex is authoritative.

## Server settings

Required: `VERTEX_CLOUD_PROJECT`, `VERTEX_LOCATION`, `VERTEX_ARTIFACTS_DIR`, `VERTEX_MODEL_STORE`,
`SEED_CANDIDATES_FILE`, `WORKER_IMAGE_URI`, `WORKER_SERVICE_ACCOUNT`.
Use the project ID or number, not the display name. Artifact/model/seed paths
must be GCS URIs. The service-account setting is an email, not a key file.

`MACHINE_TYPE` defaults to `n1-standard-4`; `ACCELERATOR_TYPE` and
`ACCELERATOR_COUNT` default to `NVIDIA_TESLA_T4` and `1`. The current API
specifies one replica and a 200 GB SSD boot disk. T4 is Turing architecture
and does not support `bfloat16` natively — worker/model code must use
`torch.float16` for any explicit compute dtype instead. The worker receives
`DEV_MODE=false` and every `.env` key present at API-server startup (see
`common.env.DOTENV_KEYS`) as container environment variables. All containers
install the root [`requirements.txt`](requirements.txt).

The OpenAPI file is a standalone contract for these two operations. It does not
change the service's routes or its automatically generated `/openapi.json`.

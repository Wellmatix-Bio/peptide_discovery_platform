# End-to-end Job API Specification

This specification covers **job creation, status checks, result reads and cancellation** for the three pipelines the API runs: **wound healing** (the default), **metabolic** (weight control) and **cosmetic** (skin beauty).
Machine-readable contract: `openapi.yaml`, OpenAPI 3.1.
Implementation: src/backend/api_e2e/api.py.
Stage and configuration validation: src/schemas/stage_configs.py
and src/schemas/e2e_config.py

All bodies use `application/json`.
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
| `pipeline` | string | No | `wound_healing` (default), `metabolic` or `cosmetic`. Selects the stage set and the worker image; it must match the shape of `stages` (see [Metabolic pipeline](#metabolic-pipeline) and [Cosmetic pipeline](#cosmetic-pipeline)) |
| `stages` | object | Yes | Stage name mapped to its parameters |

Unknown top-level fields and unknown stage names are rejected. There is no
top-level `seed`, `run_id`, or `job_id` input. The pipeline seed comes from the brief stage's `seed`
(`s01_therapeutic_product_brief` or `s01_metabolic_therapeutic_product_brief`; default `42`).

### Choosing a pipeline

`pipeline` picks the stage set, the worker image and the brief. The brief stage inside `stages` must agree with it.

| | Wound healing | Metabolic | Cosmetic |
|---|---|---|---|
| `pipeline` | `wound_healing` (default; may be omitted) | `metabolic` | `cosmetic` |
| Brief stage (identifies the request shape) | `s01_therapeutic_product_brief` | `s01_metabolic_therapeutic_product_brief` | `s01_cosmetic_therapeutic_product_brief` |
| Brief fields | wound context, desired functions, pathogens, min/max length (6-50), dosing interval | `target_receptors` (GLP1R, GIPR, GCGR), min/max length (5-50) | primary and secondary function, target depth, application area, product pH, min/max length (6-20) |
| Stages that differ | `s04_candidate_generation`, `s05_physchem_screening`, `s06_functional_models`, `s08_safety_developability`, `s11_ranking` | `s04_metabolic_candidate_generation`, `s05_metabolic_physchem_screening`, `s06_metabolic_functional_screening`, `s08_metabolic_safety_developability`, `s11_metabolic_ranking` | `s04_cosmetic_candidate_generation`, `s05_cosmetic_physchem_screening`, `s06_cosmetic_functional_models`, `s08_cosmetic_safety_developability`, `s09_cosmetic_synthesis_cmc`, `s11_cosmetic_ranking` |
| Shared stages | `s07_structure_mechanism`, `s09_synthesis_cmc` | `s07_structure_mechanism`, `s09_synthesis_cmc` | `s07_structure_mechanism` |
| Worker image setting | `WORKER_IMAGE_URI` | `WORKER_METABOLIC_IMAGE_URI` | `WORKER_COSMETIC_IMAGE_URI` |
| Seed file setting | `SEED_CANDIDATES_FILE` | `SEED_CANDIDATES_FILE_METABOLIC` | `SEED_CANDIDATES_FILE_COSMETIC` |
| Vertex job display name | `ai-peptide-discovery-e2e-<request_id, first 12 characters>` | `ai-peptide-discovery-metabolic-<request_id, first 12 characters>` | `ai-peptide-discovery-cosmetic-<request_id, first 12 characters>` |
| Vertex job label `pipeline` | `wound-healing` | `metabolic` | `cosmetic` |
| Example | [`example_request.json`](src/backend/api_e2e/example_request.json) | below | below |

A request whose stage names belong to a different pipeline (for example a wound stage name under `"pipeline": "metabolic"`) returns 422.

### Wound-healing pipeline

`pipeline` omitted or `"wound_healing"`. Example:

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

#### Stage validation

| Stage | Default when omitted |
|---|---|
| `s01_therapeutic_product_brief` | Required; omission is rejected |
| `s04_candidate_generation` | Schema defaults |
| `s05_physchem_screening` | Schema defaults |
| `s06_functional_models` | Schema defaults |
| `s07_structure_mechanism` | Schema defaults |
| `s08_safety_developability` | Schema defaults |
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

### Metabolic pipeline

`"pipeline": "metabolic"` runs the metabolism / weight-control branch on the same endpoint. `stages` must then
be the metabolic request: it is recognized by `s01_metabolic_therapeutic_product_brief`, and the wound-healing
stage names are rejected. A `pipeline` that disagrees with the shape of `stages` returns 422 and names the
value to use. The job runs on the image in `WORKER_METABOLIC_IMAGE_URI` and is labelled `pipeline: metabolic`.

| Stage | Default when omitted |
|---|---|
| `s01_metabolic_therapeutic_product_brief` | Required; omission is rejected |
| `s04_metabolic_candidate_generation` | Schema defaults |
| `s05_metabolic_physchem_screening` | Schema defaults |
| `s06_metabolic_functional_screening` | Schema defaults |
| `s07_structure_mechanism` | Schema defaults (shared with the wound pipeline; stays on) |
| `s08_metabolic_safety_developability` | Schema defaults |
| `s09_synthesis_cmc` | Schema defaults (shared with the wound pipeline; stays on) |
| `s11_metabolic_ranking` | Schema defaults |

The brief (`MetabolicBrief`) requires `min_length` and `max_length` (5-50) and `target_receptors` (GLP1R, GIPR
or GCGR, exact spelling; duplicates are dropped). Stage 8 takes `hemolysis_screening` (default `true`);
`false` skips hemolysis and Stage 11 drops it from the safety score. Stage 6 predicts each candidate's anti-inflammatory probability and rejects those below `anti_inflammatory_probability_min` (default `0.5`); Stage 11 also scores it. Stage 4 can override the length range and receptors,
and takes `n_peptides`, `route_a_ga_params`, `lam` and `tol`. The OpenAPI schemas list every stage's parameters.

```json
{
  "request_id": "request-002",
  "pipeline": "metabolic",
  "stages": {
    "s01_metabolic_therapeutic_product_brief": {
      "brief": {
        "target_receptors": ["GLP1R", "GIPR"],
        "min_length": 6,
        "max_length": 40
      }
    },
    "s04_metabolic_candidate_generation": {"n_peptides": 100},
    "s08_metabolic_safety_developability": {"hemolysis_screening": true, "hemolysis_probability_reject_max": 0.5}
  }
}
```

### Cosmetic pipeline

`"pipeline": "cosmetic"` runs the skin-beauty branch on the same endpoint. `stages` must then be the cosmetic request: it is recognized by `s01_cosmetic_therapeutic_product_brief`, and the wound-healing and metabolic stage names are rejected. A `pipeline` that disagrees with the shape of `stages` returns 422. The job runs on the image in `WORKER_COSMETIC_IMAGE_URI`, takes its seed peptides from `SEED_CANDIDATES_FILE_COSMETIC`, and is labelled `pipeline: cosmetic`.

The seed peptides are a server setting. A request cannot supply or change them.

| Stage | Default when omitted |
|---|---|
| `s01_cosmetic_therapeutic_product_brief` | Required; an empty `brief` is valid and takes every default |
| `s04_cosmetic_candidate_generation` | Schema defaults |
| `s05_cosmetic_physchem_screening` | Schema defaults |
| `s06_cosmetic_functional_models` | Schema defaults |
| `s07_structure_mechanism` | Schema defaults (shared with the other pipelines; stays on) |
| `s08_cosmetic_safety_developability` | Schema defaults |
| `s09_cosmetic_synthesis_cmc` | Schema defaults |
| `s11_cosmetic_ranking` | Schema defaults |

The brief (`CosmeticBrief`):

| Field | Type | Default | Constraint |
|---|---|---|---|
| `functions.primary` | `collagen_ecm`, `antioxidant` or `pigmentation` | `collagen_ecm` | |
| `functions.secondary` | up to 2 of the same values | `[]` | no repeats; must not include the primary |
| `target_depth` | `outer_layer`, `epidermis` or `dermis` | set from the primary function: `collagen_ecm` dermis, `antioxidant` and `pigmentation` epidermis | |
| `application.area` | `face`, `eye_area` or `body` | `face` | `eye_area` makes the safety level strict |
| `application.sensitive_skin` | boolean | `false` | `true` makes the safety level strict |
| `product.ph` | number | `5.5` | 4.0 to 7.5 |
| `peptide.min_length` | integer | `6` | 6 to 20; the floor is 6 residues |
| `peptide.max_length` | integer | `12` | 6 to 20; at least `min_length` |

Stage 4 `min_length` has the same floor of 6. Stage 8 takes `reject_skin_sensitizers` (default `true`): when `true`, a predicted sensitizer is rejected; when `false`, it is kept and flagged. Stage 11 penalises flagged sensitizers. The penetration filter is currently commented out, so `target_depth` does not remove any candidate; the rule result is recorded only.

```json
{
  "request_id": "request-003",
  "pipeline": "cosmetic",
  "stages": {
    "s01_cosmetic_therapeutic_product_brief": {
      "brief": {
        "functions": {"primary": "collagen_ecm", "secondary": ["antioxidant"]},
        "application": {"area": "eye_area"},
        "product": {"ph": 5.5},
        "peptide": {"min_length": 6, "max_length": 12}
      }
    },
    "s08_cosmetic_safety_developability": {"reject_skin_sensitizers": true}
  }
}
```

### Submission behaviour (all pipelines)

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
  "result_path": "gs://bucket/artifacts/runs/456"
}
```

| Field | Meaning |
|---|---|
| `request_id` | Echoed request identifier |
| `job_id` | Full Vertex resource name; pass unchanged to the status endpoint |
| `result_path` | Artifact directory for this job (the worker config is stored there as `config.json`) |

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
  "run_state": "running",
  "error": null
}
```

| Field | Meaning |
|---|---|
| `job_id` | Full Vertex job resource name |
| `status` | The worker's own last word, from its `results.json` (`"pending"` if not yet written): `running`, `success`, or `failed`. **It can be stale** — see `run_state` |
| `stage` | From `results.json`: the stage the worker was on at its last progress write, or `"pending"` if not yet written or null. Typed `string \| null` in the schema, but the handler always substitutes `"pending"`, so clients should not see null |
| `vertex_state` | Real Vertex `CustomJob.state` enum name, queried live (e.g. `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`) |
| `run_state` | `status` and `vertex_state` reconciled: `submitted`, `running`, `succeeded`, `failed`, `cancelled` or `abandoned`. **This is the field to show a user** |
| `error` | Vertex error message, or null |

### Why there are three fields and which to read

`status` is the worker's **self-report** and `vertex_state` is the **observed** job state. They
disagree whenever a worker dies without a final write: a container that is OOM-killed, pre-empted
or cancelled leaves `status` reading `"running"` — or `"pending"`, if it died before writing
anything — while `vertex_state` has already gone terminal. Both fields are individually honest and
neither answers "is this run over".

**`run_state` answers it.** The precedence:

1. A worker that wrote `success` or `failed` wins: a first-hand report about the run is more
   specific than the job's exit code.
2. Then Vertex's terminal states, because they are observed rather than self-reported.
   `cancelled` is kept distinct from `failed` — somebody chose to stop it.
3. A job Vertex finished with no successful result is **`abandoned`**: the container exited
   cleanly and produced nothing. Neither `pending` nor `succeeded` is true of it.
4. Otherwise the worker's `running`, or `submitted` when it has said nothing yet.

`status` and `stage` are unchanged and still mean exactly what they did, so a client reading them
keeps working. They are the worker's words; `run_state` is the answer.

## Read job results

`GET /api/v1/jobs/{job_id}/results`

Uses the same full-resource-name `job_id` as the status endpoint. Reads
`results.json` from the run directory; if the worker reported `success`, it also
reads `candidates_final.json`. Worker output files, not live Vertex state, are
the source of truth. A job that is still running or failed returns `status` and
`n_final` only, with an empty `candidates` list.

### Response: 200 OK

```json
{
  "job_id": "projects/123/locations/us-central1/customJobs/456",
  "run_id": "456",
  "status": "success",
  "n_final": 12,
  "ranked_candidates": 10,
  "insufficient_evidence_candidates": 2,
  "component_stats": {
    "wound_closure": {"count": 10, "mean": 0.62, "median": 0.60, "min": 0.31, "max": 0.94}
  },
  "candidates": [
    ...
  ]
}
```

| Field | Meaning |
|---|---|
| `run_id` | Numeric Vertex job ID |
| `n_final` | Final candidate count reported by the worker |
| `ranked_candidates` | Candidates whose ranking status is `ranked` |
| `insufficient_evidence_candidates` | Candidates in the final file that are not `ranked` |
| `component_stats` | Mean/median/min/max per ranking module, over ranked candidates' module `score` (after flag deductions); candidates without a score for a module are excluded |
| `candidates` | Typed summaries, see below |

Each entry of `candidates` is a flat
`CandidateResponse` projected from the candidate record (the full prediction
tree is not included; read `candidates_final.json` for it). Scalar fields are
`null` when the producing stage did not run or abstained.

**The shape depends on the pipeline.** The response carries `pipeline` (`wound_healing`, `metabolic` or `cosmetic`), taken from the job's stored config, and `candidates` has the matching shape. The table below is the wound-healing one. Metabolic candidates are described under [Metabolic candidate fields](#metabolic-candidate-fields) and cosmetic candidates under [Cosmetic candidate fields](#cosmetic-candidate-fields).

| Field | Source |
|---|---|
| `id` | Candidate `id`; `"?"` if the record has none |
| `sequence` | Top-level candidate `sequence` (not under `predictions`) |
| `final_score_rank` | `predictions.ranking.final_score_rank`, 1-based position by weighted `final_score`; `null` if not ranked |
| `pareto_rank` | `predictions.ranking.pareto_rank`, the Pareto front over the ranking modules' scores (after flag deductions): `1` is non-dominated, and candidates on the same front share a rank, so it is not a position. It does not affect the order of `final_score_rank`; `null` if not ranked |
| `final_score` | `predictions.ranking.final_score`; `null` if not ranked |
| `evidence_coverage` | `predictions.ranking.evidence_coverage` |
| `missing_modules` | `predictions.ranking.missing_modules` |
| `amp_probability` | `amp_probability` |
| `hemolytic_activity_phc50` | `hemolysis.phc50` |
| `molecular_weight` | `molecular_weight` (Daltons) |
| `net_charge` | `net_charge` |
| `instability_index` | `instability_index` |
| `deamidation_risk`, `oxidation_risk` | `<name>.risk_category`: `low`, `medium`, `high` |
| `solubility` | `solubility.score` |
| `aggregation_tendency` | `aggregation_tendency.score` |
| `anti_inflammatory_probability` | `anti_inflammatory_probability` |
| `angiogenic_activity` | `angiogenic_activity.angiogenic` |
| `proliferation_probability`, `migration_probability` | `proliferation_migration.proliferation` / `.migration`; default `0` (not null) when absent |
| `cytotoxicity_probability` | `cytotoxicity.score` |
| `cleavage_stability` | `cleavage_stability.score` |
| `log_mic_um` | `mic.log_mic_um`, keyed by `Escherichia coli`, `Staphylococcus aureus`, `Pseudomonas aeruginosa`; `{}` when absent |
| `pmbic` | `mbic.pmbic`, keyed by 13 pathogens (see `openapi.yaml`); `{}` when absent |
| `activated_pathways` | `mechanism.activated_pathways`: pathways whose v2 P(activator) is >= `pathway_engagement_min_probability` |
| `inhibited_pathways` | `mechanism.inhibited_pathways`: pathways whose v2 P(activator) is <= 1 - `pathway_engagement_min_probability` |

Pathogen keys here use spaces (`Staphylococcus aureus`), unlike the request's
brief vocabulary, which uses underscores (`Staphylococcus_aureus`). A key
outside the listed pathogens fails response validation.

### Metabolic candidate fields

For a `metabolic` job each entry of `candidates` is a `MetabolicCandidateResponse`. The top-level fields (`job_id`, `run_id`, `status`, `n_final`, `ranked_candidates`, `insufficient_evidence_candidates`, `component_stats`) are the same as above. Per-receptor fields are objects keyed by `GLP1R`, `GIPR` and `GCGR`; a value is `null` when it was not predicted.

| Field | Source |
|---|---|
| `id`, `sequence` | as for wound healing |
| `final_score_rank`, `pareto_rank`, `final_score`, `evidence_coverage`, `missing_modules` | `predictions.ranking.*`, as for wound healing |
| `pec50` | predicted potency per receptor, -log10(EC50 in M) |
| `ec50_nm` | the same potency as EC50 in nM, floored at the Stage 6 `ec50_floor_nm` |
| `binding_pkd` | predicted binding per receptor, pKd = 9 - log10(Kd in nM) |
| `agonism_probability` | agonism classifier, P(EC50 below 1 nM) per receptor; `null` when the classifier is not installed |
| `pec50_pkd_gap` | disagreement between the pEC50 and pKd models (geometric means over the target receptors); Route A candidates only, `null` for others |
| `metabolic_activity_score` | Stage 6 combined potency score in [0, 1] |
| `anti_inflammatory_probability` | `anti_inflammatory_probability` |
| `half_life_minutes` | `plasma_half_life.minutes` |
| `hemolytic_activity_phc50`, `p_hemolytic` | `hemolysis.phc50` and `hemolysis.p_hemolytic`; `null` when hemolysis screening is off |
| `cleavage_stability`, `solubility`, `aggregation_tendency`, `cytotoxicity_probability` | `<name>.score` |
| `molecular_weight`, `net_charge`, `instability_index`, `deamidation_risk`, `oxidation_risk` | as for wound healing |
| `activated_pathways`, `inhibited_pathways`, `undetermined_pathways` | `mechanism.*`, as for wound healing |

The wound-only fields (`amp_probability`, `log_mic_um`, `pmbic`, `angiogenic_activity`, `proliferation_probability`, `migration_probability`) are not in the metabolic shape.

### Cosmetic candidate fields

For a `cosmetic` job each entry of `candidates` is a `CosmeticCandidateResponse`. The top-level fields are the same as above, with these definitions for the counts:

- `ranked_candidates`: candidates with a `final_score`. Stage 11 scores every survivor, so this normally equals `n_final`.
- `insufficient_evidence_candidates`: candidates with at least one ranking component not screened.
- `component_stats`: mean, median, min and max of each ranking component's `score`, over the candidates that screened it, keyed by component name.

Each candidate has a `not_screened` list (the names of components not screened) and `component_scores`, one object per component with `score` and `screened`. A component that was not screened has a neutral score (0.5 by default), so it is not rewarded or penalised.

| Field | Source |
|---|---|
| `id`, `sequence` | as for wound healing |
| `final_score`, `final_score_rank`, `pareto_rank`, `evidence_coverage` | `predictions.ranking.final_score`, `.final_score_rank`, `.pareto_rank`, `.coverage` |
| `not_screened`, `component_scores` | `predictions.ranking.not_screened`, `.components` (each `score` and `screened`) |
| `primary_function` | `predictions.functional_verdict.primary_function` |
| `functional_scores` | `predictions.functional_scores`: per function, `score` and `pass`; the primary is always present, secondaries are recorded only |
| `antimicrobial_probability` | `predictions.antimicrobial.probability` |
| `molecular_weight`, `net_charge`, `hydrophobic_moment` | the Stage 5 properties of the same names; `molecular_weight` in Daltons, including modification shifts |
| `physchem_flags`, `physchem_flag_count` | `predictions.physchem_flags` (the flag names), `predictions.physchem_flag_count` |
| `aggregation_tendency`, `solubility` | `predictions.aggregation_tendency.score`, `predictions.solubility.score`; `null` when the model did not run (aggregation is skipped under 5 residues) |
| `irritation_tier`, `irritation_score` | `predictions.irritation.tier`, `.score`; `null` when irritation was not screened |
| `skin_sensitizer` | `predictions.skin_sensitization.sensitizer` (threshold 0.55, fixed) |
| `skin_sensitization_probability` | `predictions.skin_sensitization.p_sensitizer` |
| `hemolysis_probability` | `predictions.hemolysis.p_hemolytic` |
| `cytotoxicity` | `predictions.cytotoxicity.by_cell_type`: per cell type, `p_cytotoxic`, `requested` and `exact` |
| `half_life_log10_seconds` | `predictions.half_life.log10_seconds`; `null` outside the model's 5-50 residue range |
| `allergenicity_probability` | `predictions.allergenicity.p_allergen`; `null` under 50 residues |
| `cleavage_stability` | `predictions.cleavage_stability.score` |
| `safety_flags` | `predictions.safety_flags` |
| `synthesis_feasibility_probability`, `synthesis_flags` | `predictions.synthesis_feasibility.probability`, `.flags` |

<!-- REMINDER, do not delete: `penetration_passed` (`predictions.penetration.passed`) is left out of the cosmetic
     response for now, because the penetration filter is commented out in Stage 8 (no penetration model yet). Restore
     this row, and the Stage 8 reject, when a penetration model exists. See docs/TODO.md and docs/LIMITATIONS.md. -->

Fields for a stage that did not run, or that abstained, are `null`, as in the other pipelines. Half-life and allergenicity are `null` for sequences outside their model ranges, which is most cosmetic candidates.

## Cancel a job

`POST /api/v1/jobs/{job_id}/cancel`

Uses the same full-resource-name `job_id` as the status endpoint. Requests
cancellation of the Vertex Custom Job.

### Response: 200 OK

```json
{"job_id": "projects/123/locations/us-central1/customJobs/456", "status": "cancelling"}
```

| `status` | Meaning |
|---|---|
| `cancelling` | Cancellation was requested; the job may take time to stop |
| `cancelled` | The job was already cancelling or cancelled |
| `failed` | The job had already failed or expired |

Errors: `404` job not found, `422` invalid job resource name, `502` Google API
failure, `503` missing credentials or server settings.

### Errors

| HTTP status | Meaning |
|---|---|
| `404` | `results.json` missing, or the job succeeded but `candidates_final.json` is missing |
| `422` | Invalid job resource name |
| `503` | Missing server settings |

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

Read results from GCS after status becomes `success`. Only the results endpoint
returns candidate data, and only the summary fields above. Files under
`result_path` include:

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
`WORKER_IMAGE_URI`, `WORKER_SERVICE_ACCOUNT`.

Pipeline-specific settings are required only for jobs of that pipeline; a missing one returns 503:

| Pipeline | Worker image | Seed peptide file |
|---|---|---|
| `wound_healing` | `WORKER_IMAGE_URI` | `SEED_CANDIDATES_FILE` |
| `metabolic` | `WORKER_METABOLIC_IMAGE_URI` | `SEED_CANDIDATES_FILE_METABOLIC` |
| `cosmetic` | `WORKER_COSMETIC_IMAGE_URI` | `SEED_CANDIDATES_FILE_COSMETIC` |

Each seed file is a `gs://` FASTA. Seeds are a server setting, so each pipeline's
seed set is fixed by the operator and cannot be changed per request.
Use the project ID or number, not the display name. Artifact/model/seed paths
must be GCS URIs. The service-account setting is an email, not a key file.

`MACHINE_TYPE`/`ACCELERATOR_TYPE`/`ACCELERATOR_COUNT` are set explicitly in
`.env` to `n1-standard-8`/`NVIDIA_TESLA_T4`/`1`.

The OpenAPI file is a standalone contract for these four operations. It does not
change the service's routes or its automatically generated `/openapi.json`.

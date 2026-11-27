# End-to-end Job API Specification

This specification covers **job creation, status checks, result reads and cancellation** for the three pipelines the API runs: **wound healing** (the default), **metabolic** (weight control) and **cosmetic** (skin beauty). Two further jobs have their own sections below: the [food-peptide pipeline](#food-peptide-pipeline) and the [immunomodulation studio](#immunomodulation-studio).
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
| `stage` | From `results.json`: the stage the worker was on at its last progress write, or `"pending"` if not yet written or null; `"finished"` once the worker reported `success` (its last write has no stage). Typed `string \| null` in the schema, but the handler always substitutes a value, so clients should not see null |
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

## Food-peptide pipeline

A separate job that takes food proteins, either as an uploaded FASTA or chosen from the food protein library, and returns ranked peptides for biology review. It uses the same status and cancel envelopes. Without `DEV_MODE` it runs as a Vertex AI Custom Job on `WORKER_FOOD_IMAGE_URI`; with `DEV_MODE` it runs as a local process on local weights.

| Method and path | Purpose |
|---|---|
| `GET /api/v1/food/digestion/settings` | The digestion rules (Expasy PeptideCutter, with pattern, pH range and what the protease benchmark says about each), the default pathways, the `proteases` a run can be limited to (name to single-enzyme pathway id) and the request limits |
| `POST /api/v1/food/digestion/run` | Run the digestion engine now on uploaded or library proteins; returns the theoretical fragments with positions, pathways and masses, and a cleavage map. Same engine and settings as stage s03, so these are the fragments a run starts from. Synchronous, at most 50 proteins and 30,000 residues; a request can pick pathways (`pathways`) or proteases (`proteases`, any of `pepsin`, `trypsin`, `chymotrypsin`; not both) and the length and missed-cleavage limits but cannot supply rules. Chosen proteases run each alone, with the rule, stage and pH of its default pathway, and, when two or more are chosen, also together in digestion order (gastric stage, then intestinal stage); together uses the default pathway of the same stages when there is one (`trypsin_chymotrypsin`, `gastric_to_intestinal`), otherwise the protease names joined (`pepsin_trypsin`) |
| `GET /api/v1/food/library/summary` | What the library holds: release, proteins per food source and family, allergen-source flags |
| `GET /api/v1/food/library/proteins` | Browse the library: filters `food_source`, `family`, `q` (accession, name or family), `limit` (1 to 500), `offset`. No sequences are returned |
| `POST /api/v1/jobs/food/create` | Submit a FASTA (upload mode) or library proteins (library mode) |
| `GET /api/v1/jobs/food/{job_id}/status` | Same envelope as `/status`; `stage` names the stage that is running |
| `POST /api/v1/jobs/food/{job_id}/cancel` | Stop the job: `cancelling`, `cancelled` or `failed` (already over) |
| `GET /api/v1/jobs/food/{job_id}/results` | Ranked peptides with provenance, scores, model results, applicability, cut sites, evidence, rationale and limitations; the shortlist, controls, evidence gate, protein QC, cluster sources and the provenance check. Empty until the job succeeds |
| `GET /api/v1/jobs/food/{job_id}/dossiers/{peptide_id}` | Review dossier of one ranked peptide as JSON, as a readable page with `?format=html`, or as a PDF download with `?format=pdf` (the overview page, the shared page and one page for the peptide) |
| `GET /api/v1/jobs/food/{job_id}/shortlist-dossiers.pdf` | One PDF download: an overview page of the shortlisted peptides, a page with what is shared by all peptides, then one A4 landscape page per shortlisted peptide, in shortlist order |
| `GET /api/v1/jobs/food/{job_id}/review` | The biological approval gate of a finished run: its status, the current decision on each reviewed peptide, and the full history of decisions |
| `PUT /api/v1/jobs/food/{job_id}/review/{peptide_id}` | Save the biology lead's decision (`approve`, `hold` or `reject`), the reason and the reviewer for one ranked peptide. It never changes the rank |
| `GET /api/v1/jobs/food/{job_id}/experiment-sheet` | CSV for the biology review: the shortlist in order, then the reference controls. The assay-design cells, decision, reviewer and comments are empty |

**Two input modes; send exactly one.**

- **Upload:** `{"request_id": "...", "fasta": ">id description (newline) SEQUENCE ...", "food_source": "Whey", "allergen_source": true}`. `allergen_source` is `true`, `false` or `null` (not established; the default, never `false` unless sent). It applies to proteins the library does not match.
- **Library:** `{"request_id": "...", "library_proteins": ["P02754:17-178", ...]}`, substrate ids from `GET /api/v1/food/library/proteins`. Do not send `food_source` or `allergen_source`: the family, allergen-source flag, version and UniProt release come from the library. An unknown id is a `422` that names it.

**The library is optional.** An uploaded protein is compared with the library; a record that is exactly a library protein (same sequence) gets the library's family, flag and release, and the response lists each record as `exact_sequence`, `accession_sequence_differs` or `not_in_library` in `library_matches`. A protein that is not in the library is **never rejected for that**: it is reported as `not_in_library`, its family and allergen source are not established, and the run uses what the request supplied. The library is a copy of the Phase 1 protein library (524 mature-chain substrates of whey, casein, soy, pea and wheat proteins; UniProt release 2026_04, retrieved 2026-11-13), recorded in its manifest; its flags are the data foundation's by rule, not clinical findings. Every record is checked (standard residues, identifier, duplicates); a record that fails is returned in `rejected` and skipped, and a FASTA with no usable record is a `422`. A repeat of the same `request_id` returns the first job. Response `202`: the usual `request_id`, `job_id`, `result_path`, plus `accepted`, `rejected` (`id`, `severity`, `code`, `message`), `input_mode` and `library_matches`. The results route returns the same match record as `library`.

On Vertex the create route returns `503` when the model store is the placeholder or lacks a model (including `bitterness_predictor_v1`), before any GPU job starts. The worker fails the run if any final peptide does not trace to where it says it comes from (the provenance check).

**Local mode.** With `DEV_MODE=true` the job runs as a local process on the weights in `model_store/model_weights/`, with no Vertex job, bucket or image. Job ids then start with `projects/local/locations/local/customJobs/`; the process handle lives in the API process, so after an API restart a running job cannot be cancelled. In this mode the API also answers requests from any origin (CORS `*`), so a standalone page can call it; never set it in production. The worker's output goes to `<run folder>/worker.log`. `deploy/run-food-local.ps1` starts the API in this mode on port 8000 (the dashboard's default address): it links the seven weight folders the food pipeline reads (`esm2_t30_150M` and the allergenicity, cytotoxicity, solubility, aggregation, synthesis-feasibility and bitterness predictors) from `-WeightsDir` into `model_store/model_weights/`, and removes the links when you stop it with Ctrl+C. It needs the full `requirements.txt` environment, because the worker runs the models. Behind the accounts proxy these routes should need no change, judging from its code (it matches by shape, not by route list): a job id in the path is ownership-checked, `request_id` is replaced per user, and the CSV and HTML responses pass through. This has not been run end to end.

**Machine-readable contract.** `web/openapi/peptide.json` is generated from the running app (CI checks it is current) and includes every route in this section. `openapi.yaml` is hand-maintained and covers the original four operations only.

### Food-peptide response shapes

**Create `202`:** `request_id`, `job_id`, `result_path`, `accepted`, `rejected[]`, `input_mode` (`upload` or `library`) and `library_matches[]` (`id`, `status`, `library_substrate_id`, `protein_family`, `allergen_source`, `message`).

**Status `200`:** the same fields as `/status` (`status`, `stage`, `vertex_state`, `run_state`, `error`). Read `run_state`; it reads `cancelled` after a cancel.

**Results `200`:** `pipeline` (`"food"`), `job_id`, `run_id`, `status`, `n_final`, then, once the job has succeeded:
- `peptides[]`, best first. Each has `id`, `sequence`, `rank`, `origin_type` (`DIGESTION_DERIVED` or `SYNTHETIC_DESIGN`), `evidence_class` (`THEORETICAL`), `shortlist_position`, `shortlist_reason`, `pareto_front` (the peptide's Pareto front in the model-stage pool, kept apart from `rank` and `shortlist_position`), `objective_values` (its value on each objective the run used, 0 to 1, higher is favourable; for a model objective it is the ranking value, not the model's own output; empty when the run did not record its objectives), `parents[]` (protein, food source, allergen source, `protein_n_terminus` and `protein_c_terminus` (the two ends of the parent chain, codes as in the library listing; `null` when not recorded), position, pathway, enzymes, pH, and `digestion_detail`: the digestion run again for that origin, with the `stages` (enzymes and pH), `n_side_cut` and `c_side_cut` (P1 and P1' residue, P1's position in the UniProt entry and the enzymes that cut there, or `protein terminus`), `residue_before` and `residue_after`, `cuts_inside_after_the_pathway` and the `product_type`; `null` for a designed peptide and for runs made before it was saved), `parent_clusters[]`, `any_parent_allergen_source` (`true`, `false` or `null`), `digestion_product_type` (`FINAL_PRODUCT` when no enzyme of the pathway can cut inside the peptide, `INTERMEDIATE_PRODUCT` when one still could; `null` for a designed peptide), `scores` (`safety`, `developability`, `digestion_plausibility`, `uncut_bond_fraction`, `evidence`, `coverage`; **each is built so that higher is the favourable outcome**: `safety` is the cytotoxicity score, 1 minus the predicted probability of cytotoxicity, and `developability` is the lowest of the solubility, aggregation, synthesis-feasibility and bitterness scores that count and the sequence-liability score; `safety` comes from the safety models other than IgE binding, which is reported on its own, and `uncut_bond_fraction` does **not** measure survival in the gut), `models` (per screening model: `score`, `label`, `applicability`, `detail`; `output` is the model's own value for what it predicts, and the reports show it: higher means more likely to bind IgE (`allergenicity`; 0.5 is the model's decision boundary), more likely cytotoxic, more likely soluble, more likely aggregation-prone, more likely feasible to synthesise, more likely bitter; `score` is the ranking value, oriented so that higher is the favourable outcome (1 minus `output` for allergenicity, cytotoxicity, aggregation and bitterness; `output` for solubility and synthesis feasibility), and only the ranking uses it; the dossier's `screening.model_outputs` gives, per model, what it predicts, its `output` and what a higher value means), `applicability` (per dimension and `overall`, the worst case), `cleavage` (cut-site counts per protease, `cleavage_positions` per protease, `uncut_bond_fraction`, the rules used and a `statement` of what the fraction does not measure), `evidence` (see below), `predictions_not_run[]` (the models that did not run for this peptide, for example the aggregation model for a peptide under 5 amino acids), `glp1_classifier_status` (always `Not validated`; it never ranks), `rationale`, `limitations` and `review` (the biology lead's current decision, or `null`). **Read `applicability` before `scores`.**
- `evidence` for a peptide: `evidence_type`, one of `EXACT_SEQUENCE` (the candidate equals an evidence peptide), `CONTAINS_REFERENCE` (it contains the whole evidence peptide, 4 residues or more), `CONTAINED_IN_REFERENCE` (it is a subsequence of the evidence peptide), `SEQUENCE_SIMILARITY` (alignment score at or above the similarity cut-off, with no exact or containment match) or `NO_EVIDENCE`; `evidence_applies_to` (`CANDIDATE`); `evidence_gate_passed`; and, when there is a match, `evidence_peptide`, `evidence_direction`, `evidence_similarity`, `evidence_study`, `evidence_assay`, `evidence_food_derivation`, `evidence_tier`, plus the nearest positive and negative peptides and any `prior_art_alert`. These are **sequence relationships, not proof of GLP-1 activity.** The gate passes only for `EXACT_SEQUENCE` with a positive or mixed `evidence_direction`. A designed peptide is looked up as itself; its template's evidence is reported under `template_evidence_*` keys (`template_evidence_applies_to` is `TEMPLATE`) and is never transferred to it or scored for it.
- `shortlist[]` (the shortlisted peptides in shortlist order). A peptide is shortlisted, in rank order, when its evidence is an exact, contains or contained-in relationship to an evidence peptide with a positive or mixed effect, no model itself calls it cytotoxic, aggregation-prone, insoluble or hard to synthesise, no higher-ranked peptide of its parent protein family is already on the list, and fewer than 4 designed peptides are; the list holds up to 10. There is no cap on peptides without an exact match. `shortlist_reason` says, for every ranked peptide, why it is on the shortlist or not.
- Also in the results: `controls[]` (reference positive and negative controls proposed by rule), `gate` (the evidence gate and diversity report: `min_exact_evidence_requested`, `exact_evidence_selected_in_pass_1`, `exact_evidence_in_final`, `shortfall`, and the peptides dropped for diversity or for the gate), `facts`, `ranking_summary` (how the run narrowed its candidates, from the run's own audit log and ranking report: `stages[]` with the peptides in, out and removed per reason, `hard_rules` with what they removed, `objectives` used and not used, `pareto` with the selection rule, the tie-break order, the number of fronts and the statement that the ranking has no weights, `evidence_gate`, `caps` with the peptides each dropped; a value an older run did not record is `null`), `digestion_warnings[]` (where the digestion run again disagrees with a recorded origin of a ranked peptide: a warning only, it never fails a run), `review_gate` (the biological approval gate: `status` is `NO_SHORTLIST`, `AWAITING_REVIEW`, `IN_REVIEW` or `REVIEW_COMPLETE`; `shortlisted`, `approved[]`, `on_hold[]`, `rejected[]`, `pending[]` and the `rule` that ranks are never changed by the review).
- `protein_qc[]` (FASTA records skipped or warned about), `clusters` (`notes`, `proteins_per_source`), `library` (`mode`, the library release, and for an upload each record's match) and `provenance` (`checked`, `issues[]`; a run with an issue fails, so this is normally empty).

**Dossier `200`:** `peptide_id`, `sequence`, `length`, `rank`, `shortlist_position`, `status` (`proposed: biology lead must approve the shortlist, assay concentrations and controls` until a decision is saved, then the decision in words), `origin_type`, `evidence_class`, `summary`, `properties`, `digestion` (`observation_status` `PREDICTED` or `OBSERVED`, with `lcms_evidence` when observed, the `digestion_product_type`, and `routes[]`: one row per protein site and digestion route with the stages, enzymes and pH, the cut before and after the fragment with its flanking residues, the cuts still inside it and `product_type`; for a designed peptide the routes are those of its template fragment, over every pathway that releases it, because the pathway stored on the design's parent origin describes the edited sequence), `provenance` (parents, clusters, and for a designed peptide its template, edits and guide), `evidence` (`candidate` and, for a designed peptide, `template`, each with `applies_to`, `evidence_type`, its meaning, the evidence peptide, the reported effect, the similarity, its `source` (study, assay, tier, food derivation) and its `alignment`: the relationship (`identical`, `reference inside candidate`, `candidate inside reference` or `similar`), the aligned candidate and reference strings and the match line, both lengths, the aligned residue ranges, `identity` (identical aligned positions over aligned positions), `candidate_coverage` and `reference_coverage` (aligned residues over that sequence's length) and the `alignment_score` of the local alignment; it is for display and feeds no score, and a similarity is not experimentally confirmed GLP-1 activity; the `gate`; nearest positive and negative peptides; the prior-art alert), `ranking` (`computational_rank`, `pareto_front` and `n_pareto_fronts`, `shortlist_position` and `shortlist_reason`, the cluster and parent families, and the `objectives` values; `objectives` is `null` for a run that did not record which it used), `ranking_summary` (the run's summary, as in the results), `chemistry` (`monoisotopic_mass_da` and `average_mass_da`, with `mass_source`; the `terminal_modifications` the mass assumes (none: free termini); `predicted_liabilities[]` each with its residue `sites` and a text such as `M4: oxidation` or `whole sequence: ...`; `measured_chemical_stability` (`Not measured`); and `protease_susceptibility`, which points to the gut-protease section: the three are separate fields), `safety_screening` (predicted IgE binding as its own risk, reported as the model's own `ige_binding_score` and not as 1 minus it, the parent allergen flag, the other safety models, `sequence_liabilities[]` and `oxidation_risks[]` in words, `predictions_not_run[]`, the models left out of the safety score, `applicability_limits` and the rule that a model that did not run is never a pass), `screening`, `gut_stability` (`uncut_bond_fraction`, its `statement`, cut-site counts and `predicted_cleavage_positions` per protease), `comparison_materials` (the parent proteins, the hydrolysates those proteins and the run's digestion pathways define, and reference peptides: the template, the guide, the closest evidence peptide, the nearest positive and negative peptides and the run's controls), `limitations`, `not_established[]`, `controls[]`, `assay_plan` and `review` (`decision`, `reason`, `reviewer`, `decided_at`, `allowed_decisions`). The values the biology lead chooses (`assay_plan.set_by_biology_lead.*` and, until they decide, the `review` fields) are `null`. Nothing is estimated for them. A list of required experiments per candidate is not part of the dossier yet.

**Dossier PDF (`application/pdf`, `Content-Disposition: attachment`):** written for the biology lead, in plain words: labelled values (no field names or raw codes), digestion routes spelled out, and edits of a designed peptide as residue, position and change. The first page is an overview of the shortlisted peptides, one line each in shortlist order (the same peptides as the experiment sheet) with origin, evidence, applicability and scores, and explains how to read the peptide pages. The second page holds what is the same for every peptide: the reference controls, what is not established, the empty assay-plan cells, how to read the digestion and alignment blocks, and the limitations every peptide shares. When the run recorded how it ranked, a third page, "How the list was made", shows the candidate pool at each step with what each hard rule removed, the evidence gate, the objectives, the selection and tie-break rules and the caps with the peptides they dropped. Then comes one A4 landscape page per shortlisted peptide with only what is specific to it: why it ranks, properties, where it comes from, how a designed peptide was made (template, edits, reference peptide), evidence, predicted gut cuts, every screening model with result, score and applicability, predicted IgE binding as its own risk, oxidation risks and other sequence liabilities, the models that did not run, the evidence type with its source (and, for a designed peptide, its template's evidence marked as not transferred), the predicted cut positions of each gut protease with the uncut-bond fraction and what it does not measure, the materials to compare with (parent proteins, parent hydrolysates, reference peptides), any limitation not on the shared page, and the review: the saved decision with its reason, reviewer and time, or empty cells. The shortlist PDF therefore has two pages more than it has peptides, and the one-peptide PDF has three pages. The overview page shows each peptide's review status. The page leaves out technical detail that the JSON dossier keeps (random seed, generator version, database checksum, template id, model methods, Pareto and cluster numbers) and the "; uncalibrated" note that ends each model's detail text in the JSON and HTML dossiers. A page that holds more than fits is set in smaller type, down to 70% of the normal size. A peptide with more than 10 parent sites shows the first 10 and says how many are left out. The text uses the PDF standard font, so a character outside Windows-1252 prints as `?`.

**Experiment sheet (CSV, `Content-Disposition: attachment`):** one row per shortlisted peptide in shortlist order (`row_type` `candidate`), then one per reference control (`positive_control`, `negative_control`). Columns: `row_type`, `shortlist_position`, `rank`, `peptide_id`, `sequence`, `length`, `origin_type`, `parent_proteins`, `food_source`, `parent_allergen_flag`, `predicted_ige_binding`, `oxidation_risks`, `evidence_type`, `evidence_gate_passed`, `evidence_peptide`, `evidence_study`, `template_evidence_type`, `applicability_overall`, `predictions_not_run`, `safety_flagged_models`, `prior_art_alert`, then the cells for the biology lead, left empty: `glp1_endpoint`, `concentration_range`, `exposure_time`, `cell_line`, `vehicle`, `viability_control`, `interference_control`, `biological_replicates`, `technical_replicates`, and finally `status`, `decision`, `reviewer`, `comments`. `decision`, `reviewer` and `comments` (the reason) are filled from the decisions saved through the review route and are empty until the biology lead decides.

**Digestion `200`:** `evidence_class` (`THEORETICAL`), `settings` (the pathways and limits used), `settings_warnings[]`, `rule_validation` (what the protease benchmark says about each rule used), `n_proteins`, `n_peptides` (unique, over all proteins and pathways), `n_origins`, `offset`, `limit`, `truncated`, `peptides[]` (`peptide_id`, `sequence`, `length`, `mono_mass`, `avg_mass`, `pathways[]`, `n_origins`, `parents[]`, `is_limit_product`, `chemistry_notes`, `origins[]` with parent, pathway, stage, enzymes, pH, positions in the product and the precursor, missed cleavages, termini), `cleavage_map[]` (per protein and pathway: `length` and `sites[]` with `p1_position`, `p1_residue`, `p1p_residue`, `cleaved_by[]`) and `limitations[]`. The request fields are `fasta` or `library_proteins` (exactly one), and optionally `pathways[]` (ids from the settings route), `max_missed_cleavages` (0 to 4), `min_length`, `max_length` (1 to 60), `limit` (1 to 2000, default 500) and `offset`. A rule cannot be supplied.

**Library:** `GET /api/v1/food/library/summary` returns `manifest` (`name`, `uniprot_release`, `retrieved_at`, `note`), `total`, `by_food_source`, `families[]` (`food_source`, `family`, `count`) and `allergen_source_flags` (`allergen_source`, `not_flagged`, `not_recorded`). `GET /api/v1/food/library/proteins` returns `total`, `offset`, `limit` and `proteins[]` (`substrate_id`, `accession`, `name`, `food_source`, `kind`, `length`, `protein_family`, `organism`, `allergen_source`, `sequence_version`, `uniprot_release`, `retrieved_at`, `n_terminus`, `c_terminus`). `n_terminus` and `c_terminus` say what is at each end of the chain, from the FASTA header (`NTERM=`, `CTERM=`): `SIGNAL_CLEAVED`, `PROPEP_CLEAVED`, `TRANSIT_CLEAVED` or `INIT_MET_CLEAVED` (that end is where a signal peptide, propeptide, transit peptide or starting methionine was cut off), `CHAIN_JUNCTION` (that end meets another chain directly, nothing removed between them), `SEQ_START` or `SEQ_END` (first or last residue of the UniProt sequence, so the protein's real end may lie beyond it) and `UNANNOTATED` (UniProt has no annotation for that end). `null` when the input did not say (an uploaded FASTA without those tags).

### Food-peptide errors

| HTTP status | Meaning |
|---|---|
| `404` | Unknown job; the job has not succeeded (results, dossier, PDF and sheet); the run has no shortlisted peptide (shortlist PDF); the peptide is not among the ranked peptides; success was reported but the ranked file is missing |
| `422` | Invalid request (neither or both of `fasta` and `library_proteins`; `food_source` missing with a FASTA, or sent with library proteins); no usable FASTA record; an unknown library protein or pathway; a digestion run over 50 proteins or 30,000 residues; a malformed job id |
| `502` | Google API failure, or the run config could not be written (the job is cancelled) |
| `503` | Missing server settings; the model store is the placeholder, lacks a model weight, or is not a `gs://` path; Google credentials unavailable |

### Food-peptide run folder

Under `result_path`: `config.json` (the worker input), `proteins.fasta`, `source_records.csv` (only when the library supplied records), `library_match.json`, `protein_qc.json`, `protein_clusters.json`, `candidates/s03_food_digestion.jsonl` to `candidates/s07_food_ranking_decision.jsonl` (the ranked peptides), `ranking_report.json` (gate, facts, controls), `provenance_check.json`, `audit_log.jsonl`, `candidates_final.json`, `config_snapshot.yaml`, `stats_<job id>.txt`, `results.json` (worker status) and, in local mode, `worker.log`.

## Immunomodulation studio

A separate, much smaller job: it scores a list of peptides on six cytokine-induction models and runs no pipeline stages. It uses the same status, results-by-job-id and cancel mechanics, under its own prefix.

| Method and path | Purpose |
|---|---|
| `POST /api/v1/jobs/immunomodulation/create` | Submit peptides |
| `GET /api/v1/jobs/immunomodulation/{job_id}/status` | Same envelope as `/status`; `stage` carries batch progress |
| `GET /api/v1/jobs/immunomodulation/{job_id}/results` | Per-peptide predictions |
| `POST /api/v1/jobs/immunomodulation/{job_id}/cancel` | Same as `/cancel` |
| `GET /api/v1/immunomodulation/models` | The six models: version, checksum, cross-validation figures, trained lengths, per-length applicability ranges and similarity cut-offs |

Request: `{"request_id": "...", "peptides": ["ALPMHIR", ">name
AMKPWIQPK", ...]}`, 1 to 10000 entries. Each is checked for standard residues, length 6-50, and duplicates; `422` if none survive. `request_id` names the run; the API accepts any non-empty string but uses it as a storage path segment, so stick to letters, digits, `.`, `-` and `_`. A repeat of the same `request_id` returns the first job instead of starting another, so a client must give every run its own.

**Batches.** The worker splits the accepted peptides into batches of `IMMUNO_BATCH_SIZE` (default 100), loads the models once, and scores one batch at a time through all six models. Results come back in input order. After each batch the status route's `stage` reads, for example, `batch 3 of 12 (300/1200 peptides)`. A failure in any batch fails the whole run; there are no partial results.

Response `202`: the usual `request_id`, `job_id`, `result_path`, plus `accepted` (peptides submitted) and `rejected` (each with `id`, `sequence`, `reason` of `invalid_residues`, `too_short`, `too_long`, `duplicate` or `empty`, and `detail`).

Results: `{pipeline: "immunomodulation", job_id, run_id, status, n_final, results: [{sequence, predictions}]}`. `predictions` is keyed `TNF-alpha`, `IL-6`, `IL-10`, `IL-4`, `IL-13`, `IFN-gamma`; each holds:

| Field | Meaning |
|---|---|
| `probability` | The model's estimate that the peptide induces the cytokine in the training assays' context. Not calibrated and not comparable across cytokines |
| `label` | `1` when `probability >= 0.5` |
| `in_length_range` | Whether the length is inside the model's trained range (kept for compatibility; `applicability` supersedes it) |
| `model` | The `model_store` package that produced the value |
| `applicability` | `{status, reasons, checks, reference, not_checked}`, below |

**Read `applicability` before trusting a probability.** `status` is `in_domain`, `limited_evidence` or `out_of_domain`. It comes from two checks: the peptide's length against the model's trained range (`checks.length_range`), and the number of positive and negative training peptides at exactly that length (`checks.reference_support`; `reference` gives the counts and the minimum of 10 per class). `reasons` explains the status in words. A third check answers *is the peptide similar to what the model was trained on?*: `similarity` is `{best_identity, verdict, similar_at, distant_below}` and `checks.sequence_similarity` is `pass`, `weak` or `fail`. `best_identity` (0-1) is the highest BLOSUM62 alignment identity, over the peptide's length, between it and a shortlist of the model's training peptides (the 30 closest by 2-mer composition); `verdict` is `similar` at or above `similar_at`, `distant` below `distant_below`, else `moderate`. The two cut-offs are per model, taken from how far held-out training peptides were from the other cross-validation folds (10th and 50th percentiles). A `distant` peptide is capped at `limited_evidence`; a similarity result never raises a status. The training peptides themselves are never returned. `not_checked` lists what was not assessed: `embedding_distribution` always, and `sequence_similarity` for weights without a similarity profile, so `in_domain` is not a full domain check. Trained length ranges: IFN-gamma 8-20, IL-6 8-25, IL-13 8-35, IL-10 8-46, IL-4 7-45, TNF-alpha 7-50.

Build details of the models are not returned anywhere: the results carry one probability per cytokine, and the registry and `/api/v1/models` serve the model cards without them.

**Local mode.** With `DEV_MODE=true` the job runs as a local process on the weights in `model_store/model_weights/`, with no Vertex job, bucket or image. Job ids then start with `projects/local/locations/local/customJobs/`; the process handle lives in the API process, so after an API restart a running job cannot be cancelled. In this mode the API also answers requests from any origin (CORS `*`), so a standalone page can call it; never set it in production. Otherwise the job runs on a Vertex Custom Job on `WORKER_IMMUNO_IMAGE_URI` and downloads the weights from `VERTEX_MODEL_STORE`.

Needs `WORKER_IMMUNO_IMAGE_URI` (503 without it, unless `DEV_MODE`) and the shared settings; no seed file. `IMMUNO_BATCH_SIZE` is optional.

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
| `immunomodulation` (own endpoint, see below) | `WORKER_IMMUNO_IMAGE_URI` | none: the peptides are in the request |
| `food` (own endpoint, see below) | `WORKER_FOOD_IMAGE_URI` | none: the proteins are in the request (the optional food library is read by the API from `FOOD_LIBRARY_DIR`: a local folder under `DEV_MODE`, a `gs://` folder in the cloud holding `digestion_substrates.fasta`, `source_records.csv` and `MANIFEST.json`, downloaded once at first use; empty means the copy bundled in the API image; the worker never reads it) |

`IMMUNO_BATCH_SIZE` (optional, default 100) sets how many peptides the immunomodulation worker scores at a time.

Each seed file is a `gs://` FASTA. Seeds are a server setting, so each pipeline's
seed set is fixed by the operator and cannot be changed per request.
Use the project ID or number, not the display name. Artifact/model/seed paths
must be GCS URIs. The service-account setting is an email, not a key file.

`MACHINE_TYPE`/`ACCELERATOR_TYPE`/`ACCELERATOR_COUNT` are set explicitly in
`.env` to `n1-standard-8`/`NVIDIA_TESLA_T4`/`1`.

The OpenAPI file is a standalone contract for these four operations. It does not
change the service's routes or its automatically generated `/openapi.json`.

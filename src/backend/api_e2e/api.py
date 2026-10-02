"""Start and inspect complete-pipeline Vertex AI Custom Jobs."""

import json
import os
import re
import statistics
import logging
from functools import lru_cache
from typing import Literal
from fastapi import FastAPI, HTTPException
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import GoogleAuthError
from pydantic import BaseModel, ConfigDict, Field, config
from common import env, storage
from schemas.brief import Brief
from schemas.e2e_config import validate_job_config
from schemas.stage_configs import E2ERequest

app = FastAPI(title="End-to-end Pipeline Job API")


def required(name):
    value = os.environ.get(name, "")
    if not value:
        raise HTTPException(503, f"Server setting {name} is required")
    return value


@lru_cache
def job_client():
    from google.cloud import aiplatform_v1

    return aiplatform_v1.JobServiceClient(
        client_options={
            "api_endpoint": f"{required('VERTEX_LOCATION')}-aiplatform.googleapis.com"
        }
    )


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    stages: E2ERequest


class CreateJobResponse(BaseModel):
    request_id: str
    job_id: str
    result_path: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = "pending"
    vertex_state: str
    error: str | None = None


def _label_value(value: str) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-_")
    return (label or "request")[:63]


def job_name(job_id):
    if not re.fullmatch(r"projects/[^/]+/locations/[^/]+/customJobs/[0-9]+", job_id):
        raise HTTPException(
            422, "job_id must be the full Vertex Custom Job resource name"
        )
    return job_id


@app.post("/api/v1/jobs/create", response_model=CreateJobResponse, status_code=202)
def create_job(request: CreateJobRequest):
    parent = f"projects/{required('VERTEX_CLOUD_PROJECT')}/locations/{required('VERTEX_LOCATION')}"
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    models = required("VERTEX_MODEL_STORE")
    seeds = required("SEED_CANDIDATES_FILE")
    image = required("WORKER_IMAGE_URI")
    service_account = required("WORKER_SERVICE_ACCOUNT")
    machine_type = required("MACHINE_TYPE")
    accelerator_type = required("ACCELERATOR_TYPE")
    accelerator_count = int(required("ACCELERATOR_COUNT"))
    if not all(storage.is_gcs_path(p) for p in (artifacts, models, seeds)):
        raise HTTPException(
            503, "Artifacts, model store, and seed FASTA must use gs:// paths"
        )
    # request_key = hashlib.sha256(request.request_id.encode()).hexdigest()
    staging_dir = storage.join(artifacts, "requests", request.request_id)
    staging_config_path = storage.join(staging_dir, "config.json")
    try:
        config = validate_job_config(
            {
                "artifacts_dir": artifacts,
                "model_store": models,
                "seed_candidates_path": seeds,
                "stages": request.stages.model_dump(),
            }
        )
        Brief.model_validate(
            config.for_stage("s01_therapeutic_product_brief").params["brief"]
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    custom_job = {
        "display_name": f"ai-peptide-discovery-e2e-{request.request_id[:12]}",
        "labels": {
            "service": "battery-ai",
            "request-id": _label_value(request.request_id),
        },
        "job_spec": {
            "service_account": service_account,
            "worker_pool_specs": [
                {
                    "replica_count": 1,
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                    "container_spec": {
                        "image_uri": image,
                        "args": [
                            "--config",
                            staging_config_path,
                            "--wait-for-config",
                            "120",
                        ],
                        "env": [
                            {"name": "DEV_MODE", "value": "false"},
                            *(
                                {"name": key, "value": os.environ[key]}
                                for key in env.DOTENV_KEYS
                                if key != "DEV_MODE" and key in os.environ
                            ),
                        ],
                    },
                }
            ],
        },
    }
    try:
        client = job_client()
        job = client.create_custom_job(parent=parent, custom_job=custom_job)
        job_id = job.name.rsplit("/", 1)[-1]
        config.run_id = job_id
        result_path = storage.join(artifacts, "runs", job_id)
        config_path = storage.join(result_path, "config.json")
        try:
            body = config.model_dump_json(indent=2)
            storage.write_text(config_path, body)
            storage.write_text(staging_config_path, body)
        except Exception as exc:
            try:
                client.cancel_custom_job(name=job.name, timeout=30)
            except Exception:
                logging.exception(
                    "Could not cancel job %s after config upload failed", job.name
                )
            raise HTTPException(
                502, f"Config upload failed for {job.name}; cancellation requested"
            ) from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return CreateJobResponse(
        request_id=request.request_id,
        job_id=job.name,
        result_path=result_path,
    )


@app.get("/api/v1/jobs/{job_id:path}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    name = job_name(job_id)
    try:
        job = job_client().get_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    vertex_state = job.state.name

    artifacts = required("VERTEX_ARTIFACTS_DIR")
    run_id = _run_id_from_job_id(job_id)
    run_dir = storage.join(artifacts, "runs", run_id)
    status_path = storage.join(run_dir, "results.json")

    results = (
        json.loads(storage.read_text(status_path))
        if storage.exists(status_path)
        else None
    )

    return JobStatusResponse(
        job_id=job.name,
        status=(results.get("status") if results else None) or "pending",
        stage=(results.get("stage") if results else None) or "pending",
        vertex_state=vertex_state,
        error=job.error.message or None,
    )


class ComponentStats(BaseModel):
    count: int
    mean: float
    median: float
    min: float
    max: float


class CandidateResponse(BaseModel):
    id: str = "?"
    sequence: str | None = None  # 'sequence' key - sibling of predictions
    ranking: int | None = None  # 'ranking.rank' key in predictions
    final_score: float | None = None  # 'ranking.final_score' key in predictions
    evidence_coverage: float | None = None  # 'ranking.evidence_coverage' key
    missing_modules: list[str] | None = None  # 'ranking.missing_modules' key
    amp_probability: float | None = None  # 'amp_prob' key in predictions
    hemolytic_activity_phc50: float | None = (
        None  # 'hemolysis_phc50' key in predictions
    )
    molecular_weight: float | None = (
        None  # 'molecular_weight' key in predictions (unit: Daltons)
    )
    net_charge: float | None = None  # 'net_charge' key in predictions
    instability_index: float | None = None  # 'instability_index' key in predictions
    deamidation_risk: Literal["low", "medium", "high"] | None = (
        None  # deamidation_risk.risk_category
    )
    oxidation_risk: Literal["low", "medium", "high"] | None = (
        None  # oxidation_risk.risk_category
    )
    solubility: float | None = None  # 'solubility.score' key in predictions
    aggregation_tendency: float | None = (
        None  # 'aggregation_tendency.score' key in predictions
    )
    anti_inflammatory_probability: float | None = (
        None  # 'anti_inflammatory_prob' key in predictions
    )
    angiogenic_activity: float | None = (
        None  # 'angiogenic_activity.angiogenic' key in predictions
    )
    proliferation_probability: float | None = (
        0  # proliferation_migration.proliferation in predictions
    )
    migration_probability: float | None = (
        0  # proliferation_migration.migration in predictions
    )
    cytotoxicity_probability: float | None = (
        None  # 'cytotoxicity.score' key in predictions
    )
    cleavage_stability: float | None = None  # cleavage_stability.score in predictions
    log_mic_um: dict[
        Literal["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"],
        float,
    ]  # 'mic.log_mic_um' key in predictions
    pmbic: dict[
        Literal[
            "Acinetobacter baumannii",
            "Candida albicans",
            "Candida tropicalis",
            "Cutibacterium acnes",
            "Enterococcus faecium",
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Pseudomonas aeruginosa",
            "Salmonella enterica",
            "Staphylococcus aureus",
            "Staphylococcus epidermidis",
            "Streptococcus mutans",
            "Streptococcus sanguinis",
        ],
        float,
    ]
    engaged_pathways: list[str] | None = (
        None  # 'mechanism.engaged_pathways' key in predictions
    )


class JobResultsResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
    candidates: list[CandidateResponse] = []


def _run_id_from_job_id(job_id: str) -> str:
    return job_id.rsplit("/", 1)[-1]


def _candidate_response(candidate: dict) -> CandidateResponse:
    preds = candidate.get("predictions") or {}

    def nested(key: str, field: str):
        return (preds.get(key) or {}).get(field)

    return CandidateResponse(
        id=candidate.get("id", "?"),
        sequence=candidate.get("sequence"),
        ranking=nested("ranking", "rank"),
        final_score=nested("ranking", "final_score"),
        evidence_coverage=nested("ranking", "evidence_coverage"),
        missing_modules=nested("ranking", "missing_modules"),
        amp_probability=preds.get("amp_probability"),
        hemolytic_activity_phc50=nested("hemolysis", "phc50"),
        molecular_weight=preds.get("molecular_weight"),
        net_charge=preds.get("net_charge"),
        instability_index=preds.get("instability_index"),
        deamidation_risk=nested("deamidation_risk", "risk_category"),
        oxidation_risk=nested("oxidation_risk", "risk_category"),
        solubility=nested("solubility", "score"),
        aggregation_tendency=nested("aggregation_tendency", "score"),
        anti_inflammatory_probability=preds.get("anti_inflammatory_probability"),
        angiogenic_activity=nested("angiogenic_activity", "angiogenic"),
        proliferation_probability=nested("proliferation_migration", "proliferation"),
        migration_probability=nested("proliferation_migration", "migration"),
        cytotoxicity_probability=nested("cytotoxicity", "score"),
        cleavage_stability=nested("cleavage_stability", "score"),
        log_mic_um=nested("mic", "log_mic_um") or {},
        pmbic=nested("mbic", "pmbic") or {},
        engaged_pathways=nested("mechanism", "engaged_pathways"),
    )


def _component_stats(candidates: list[dict]) -> dict[str, ComponentStats]:
    """Mean/median/min/max per ranking module over ranked candidates' module
    `score` (after flag deductions) -- insufficient_evidence candidates and
    modules without a score are excluded rather than treated as zero."""
    values_by_component: dict[str, list[float]] = {}
    for candidate in candidates:
        ranking = candidate.get("predictions", {}).get("ranking")
        if not ranking or ranking.get("status") != "ranked":
            continue
        for component, module in (ranking.get("modules") or {}).items():
            value = module.get("score")
            if value is not None:
                values_by_component.setdefault(component, []).append(value)

    return {
        component: ComponentStats(
            count=len(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        for component, values in values_by_component.items()
    }


@app.get("/api/v1/jobs/{job_id:path}/results", response_model=JobResultsResponse)
def get_job_results(job_id: str):
    name = job_name(job_id)
    run_id = _run_id_from_job_id(name)
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    run_dir = storage.join(artifacts, "runs", run_id)
    print("Fetching results for job %s in %s", name, run_dir)
    status_path = storage.join(run_dir, "results.json")
    if not storage.exists(status_path):
        raise HTTPException(404, "Job results not found")
    status_payload = json.loads(storage.read_text(status_path))
    status = status_payload.get("status", "unknown")

    if status != "success":
        return JobResultsResponse(
            job_id=name,
            run_id=run_id,
            status=status,
            n_final=status_payload.get("n_final"),
            candidates=[],
        )

    candidates_path = storage.join(run_dir, "candidates_final.json")
    if not storage.exists(candidates_path):
        raise HTTPException(
            404, "Job reported success but candidates_final.json is missing"
        )
    candidates = json.loads(storage.read_text(candidates_path))

    ranked = sum(
        1
        for c in candidates
        if c.get("predictions", {}).get("ranking", {}).get("status") == "ranked"
    )

    candidates_response = [_candidate_response(c) for c in candidates]

    return JobResultsResponse(
        job_id=name,
        run_id=run_id,
        status=status,
        n_final=status_payload.get("n_final"),
        ranked_candidates=ranked,
        insufficient_evidence_candidates=len(candidates) - ranked,
        component_stats=_component_stats(candidates),
        candidates=candidates_response,
    )


class CancelJobResponse(BaseModel):
    job_id: str
    status: Literal["cancelling", "cancelled", "failed"]


_ALREADY_CANCELLED_STATES = {"JOB_STATE_CANCELLING", "JOB_STATE_CANCELLED"}
_TERMINAL_FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_EXPIRED"}


@app.post("/api/v1/jobs/{job_id:path}/cancel", response_model=CancelJobResponse)
def cancel_job(job_id: str):
    name = job_name(job_id)
    client = job_client()
    try:
        job = client.get_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc

    state = job.state.name
    if state in _ALREADY_CANCELLED_STATES:
        return {"job_id": name, "status": "cancelled"}
    if state in _TERMINAL_FAILED_STATES:
        return {"job_id": name, "status": "failed"}

    try:
        client.cancel_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"job_id": name, "status": "cancelling"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("PORT", 8080)), log_level="info"
    )

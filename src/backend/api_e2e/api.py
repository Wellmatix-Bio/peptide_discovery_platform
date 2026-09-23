"""Start and inspect complete-pipeline Vertex AI Custom Jobs."""

import os
import re
import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import GoogleAuthError
from pydantic import BaseModel, ConfigDict, Field
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
    use_feature_cache: bool = True


class CreateJobResponse(BaseModel):
    request_id: str
    job_id: str
    config_path: str
    result_path: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    vertex_state: str
    error: str | None = None


KNOWN_STAGES = {
    "s01_therapeutic_product_brief",
    "s02_wound_biology_and_targets",
    "s03_data_integration",
    "s04_candidate_generation",
    "s05_physchem_screening",
    "s06_functional_models",
    "s07_structure_mechanism",
    "s08_safety_developability",
    "s09_synthesis_cmc",
    "s11_ranking",
}


def _label_value(value: str) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-_")
    return (label or "request")[:63]


@app.post("/api/v1/jobs/create", response_model=CreateJobResponse, status_code=202)
def create_job(request: CreateJobRequest):
    parent = f"projects/{required('VERTEX_CLOUD_PROJECT')}/locations/{required('VERTEX_LOCATION')}"
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    models = required("VERTEX_MODEL_STORE")
    seeds = required("SEED_CANDIDATES_FILE")
    image = required("WORKER_IMAGE_URI")
    service_account = required("WORKER_SERVICE_ACCOUNT")
    if not all(storage.is_gcs_path(p) for p in (artifacts, models, seeds)):
        raise HTTPException(
            503, "Artifacts, model store, and seed FASTA must use gs:// paths"
        )
    # request_key = hashlib.sha256(request.request_id.encode()).hexdigest()
    staging_dir = storage.join(artifacts, "requests", request.request_id)
    staging_config_path = storage.join(staging_dir, "config.json")
    try:
        unknown = set(request.stages) - KNOWN_STAGES
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        config = validate_job_config(
            {
                "run_id": "pending",
                "artifacts_dir": artifacts,
                "model_store": models,
                "seed_candidates_path": seeds,
                "use_feature_cache": request.use_feature_cache,
                "stages": request.stages,
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
                        "machine_type": os.environ.get(
                            "MACHINE_TYPE", "n1-standard-4"
                        ).strip(),
                        "accelerator_type": "NVIDIA_TESLA_T4",
                        "accelerator_count": 1,
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
        config_path=config_path,
        result_path=result_path,
    )


def job_name(job_id):
    if not re.fullmatch(r"projects/[^/]+/locations/[^/]+/customJobs/[0-9]+", job_id):
        raise HTTPException(
            422, "job_id must be the full Vertex Custom Job resource name"
        )
    return job_id


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
    state = job.state.name
    status = {
        "JOB_STATE_RUNNING": "running",
        "JOB_STATE_SUCCEEDED": "success",
        "JOB_STATE_FAILED": "fail",
        "JOB_STATE_EXPIRED": "fail",
        "JOB_STATE_PARTIALLY_SUCCEEDED": "fail",
        "JOB_STATE_CANCELLING": "stopped",
        "JOB_STATE_CANCELLED": "stopped",
        "JOB_STATE_PAUSED": "stopped",
    }.get(state, "pending")
    return JobStatusResponse(
        job_id=job.name,
        status=status,
        vertex_state=state,
        error=job.error.message or None,
    )


@app.post("/api/v1/jobs/{job_id:path}/cancel")
def cancel_job(job_id: str):
    name = job_name(job_id)
    try:
        job_client().cancel_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"job_id": name, "status": "cancelling"}

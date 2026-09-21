# Vertex AI job-management API (stub). Separate from main.py, which stays
# the pipeline CLI entry point. Run with: uvicorn api:app --reload
#
# TODO: wire up real project/location/service-account config (see
# aiplatform.init() below) and real submission logic before this is usable.
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import GoogleAuthError
from google.cloud import aiplatform
from pydantic import BaseModel

app = FastAPI(title="Vertex AI Job API")

# TODO: pull real project/location from config/env and call aiplatform.init()
# here (or in a startup hook) before any endpoint below will work. Left
# uncalled so this stub module can be imported without live GCP config:
#
#   aiplatform.init(project="my-project", location="us-central1")


class PredictRequest(BaseModel):
    """TODO: real prediction request payload (model inputs, etc.)."""


class PredictResponse(BaseModel):
    request_id: str
    job_name: str
    status: str


class JobStatusResponse(BaseModel):
    job_name: str
    state: str


class JobCancelResponse(BaseModel):
    job_name: str
    status: str


@app.post("/api/v1/predict", response_model=PredictResponse)
def submit_prediction_job(request: PredictRequest) -> PredictResponse:
    """Submit work to Vertex AI and return immediately with the request ID
    and Vertex job name; does not block on job completion."""
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    # TODO: build the actual CustomJob (worker pool specs, container image,
    # args derived from `request`) and submit it non-blockingly, e.g.:
    #
    #   job = aiplatform.CustomJob(
    #       display_name=request_id,
    #       worker_pool_specs=[...],
    #   )
    #   job.submit()  # non-blocking; job.run() would block until completion
    #   job_name = job.resource_name
    #
    # Stubbed for now:
    job_name = (
        f"projects/TODO-project/locations/TODO-region/customJobs/TODO-{request_id}"
    )

    return PredictResponse(request_id=request_id, job_name=job_name, status="submitted")


@app.get("/api/v1/jobs/{job_name:path}/status", response_model=JobStatusResponse)
def get_job_status(job_name: str) -> JobStatusResponse:
    """Query Vertex directly via the SDK for the job's current state --
    never infer status from GCS file presence."""
    try:
        # TODO: aiplatform.CustomJob.get() requires a bare resource name or
        # display name depending on SDK version/config; confirm the exact
        # form job_name arrives in from the client before wiring this up.
        job = aiplatform.CustomJob.get(resource_name=job_name)
    except NotFound:
        raise HTTPException(status_code=404, detail=f"job not found: {job_name}")
    except GoogleAuthError as exc:
        raise HTTPException(status_code=500, detail=f"Vertex AI not configured: {exc}")
    except GoogleAPICallError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # job.state is a google.cloud.aiplatform_v1.types.JobState enum, e.g.
    # JOB_STATE_QUEUED, JOB_STATE_PENDING, JOB_STATE_RUNNING,
    # JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED, ...
    return JobStatusResponse(job_name=job_name, state=job.state.name)


@app.post("/api/v1/jobs/{job_name:path}/cancel", response_model=JobCancelResponse)
def cancel_job(job_name: str) -> JobCancelResponse:
    """Call Vertex's cancellation API. Cancellation is asynchronous -- the
    client should poll the status endpoint afterward to see whether the job
    reached JOB_STATE_CANCELLED."""
    try:
        job = aiplatform.CustomJob.get(resource_name=job_name)
        job.cancel()
    except NotFound:
        raise HTTPException(status_code=404, detail=f"job not found: {job_name}")
    except GoogleAuthError as exc:
        raise HTTPException(status_code=500, detail=f"Vertex AI not configured: {exc}")
    except GoogleAPICallError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return JobCancelResponse(job_name=job_name, status="cancel_requested")

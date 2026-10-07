import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound, ServiceUnavailable
from backend.api_e2e import api
from backend.worker_e2e.worker import load_job_config
from schemas.stage_configs import resolve_params

JOB = "projects/123/locations/us-central1/customJobs/456"


@pytest.fixture
def service(monkeypatch):
    settings = {
        "VERTEX_CLOUD_PROJECT": "123",
        "VERTEX_LOCATION": "us-central1",
        "VERTEX_ARTIFACTS_DIR": "gs://test/artifacts",
        "VERTEX_MODEL_STORE": "gs://test/models",
        "SEED_CANDIDATES_FILE": "gs://test/seeds.fasta",
        "WORKER_IMAGE_URI": "us-central1-docker.pkg.dev/test/images/worker:latest",
        "WORKER_SERVICE_ACCOUNT": "worker@test.iam.gserviceaccount.com",
        # api.py reads all three through required(); there is no code fallback,
        # so omitting them makes every create/status call a 503.
        "MACHINE_TYPE": "g2-standard-4",
        "ACCELERATOR_TYPE": "NVIDIA_L4",
        "ACCELERATOR_COUNT": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    # NOTE: api.py forwards to the worker container only the keys common/env.py read
    # from a .env FILE (env.DOTENV_KEYS), not the process environment. A test
    # checkout has no .env, so only DEV_MODE is forwarded and the container_spec
    # env assertion in test_create_submits_worker_readable_json fails. Left failing
    # deliberately: see docs/BASELINE.md, "Open questions for the backend owner".
    objects, calls = {}, []
    job = SimpleNamespace(
        name=JOB,
        state=SimpleNamespace(name="JOB_STATE_RUNNING"),
        error=SimpleNamespace(message=""),
    )

    def create(**kwargs):
        path = kwargs["custom_job"]["job_spec"]["worker_pool_specs"][0][
            "container_spec"
        ]["args"][1]
        if not calls:
            assert path not in objects  # First config is published after submission.
        calls.append(kwargs)
        return job

    sdk = SimpleNamespace(
        create_custom_job=create,
        get_custom_job=lambda **kw: job,
        cancel_custom_job=lambda **kw: calls.append(kw),
    )
    monkeypatch.setattr(api, "job_client", lambda: sdk)

    monkeypatch.setattr(
        api.storage, "write_text", lambda path, data: objects.__setitem__(path, data)
    )
    monkeypatch.setattr(api.storage, "read_text", lambda path: objects[path])
    # get_job_status() calls exists() before read_text(); unmocked it reaches real GCS.
    monkeypatch.setattr(api.storage, "exists", lambda path: path in objects)
    example = (
        Path(__file__).resolve().parents[2] / "src/backend/api_e2e/example_request.json"
    )
    payload = json.loads(example.read_text())
    return TestClient(api.app), sdk, job, objects, calls, payload


def test_create_submits_worker_readable_json(service):
    client, sdk, job, objects, calls, payload = service
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["job_id"] == JOB
    # CreateJobResponse carries result_path, not config_path; api.py writes the
    # run config to <result_path>/config.json.
    config_path = result["result_path"] + "/config.json"
    config = load_job_config(config_path)
    assert config.run_id == "456"
    assert "run_id" not in result
    assert result["result_path"] == "gs://test/artifacts/runs/456"
    assert not config.for_stage("s02_wound_biology_and_targets").enabled
    for name, params in payload["stages"].items():
        assert config.for_stage(name).enabled
        assert config.for_stage(name).params == resolve_params(name, params)
    pool = calls[0]["custom_job"]["job_spec"]["worker_pool_specs"][0]
    args = pool["container_spec"]["args"]
    assert args[0] == "--config"
    assert args[2:] == ["--wait-for-config", "120"]
    assert load_job_config(args[1]).run_id == "456"
    assert objects[args[1]] == objects[config_path]
    assert {e["name"]: e["value"] for e in pool["container_spec"]["env"]} == {
        "DEV_MODE": "false",
        "VERTEX_MODEL_STORE": config.model_store,
    }


@pytest.mark.parametrize(
    "state,status",
    [
        ("JOB_STATE_QUEUED", "pending"),
        ("JOB_STATE_RUNNING", "running"),
        ("JOB_STATE_SUCCEEDED", "success"),
        ("JOB_STATE_FAILED", "fail"),
        ("JOB_STATE_CANCELLED", "stopped"),
    ],
)
def test_status_reads_worker_results_and_vertex_state(service, state, status):
    client, sdk, job, objects, *_ = service
    job.state.name = state
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps(
        {"status": status, "stage": "s06_functional_models"}
    )
    response = client.get(f"/api/v1/jobs/{JOB}/status")
    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["stage"] == "s06_functional_models"
    assert response.json()["vertex_state"] == state


def test_status_is_pending_before_worker_writes_results(service):
    client, *_ = service
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["status"] == "pending" and body["stage"] == "pending"


@pytest.mark.parametrize(
    "state,expected",
    [
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("JOB_STATE_EXPIRED", "failed"),
        ("JOB_STATE_SUCCEEDED", "abandoned"),
    ],
)
def test_a_dead_worker_is_reported_as_over_not_as_pending(service, state, expected):
    """The defect this endpoint used to have, now fixed and pinned.

    A worker killed before writing results.json leaves no file, so `status` reads "pending" --
    indefinitely, for a job Vertex has already given up on. That field still says "pending",
    because it is documented as the worker's own last word and a stale self-report is not a lie
    about the worker. `run_state` is the reconciled answer, and it says the run is over.

    JOB_STATE_SUCCEEDED with no results is `abandoned`: the container exited cleanly and produced
    nothing. Neither "pending" nor "succeeded" would be true of it.
    """
    client, sdk, job, *_ = service
    job.state.name = state
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["vertex_state"] == state
    assert body["status"] == "pending", "status remains the worker's own word"
    assert body["run_state"] == expected


def test_a_worker_that_finished_outranks_the_vertex_state(service):
    """A first-hand report from the worker is more specific than the job's exit. A results.json
    saying success stays `succeeded` even though Vertex is merely SUCCEEDED too -- and would stay
    `succeeded` if Vertex had been slow to update."""
    client, sdk, job, objects, *_ = service
    job.state.name = "JOB_STATE_RUNNING"
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps(
        {"status": "success", "stage": None}
    )
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["run_state"] == "succeeded"


def test_a_worker_reporting_progress_on_a_live_job_is_running(service):
    client, sdk, job, objects, *_ = service
    job.state.name = "JOB_STATE_RUNNING"
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps(
        {"status": "running", "stage": "s06_functional_models"}
    )
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["run_state"] == "running" and body["status"] == "running"


def test_a_queued_job_with_no_worker_output_is_submitted(service):
    client, sdk, job, *_ = service
    job.state.name = "JOB_STATE_QUEUED"
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["run_state"] == "submitted" and body["status"] == "pending"


def test_a_worker_that_exited_cleanly_without_results_is_abandoned(service):
    """The case the whole field exists for, and the one a sabotage pass showed was untested.

    results.json says `running`; Vertex says SUCCEEDED. The container exited 0 without writing a
    final result, so the worker's last word is stale and there is nothing to show. It is neither
    `running` -- the job is over -- nor `succeeded` -- there are no results. An earlier version of
    this rule asked for `status in (None, "pending")` before answering `abandoned`, which sent
    exactly this pair to `running` and left the UI reporting a finished job as still working.
    """
    client, sdk, job, objects, *_ = service
    job.state.name = "JOB_STATE_SUCCEEDED"
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps(
        {"status": "running", "stage": "s07_structure_mechanism"}
    )
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["status"] == "running", "still the worker's own word"
    assert body["run_state"] == "abandoned"


def test_a_worker_that_died_mid_stage_is_not_reported_as_still_running(service):
    """The case the field exists for: results.json says `running`, Vertex says FAILED. The
    worker's last word is stale and `run_state` must not repeat it."""
    client, sdk, job, objects, *_ = service
    job.state.name = "JOB_STATE_FAILED"
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps(
        {"status": "running", "stage": "s07_structure_mechanism"}
    )
    body = client.get(f"/api/v1/jobs/{JOB}/status").json()
    assert body["status"] == "running", "still the worker's own word"
    assert body["run_state"] == "failed"


def test_invalid_input_does_not_submit(service):
    client, sdk, job, objects, calls, payload = service
    payload["stages"].pop("s01_therapeutic_product_brief")
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 422
    assert not objects and not calls


def test_flat_stage_can_be_disabled(service):
    client, sdk, job, objects, calls, payload = service
    payload["stages"]["s07_structure_mechanism"]["enabled"] = False
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 202, response.text
    config = load_job_config(response.json()["result_path"] + "/config.json")
    stage = config.for_stage("s07_structure_mechanism")
    assert stage.enabled is False
    assert "enabled" not in stage.params
    assert stage.params["plddt_low_confidence_max"] == 0.5


def test_mixed_stage_format_rejected(service):
    client, sdk, job, objects, calls, payload = service
    payload["stages"]["s01_therapeutic_product_brief"]["params"] = {}
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 422
    assert not objects and not calls


def test_ranking_policy_is_always_builtin(service):
    """s11_ranking is excluded from E2ERequest entirely -- ranking is always
    the builtin, code-owned policy (see s11_ranking/stage.py), so a client
    including it at all is a 422, not a silently-discarded field."""
    client, sdk, job, objects, calls, payload = service
    payload["stages"]["s11_ranking"] = {"ranking_config": {}}
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 422
    assert not objects and not calls


def test_missing_job(service):
    client, sdk, *_ = service

    def missing(**kwargs):
        raise NotFound("missing")

    sdk.get_custom_job = missing
    assert client.get(f"/api/v1/jobs/{JOB}/status").status_code == 404


def test_submit_failure_is_not_reported_as_success(service):
    client, sdk, job, objects, calls, payload = service

    def failed(**kwargs):
        raise ServiceUnavailable("unavailable")

    sdk.create_custom_job = failed
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 502


def test_cancel(service):
    client, sdk, job, objects, calls, payload = service
    assert client.post(f"/api/v1/jobs/{JOB}/cancel").json()["status"] == "cancelling"
    assert calls[-1]["name"] == JOB


def test_repeated_request_id_submits_another_job(service):
    client, sdk, job, objects, calls, payload = service
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 202
    job.name = "projects/123/locations/us-central1/customJobs/789"
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 202
    assert response.json()["job_id"] == job.name
    assert len(calls) == 2
    assert not any(path.endswith("reserved.json") for path in objects)



def test_config_upload_failure_cancels_created_job(service, monkeypatch):
    client, sdk, job, objects, calls, payload = service

    def fail(*args, **kwargs):
        raise ServiceUnavailable("upload failed")

    monkeypatch.setattr(api.storage, "write_text", fail)
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 502
    assert calls[-1]["name"] == JOB

"""POST /api/v1/jobs/food/create on the Vertex path (no DEV_MODE), the status and cancel routes, and the local cancel."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.api_e2e import api
from food.pipeline.run_config import FOOD_REQUIRED_WEIGHT_FILES, FoodRunConfig

BASE = "/api/v1/jobs/food"
IMAGE = "us-central1-docker.pkg.dev/test/repo/worker-food:1"
JOB = "projects/123/locations/us-central1/customJobs/987654321"
FASTA = ">P1 whey beta-lactoglobulin\nVAGTWYSLAMAASDISLLDAQSAPLR\n>P2 casein\nFFVAPFPEVFGK\n>BAD1 bad\nAXB\n"
BODY = {"request_id": "req-1", "fasta": FASTA, "food_source": "Whey", "allergen_source": True}


@pytest.fixture
def service(monkeypatch):
    settings = {
        "VERTEX_CLOUD_PROJECT": "123", "VERTEX_LOCATION": "us-central1", "VERTEX_ARTIFACTS_DIR": "gs://test/artifacts", "VERTEX_MODEL_STORE": "gs://test/models",
        "WORKER_FOOD_IMAGE_URI": IMAGE, "WORKER_SERVICE_ACCOUNT": "worker@test.iam.gserviceaccount.com", "MACHINE_TYPE": "g2-standard-4",
        "ACCELERATOR_TYPE": "NVIDIA_L4", "ACCELERATOR_COUNT": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(api, "DEV_MODE", False)  # the suite runs under DEV_MODE=true; these tests are the Vertex path
    objects, calls, created = {f"gs://test/models/{f}": "{}" for f in FOOD_REQUIRED_WEIGHT_FILES}, [], []
    job = SimpleNamespace(name=JOB, state=SimpleNamespace(name="JOB_STATE_RUNNING"), error=SimpleNamespace(message=""))

    def create(**kwargs):
        created.append(kwargs)
        return job

    sdk = SimpleNamespace(create_custom_job=create, get_custom_job=lambda **kw: job, cancel_custom_job=lambda **kw: calls.append(kw))
    monkeypatch.setattr(api, "job_client", lambda: sdk)
    monkeypatch.setattr(api.storage, "write_text", lambda path, data: objects.__setitem__(path, data))
    monkeypatch.setattr(api.storage, "read_text", lambda path: objects[path])
    monkeypatch.setattr(api.storage, "exists", lambda path: path in objects)
    client = TestClient(api.app)
    client.created = created
    return client, objects, calls, job


def test_create_submits_a_job_on_the_food_image_and_writes_the_fasta_and_config(service):
    client, objects, _, _ = service
    response = client.post(f"{BASE}/create", json=BODY)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == JOB and body["accepted"] == 2 and [r["code"] for r in body["rejected"]] == ["NONSTANDARD_RESIDUES"]
    assert body["result_path"] == "gs://test/artifacts/runs/987654321"
    spec = client.created[0]["custom_job"]
    pool = spec["job_spec"]["worker_pool_specs"][0]
    assert pool["container_spec"]["image_uri"] == IMAGE and pool["machine_spec"] == {"machine_type": "g2-standard-4", "accelerator_type": "NVIDIA_L4", "accelerator_count": 1}
    assert pool["container_spec"]["args"] == ["--config", "gs://test/artifacts/requests/food/req-1/config.json", "--wait-for-config", "120"]
    assert {e["name"]: e["value"] for e in pool["container_spec"]["env"]} == {"DEV_MODE": "false", "VERTEX_MODEL_STORE": "gs://test/models"}  # the worker loads weights from the bucket
    assert spec["labels"]["pipeline"] == "food"
    assert objects["gs://test/artifacts/runs/987654321/proteins.fasta"] == FASTA


def test_the_config_the_api_writes_is_the_one_the_worker_reads(service):
    client, objects, _, _ = service
    client.post(f"{BASE}/create", json=BODY)
    config = FoodRunConfig.model_validate_json(objects["gs://test/artifacts/runs/987654321/config.json"])
    assert (config.run_id, config.food_source, config.allergen_source, config.model_store) == ("987654321", "Whey", True, "gs://test/models")
    assert config.fasta_path == "gs://test/artifacts/runs/987654321/proteins.fasta"
    assert objects["gs://test/artifacts/requests/food/req-1/config.json"] == objects["gs://test/artifacts/runs/987654321/config.json"]


def test_a_retry_returns_the_first_job_without_submitting_another(service):
    client, _, _, _ = service
    first = client.post(f"{BASE}/create", json=BODY).json()
    second = client.post(f"{BASE}/create", json=BODY).json()
    assert second["job_id"] == first["job_id"] and len(client.created) == 1


def test_a_fasta_with_no_usable_record_is_refused_before_anything_is_submitted(service):
    client, objects, _, _ = service
    response = client.post(f"{BASE}/create", json={**BODY, "fasta": ">BAD1 bad\nAXB\n"})
    assert response.status_code == 422 and response.json()["detail"]["rejected"][0]["code"] == "NONSTANDARD_RESIDUES"
    assert client.post(f"{BASE}/create", json={**BODY, "fasta": "MKWVT\n"}).status_code == 422  # not FASTA
    assert client.created == [] and not any(k.startswith("gs://test/artifacts") for k in objects)


def test_create_refuses_when_the_bucket_lacks_a_model(service):
    client, objects, _, _ = service
    del objects["gs://test/models/bitterness_predictor_v1/classification_thresholds.json"]
    response = client.post(f"{BASE}/create", json=BODY)
    assert response.status_code == 503 and "bitterness_predictor_v1/classification_thresholds.json" in response.json()["detail"] and client.created == []


def test_create_refuses_the_placeholder_model_store(service, monkeypatch):
    client, _, _, _ = service
    monkeypatch.setenv("VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store")
    assert client.post(f"{BASE}/create", json=BODY).status_code == 503 and client.created == []


def test_create_without_its_image_setting_or_a_gcs_artifact_folder_is_a_503(service, monkeypatch):
    client, _, _, _ = service
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", "/local/artifacts")
    assert client.post(f"{BASE}/create", json=BODY).status_code == 503
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", "gs://test/artifacts")
    monkeypatch.delenv("WORKER_FOOD_IMAGE_URI")
    assert client.post(f"{BASE}/create", json=BODY).status_code == 503


def test_a_config_upload_failure_cancels_the_job(service, monkeypatch):
    client, _, calls, _ = service

    def fail(path, data):
        raise OSError("bucket unreachable")

    monkeypatch.setattr(api.storage, "write_text", fail)
    response = client.post(f"{BASE}/create", json=BODY)
    assert response.status_code == 502 and calls and calls[0]["name"] == JOB


def test_healthz_reports_the_food_pipeline_on_its_own_image_setting(service, monkeypatch):
    client, _, _, _ = service
    assert client.get("/healthz").json()["pipelines"]["food"] == {"ready": True, "settings_missing": []}
    monkeypatch.delenv("WORKER_FOOD_IMAGE_URI")
    assert client.get("/healthz").json()["pipelines"]["food"] == {"ready": False, "settings_missing": ["WORKER_FOOD_IMAGE_URI"]}


def test_status_through_the_food_route_reconciles_the_worker_and_vertex(service):
    client, objects, _, job = service
    path = f"{BASE}/{JOB}/status"
    assert client.get(path).json()["run_state"] == "submitted"
    objects["gs://test/artifacts/runs/987654321/results.json"] = json.dumps({"status": "running", "stage": "s05_food_screening"})
    assert (client.get(path).json()["run_state"], client.get(path).json()["stage"]) == ("running", "s05_food_screening")
    objects["gs://test/artifacts/runs/987654321/results.json"] = json.dumps({"status": "success", "n_final": 25})
    job.state.name = "JOB_STATE_SUCCEEDED"
    assert client.get(path).json()["run_state"] == "succeeded"


def test_cancel_cancels_the_vertex_job_and_says_so_when_it_is_already_stopped(service):
    client, _, calls, job = service
    assert client.post(f"{BASE}/{JOB}/cancel").json() == {"job_id": JOB, "status": "cancelling"} and calls[0]["name"] == JOB
    job.state.name = "JOB_STATE_CANCELLED"
    assert client.post(f"{BASE}/{JOB}/cancel").json()["status"] == "cancelled"
    job.state.name = "JOB_STATE_FAILED"
    assert client.post(f"{BASE}/{JOB}/cancel").json()["status"] == "failed"


def test_a_malformed_job_id_is_a_422_on_the_food_routes(service):
    client, _, _, _ = service
    assert client.post(f"{BASE}/not-a-job/cancel").status_code == 422 and client.get(f"{BASE}/not-a-job/status").status_code == 422


# --- DEV_MODE: the job is a local process ------------------------------------------------------------------

@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "DEV_MODE", True)
    monkeypatch.setattr(api, "job_client", lambda: pytest.fail("local mode must never touch Vertex"))

    class FakeProcess:
        code = None

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

    monkeypatch.setattr(api.subprocess, "Popen", lambda command, **kwargs: FakeProcess())
    monkeypatch.setattr(api, "_local_processes", {})
    monkeypatch.setattr(api, "_local_cancelled", set())
    return TestClient(api.app), tmp_path


def test_local_cancel_stops_the_process_and_the_status_then_reads_cancelled(local):
    client, tmp_path = local
    body = client.post(f"{BASE}/create", json=BODY).json()
    (tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1] / "results.json").write_text(json.dumps({"status": "running", "stage": "s05_food_screening"}))
    assert client.get(f"{BASE}/{body['job_id']}/status").json()["run_state"] == "running"
    assert client.post(f"{BASE}/{body['job_id']}/cancel").json()["status"] == "cancelling"
    assert client.get(f"{BASE}/{body['job_id']}/status").json()["run_state"] == "cancelled"
    assert client.post(f"{BASE}/{body['job_id']}/cancel").json()["status"] == "failed"  # already stopped


def test_local_cancel_of_a_job_this_process_did_not_start_says_failed(local):
    client, _ = local
    assert client.post(f"{BASE}/projects/local/locations/local/customJobs/1/cancel").json()["status"] == "failed"

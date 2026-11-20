# The immunomodulation studio endpoints: the create stub, and status/results/cancel reusing the job machinery.
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.api_e2e import api

JOB = "projects/123/locations/us-central1/customJobs/456"
BASE = "/api/v1/jobs/immunomodulation"
IMAGE = "us-central1-docker.pkg.dev/test/images/worker-immuno:latest"
STAGING = "gs://test/artifacts/requests/immunomodulation/r1/config.json"


@pytest.fixture
def service(monkeypatch):
    settings = {
        "VERTEX_CLOUD_PROJECT": "123",
        "VERTEX_LOCATION": "us-central1",
        "VERTEX_ARTIFACTS_DIR": "gs://test/artifacts",
        "VERTEX_MODEL_STORE": "gs://test/models",
        "WORKER_IMMUNO_IMAGE_URI": IMAGE,
        "WORKER_SERVICE_ACCOUNT": "worker@test.iam.gserviceaccount.com",
        "MACHINE_TYPE": "g2-standard-4",
        "ACCELERATOR_TYPE": "NVIDIA_L4",
        "ACCELERATOR_COUNT": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(api, "DEV_MODE", False)  # the suite runs under DEV_MODE=true; these tests are the Vertex path
    from schemas.immuno_config import REQUIRED_WEIGHT_FILES

    objects, calls, created = {f"gs://test/models/{f}": "{}" for f in REQUIRED_WEIGHT_FILES}, [], []
    job = SimpleNamespace(name=JOB, state=SimpleNamespace(name="JOB_STATE_RUNNING"), error=SimpleNamespace(message=""))

    def create(**kwargs):
        created.append(kwargs)
        return job

    sdk = SimpleNamespace(
        create_custom_job=create,
        get_custom_job=lambda **kw: job,
        cancel_custom_job=lambda **kw: calls.append(kw),
    )
    monkeypatch.setattr(api, "job_client", lambda: sdk)
    monkeypatch.setattr(api.storage, "write_text", lambda path, data: objects.__setitem__(path, data))
    monkeypatch.setattr(api.storage, "read_text", lambda path: objects[path])
    monkeypatch.setattr(api.storage, "exists", lambda path: path in objects)
    client = TestClient(api.app)
    client.created = created
    return client, objects, calls, job


def test_create_submits_a_job_on_the_immuno_image_with_the_clean_peptides(service):
    client, objects, _, _ = service
    peptides = ["ALPMHIR", ">named\nAMKPWIQPK", "alpmhir", "ACDEFX", "ACD"]
    response = client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": peptides})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == JOB and body["result_path"] == "gs://test/artifacts/runs/456" and body["accepted"] == 2
    assert [(r["reason"], r["detail"]) for r in body["rejected"]] == [("duplicate", ""), ("invalid_residues", "X"), ("too_short", "3 < 6")]
    spec = client.created[0]["custom_job"]
    pool = spec["job_spec"]["worker_pool_specs"][0]["container_spec"]
    assert pool["image_uri"] == IMAGE and pool["args"] == ["--config", STAGING, "--wait-for-config", "120"]
    # production worker: DEV_MODE off, so it loads weights from the bucket the API just checked
    assert {e["name"]: e["value"] for e in pool["env"]} == {"DEV_MODE": "false", "VERTEX_MODEL_STORE": "gs://test/models"}
    assert spec["labels"]["pipeline"] == "immunomodulation" and spec["display_name"] == "ai-peptide-discovery-immuno-r1"
    config = json.loads(objects["gs://test/artifacts/runs/456/config.json"])
    assert config == {"run_id": "456", "artifacts_dir": "gs://test/artifacts", "model_store": "gs://test/models", "peptides": ["ALPMHIR", "AMKPWIQPK"], "batch_size": 100}
    assert objects[STAGING] == objects["gs://test/artifacts/runs/456/config.json"]


def test_the_config_the_api_writes_is_the_one_the_worker_reads(service):
    from schemas.immuno_config import ImmunoRunConfig

    client, objects, _, _ = service
    client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR"]})
    assert ImmunoRunConfig.model_validate_json(objects[STAGING]).peptides == ["ALPMHIR"]


def test_a_retry_returns_the_first_job_without_submitting_another(service):
    client = service[0]
    body = {"request_id": "r1", "peptides": ["ALPMHIR"]}
    first = client.post(f"{BASE}/create", json=body).json()
    second = client.post(f"{BASE}/create", json=body)
    assert second.status_code == 202 and second.json()["job_id"] == first["job_id"]
    assert len(client.created) == 1


def test_create_refuses_when_the_bucket_lacks_a_model(service):
    client, objects, _, _ = service
    del objects["gs://test/models/il6_induction_predictor_v1/info.json"]
    del objects["gs://test/models/esm2_t30_150M/config.json"]
    response = client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR"]})
    assert response.status_code == 503
    assert "il6_induction_predictor_v1/info.json" in response.text and "esm2_t30_150M/config.json" in response.text
    assert client.created == []


def test_create_refuses_the_placeholder_model_store(service, monkeypatch):
    monkeypatch.setenv("VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store")
    response = service[0].post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR"]})
    assert response.status_code == 503 and "placeholder" in response.text
    assert service[0].created == []


def test_create_without_its_image_setting_is_a_503(service, monkeypatch):
    monkeypatch.delenv("WORKER_IMMUNO_IMAGE_URI")
    response = service[0].post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR"]})
    assert response.status_code == 503 and "WORKER_IMMUNO_IMAGE_URI" in response.text


def test_a_config_upload_failure_cancels_the_job(service, monkeypatch):
    client, _, calls, _ = service

    def fail(path, data):
        raise OSError("bucket unavailable")

    monkeypatch.setattr(api.storage, "write_text", fail)
    response = client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR"]})
    assert response.status_code == 502 and calls and calls[0]["name"] == JOB


def test_healthz_reports_the_studio_as_its_own_pipeline(service, monkeypatch):
    monkeypatch.delenv("WORKER_IMMUNO_IMAGE_URI")
    body = service[0].get("/healthz").json()
    assert body["pipelines"]["immunomodulation"] == {"ready": False, "settings_missing": ["WORKER_IMMUNO_IMAGE_URI"]}


def test_the_batch_size_in_the_config_follows_the_environment(service, monkeypatch):
    client, objects = service[0], service[1]
    monkeypatch.setenv("IMMUNO_BATCH_SIZE", "25")
    client.post(f"{BASE}/create", json={"request_id": "r-batch", "peptides": ["ALPMHIR"]})
    configs = [json.loads(v) for k, v in objects.items() if k.endswith("config.json") and "r-batch" in k]
    assert configs and configs[0]["batch_size"] == 25
    monkeypatch.setenv("IMMUNO_BATCH_SIZE", "not-a-number")
    assert api._immuno_batch_size() == 100


def test_a_run_may_hold_ten_thousand_peptides_but_not_more(service):
    client = service[0]
    too_many = ["ALPMHIR"] * 10001
    assert client.post(f"{BASE}/create", json={"request_id": "big", "peptides": too_many}).status_code == 422


def test_create_rejects_a_request_with_no_valid_peptide(service):
    client = service[0]
    response = client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ACDEFX", "ACD"]})
    assert response.status_code == 422
    assert [r["reason"] for r in response.json()["detail"]["rejected"]] == ["invalid_residues", "too_short"]


@pytest.mark.parametrize("body", [{"request_id": "r1", "peptides": []}, {"peptides": ["ALPMHIR"]}, {"request_id": "r1", "peptides": ["ALPMHIR"], "extra": 1}])
def test_create_rejects_a_malformed_body(service, body):
    assert service[0].post(f"{BASE}/create", json=body).status_code == 422


def test_status_reconciles_worker_and_vertex_state(service):
    client, objects, _, _ = service
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps({"status": "running", "stage": "scoring"})
    body = client.get(f"{BASE}/{JOB}/status").json()
    assert (body["status"], body["run_state"], body["stage"]) == ("running", "running", "scoring")


def test_results_unfinished_run_has_an_empty_envelope(service):
    client, objects, _, _ = service
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps({"status": "running"})
    body = client.get(f"{BASE}/{JOB}/results").json()
    assert (body["pipeline"], body["status"], body["results"]) == ("immunomodulation", "running", [])


def test_results_of_a_finished_run(service):
    client, objects, _, _ = service
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps({"status": "success", "n_final": 1})
    objects["gs://test/artifacts/runs/456/immuno_results.json"] = json.dumps([{"sequence": "ALPMHIR", "predictions": {"IL-6": {"label": "x"}}}])
    body = client.get(f"{BASE}/{JOB}/results").json()
    assert body["n_final"] == 1 and body["results"][0]["sequence"] == "ALPMHIR"


def test_results_success_without_the_results_file_is_a_404(service):
    client, objects, _, _ = service
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps({"status": "success"})
    assert client.get(f"{BASE}/{JOB}/results").status_code == 404


def test_results_for_an_unknown_run_is_a_404(service):
    assert service[0].get(f"{BASE}/{JOB}/results").status_code == 404


def test_cancel_cancels_the_vertex_job(service):
    client, _, calls, _ = service
    response = client.post(f"{BASE}/{JOB}/cancel")
    assert response.json() == {"job_id": JOB, "status": "cancelling"}
    assert calls and calls[0]["name"] == JOB


def test_a_malformed_job_id_is_a_422(service):
    assert service[0].get(f"{BASE}/not-a-job/status").status_code == 422


# --- DEV_MODE: the job runs as a local subprocess, no Vertex, no image, no bucket ---------------------------

@pytest.fixture
def local(monkeypatch, tmp_path):
    for key in ("VERTEX_CLOUD_PROJECT", "VERTEX_LOCATION", "WORKER_IMMUNO_IMAGE_URI", "VERTEX_MODEL_STORE", "WORKER_SERVICE_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "DEV_MODE", True)
    monkeypatch.setattr(api, "job_client", lambda: pytest.fail("local mode must never touch Vertex"))
    started = []

    class FakeProcess:
        code = None

        def poll(self):
            return self.code

        def terminate(self):
            self.code = -15

    def fake_popen(command, **kwargs):
        started.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(api, "_local_processes", {})
    client = TestClient(api.app)
    client.started = started
    return client, tmp_path, api


def local_create(client):
    response = client.post(f"{BASE}/create", json={"request_id": "r1", "peptides": ["ALPMHIR", "ACDEFX"]})
    assert response.status_code == 202, response.text
    return response.json()


def test_local_create_needs_no_vertex_settings_and_starts_the_worker_on_a_local_config(local):
    client, tmp_path, _ = local
    body = local_create(client)
    run_id = body["job_id"].rsplit("/", 1)[-1]
    assert body["job_id"].startswith("projects/local/locations/local/customJobs/") and body["accepted"] == 1
    assert [r["reason"] for r in body["rejected"]] == ["invalid_residues"]
    command, kwargs = client.started[0]
    assert command[1:] == ["-m", "backend.worker_immuno.worker", "--config", str(tmp_path / "runs" / run_id / "config.json")]
    assert kwargs["env"]["DEV_MODE"] == "true" and (tmp_path / "runs" / run_id / "worker.log").exists()
    config = json.loads((tmp_path / "runs" / run_id / "config.json").read_text())
    assert config["peptides"] == ["ALPMHIR"] and config["model_store"] == "model_store"


def test_local_status_follows_the_process_then_the_worker(local):
    client, tmp_path, local_api = local
    body = local_create(client)
    run_id = body["job_id"].rsplit("/", 1)[-1]
    status = client.get(f"{BASE}/{body['job_id']}/status").json()
    assert (status["vertex_state"], status["run_state"]) == ("JOB_STATE_RUNNING", "submitted")
    (tmp_path / "runs" / run_id / "results.json").write_text(json.dumps({"status": "running", "stage": "scoring"}))
    assert client.get(f"{BASE}/{body['job_id']}/status").json()["run_state"] == "running"
    (tmp_path / "runs" / run_id / "results.json").write_text(json.dumps({"status": "success", "n_final": 1}))
    local_api._local_processes[run_id].code = 0
    assert client.get(f"{BASE}/{body['job_id']}/status").json()["run_state"] == "succeeded"


def test_local_failure_reports_the_workers_error(local):
    client, tmp_path, local_api = local
    body = local_create(client)
    run_id = body["job_id"].rsplit("/", 1)[-1]
    (tmp_path / "runs" / run_id / "results.json").write_text(json.dumps({"status": "failed", "error": "weights missing"}))
    local_api._local_processes[run_id].code = 1
    status = client.get(f"{BASE}/{body['job_id']}/status").json()
    assert (status["run_state"], status["error"]) == ("failed", "weights missing")


def test_local_results_are_read_from_the_local_run_directory(local):
    client, tmp_path, _ = local
    body = local_create(client)
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "results.json").write_text(json.dumps({"status": "success", "n_final": 1}))
    (run_dir / "immuno_results.json").write_text(json.dumps([{"sequence": "ALPMHIR", "predictions": {"IL-6": {"probability": 0.5}}}]))
    results = client.get(f"{BASE}/{body['job_id']}/results").json()
    assert results["results"][0]["sequence"] == "ALPMHIR" and results["status"] == "success"


def test_local_cancel_stops_the_process(local):
    client, _, local_api = local
    body = local_create(client)
    assert client.post(f"{BASE}/{body['job_id']}/cancel").json()["status"] == "cancelling"
    assert client.post(f"{BASE}/{body['job_id']}/cancel").json()["status"] == "failed"  # already stopped


# --- the registry endpoint and the standalone page's cross-origin access ----------------------------------

def test_registry_endpoint_lists_the_six_models_from_the_registry_and_cards(service):
    body = service[0].get("/api/v1/immunomodulation/models").json()
    from schemas.immuno_config import CYTOKINE_MODELS

    assert [m["cytokine"] for m in body["models"]] == list(CYTOKINE_MODELS)
    for entry in body["models"]:
        assert entry["registered"] and entry["version"] == "v1" and entry["card_status"] == "first version"
        assert entry["cv_metrics"]["MCC"] > 0 and entry["training"]["length_min"] >= 7
    assert "similarity" in body["caveat"]


def test_registry_endpoint_adds_the_domain_ranges_from_the_weights_info(service, monkeypatch):
    info = {
        "training": {"length_min": 8, "length_max": 12},
        "domain": {"min_examples_per_class": 10, "source": "x.csv", "length_counts": {str(n): {"pos": 20, "neg": 20} for n in range(8, 13)}},
    }
    monkeypatch.setattr(api, "_immuno_weights_info", lambda package: info)
    entry = service[0].get("/api/v1/immunomodulation/models").json()["models"][0]
    assert entry["domain"]["status_ranges"] == {"in_domain": [[8, 12]], "limited_evidence": [], "out_of_domain": [[6, 7], [13, 50]]}
    assert entry["domain"]["source"] == "x.csv" and "sequence_similarity" in entry["domain"]["not_checked"]


def test_registry_endpoint_still_answers_without_weights_info(service, monkeypatch):
    monkeypatch.setattr(api, "_immuno_weights_info", lambda package: None)
    entry = service[0].get("/api/v1/immunomodulation/models").json()["models"][0]
    assert entry["domain"] is None and entry["registered"]


@pytest.mark.skipif(not api.DEV_MODE, reason="CORS is only added under DEV_MODE")
def test_dev_mode_allows_a_page_from_another_origin(service):
    response = service[0].get("/healthz", headers={"Origin": "null"})
    assert response.headers["access-control-allow-origin"] == "*"


# --- confidential: which classifiers a model is made of never leaves the API ------------------------------------

MEMBER_NAMES = ("MLP", "XGBoost", "SVM_RBF", "LR-L2", "adasyn", "smote", "posw", "lenbin", "members", "aggregation", "ensemble", "notebook", "Weighted_5models")


def leaked(text: str) -> list[str]:
    return [word for word in MEMBER_NAMES if word.lower() in text.lower()]


def test_registry_endpoint_reveals_no_member_information(service):
    response = service[0].get("/api/v1/immunomodulation/models")
    assert response.status_code == 200 and leaked(response.text) == []


def test_generic_model_manifest_redacts_the_cytokine_cards_too(service):
    response = service[0].get("/api/v1/models")
    cards = [m for m in response.json()["models"] if m["name"].endswith("_induction_predictor_v1")]
    assert len(cards) == 6
    for entry in cards:
        assert leaked(json.dumps(entry)) == [], entry["name"]
        assert entry["model_card"]["cv_metrics_5fold_mean"]["MCC"] > 0  # the figures stay


def test_job_results_drop_the_per_member_probabilities(service):
    client, objects, _, _ = service
    objects["gs://test/artifacts/runs/456/results.json"] = json.dumps({"status": "success", "n_final": 1})
    objects["gs://test/artifacts/runs/456/immuno_results.json"] = json.dumps(
        [{"sequence": "ALPMHIR", "predictions": {"IL-6": {"probability": 0.5, "label": 1, "members": {"XGBoost_lenbin": 0.4}, "model": "il6_induction_predictor_v1"}}}]
    )
    response = client.get(f"{BASE}/{JOB}/results")
    prediction = response.json()["results"][0]["predictions"]["IL-6"]
    assert prediction == {"probability": 0.5, "label": 1, "model": "il6_induction_predictor_v1"}
    assert leaked(response.text) == []

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
        # The metabolic pipeline reads its OWN seed file, and create_job checks it BEFORE the
        # image. Without this, the two metabolic tests below failed in CI and passed on any
        # machine with a populated .env: one expected 202 and got a 503 about the seed file, and
        # the other asserted the 503 names WORKER_METABOLIC_IMAGE_URI when it actually named the
        # seed file. Neither was about seeds. (No cosmetic equivalent is set because nothing here
        # creates a cosmetic job yet; a test that does will need SEED_CANDIDATES_FILE_COSMETIC.)
        "SEED_CANDIDATES_FILE_METABOLIC": "gs://test/seeds-metabolic.fasta",
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


METABOLIC_IMAGE = "us-central1-docker.pkg.dev/test/images/worker-metabolic:latest"


def metabolic_payload():
    return {
        "request_id": "metabolic-001",
        "pipeline": "metabolic",
        "stages": {
            "s01_metabolic_therapeutic_product_brief": {
                "brief": {
                    "target_receptors": ["GLP1R", "GIPR"],
                    "min_length": 6,
                    "max_length": 40,
                }
            }
        },
    }


def test_metabolic_pipeline_routes_to_metabolic_image_and_stages(service, monkeypatch):
    client, sdk, job, objects, calls, payload = service
    monkeypatch.setenv("WORKER_METABOLIC_IMAGE_URI", METABOLIC_IMAGE)
    response = client.post("/api/v1/jobs/create", json=metabolic_payload())
    assert response.status_code == 202, response.text
    job_spec = calls[0]["custom_job"]
    assert job_spec["job_spec"]["worker_pool_specs"][0]["container_spec"]["image_uri"] == METABOLIC_IMAGE
    assert job_spec["labels"]["pipeline"] == "metabolic"
    config = json.loads(objects[response.json()["result_path"] + "/config.json"])
    enabled = {n for n, st in config["stages"].items() if st["enabled"]}
    assert "s11_metabolic_ranking" in enabled and "s11_ranking" not in enabled
    assert "s01_metabolic_therapeutic_product_brief" in enabled
    assert "s01_therapeutic_product_brief" not in enabled


def test_wound_request_still_uses_wound_image(service):
    client, sdk, job, objects, calls, payload = service
    assert "pipeline" not in payload  # the routing parameter defaults to wound_healing
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 202
    image = calls[0]["custom_job"]["job_spec"]["worker_pool_specs"][0]["container_spec"]["image_uri"]
    assert image.endswith("/worker:latest")


def test_metabolic_pipeline_needs_its_own_image_setting(service, monkeypatch):
    client = service[0]
    monkeypatch.delenv("WORKER_METABOLIC_IMAGE_URI", raising=False)
    response = client.post("/api/v1/jobs/create", json=metabolic_payload())
    assert response.status_code == 503
    assert "WORKER_METABOLIC_IMAGE_URI" in response.text


def test_pipeline_must_match_stages(service, monkeypatch):
    client, sdk, job, objects, calls, payload = service
    monkeypatch.setenv("WORKER_METABOLIC_IMAGE_URI", METABOLIC_IMAGE)
    wound_as_metabolic = {**payload, "pipeline": "metabolic"}
    assert client.post("/api/v1/jobs/create", json=wound_as_metabolic).status_code == 422
    metabolic_as_wound = {k: v for k, v in metabolic_payload().items() if k != "pipeline"}
    assert client.post("/api/v1/jobs/create", json=metabolic_as_wound).status_code == 422
    assert not calls


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


def test_repeated_request_id_does_not_submit_another_job(service):
    """REWRITTEN. This asserted the opposite -- that a repeated request_id starts a second job
    and returns the new job's name -- and so pinned the defect in place: both jobs are launched
    with the same `--config` path, so the second create overwrote it and the first job read the
    second run's config. The old expectation was the bug, written down.

    A repeated request_id is a RETRY. It returns the job the first attempt started.
    """
    client, sdk, job, objects, calls, payload = service
    first = client.post("/api/v1/jobs/create", json=payload)
    assert first.status_code == 202
    # Vertex would hand out a new name for a new job; nothing should ask it for one.
    job.name = "projects/123/locations/us-central1/customJobs/789"
    response = client.post("/api/v1/jobs/create", json=payload)
    assert response.status_code == 202
    assert response.json()["job_id"] == JOB != job.name
    assert len(calls) == 1
    assert not any(path.endswith("reserved.json") for path in objects)



def test_config_upload_failure_cancels_created_job(service, monkeypatch):
    client, sdk, job, objects, calls, payload = service

    def fail(*args, **kwargs):
        raise ServiceUnavailable("upload failed")

    monkeypatch.setattr(api.storage, "write_text", fail)
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 502
    assert calls[-1]["name"] == JOB


RUN = "gs://test/artifacts/runs/456"


def _publish_run(objects, stages, candidates):
    objects[f"{RUN}/results.json"] = json.dumps({"status": "success", "n_final": len(candidates)})
    objects[f"{RUN}/config.json"] = json.dumps({"stages": stages})
    objects[f"{RUN}/candidates_final.json"] = json.dumps(candidates)


def test_metabolic_results_carry_receptor_predictions_and_both_ranks(service):
    client, sdk, job, objects, *_ = service
    candidate = {
        "id": "m1",
        "sequence": "HAEGTFTSDV",
        "predictions": {
            "pec50": {"GLP1R": 10.5, "GIPR": 9.8, "GCGR": None},
            "ec50_nm": {"GLP1R": 0.03, "GIPR": 0.16, "GCGR": None},
            "binding_pkd": {"GLP1R": 8.9, "GIPR": 8.1, "GCGR": 7.0},
            "agonism_probability": {"GLP1R": 0.8, "GIPR": None, "GCGR": None},
            "pec50_pkd_gap": 1.4,
            "metabolic_activity_score": 0.7,
            "anti_inflammatory_probability": 0.9,
            "plasma_half_life": {"minutes": 240.0},
            "hemolysis": {"phc50": 3.1, "p_hemolytic": 0.2},
            "ranking": {"status": "ranked", "final_score": 0.6, "final_score_rank": 1, "pareto_rank": 2,
                        "evidence_coverage": 1.0, "missing_modules": [], "modules": {}},
        },
    }
    _publish_run(objects, {"s01_metabolic_therapeutic_product_brief": {"enabled": True}}, [candidate])
    body = client.get(f"/api/v1/jobs/{JOB}/results").json()
    assert body["pipeline"] == "metabolic"
    got = body["candidates"][0]
    assert (got["final_score_rank"], got["pareto_rank"]) == (1, 2)
    assert got["pec50"]["GLP1R"] == 10.5 and got["ec50_nm"]["GCGR"] is None
    assert got["binding_pkd"]["GIPR"] == 8.1 and got["pec50_pkd_gap"] == 1.4
    assert got["half_life_minutes"] == 240.0 and got["p_hemolytic"] == 0.2
    assert "amp_probability" not in got and "log_mic_um" not in got


def test_wound_results_keep_their_shape(service):
    client, sdk, job, objects, *_ = service
    candidate = {"id": "w1", "sequence": "KLAKLAK", "predictions": {
        "amp_probability": 0.9, "ranking": {"status": "ranked", "final_score_rank": 1, "pareto_rank": 1}}}
    _publish_run(objects, {"s01_therapeutic_product_brief": {"enabled": True}}, [candidate])
    body = client.get(f"/api/v1/jobs/{JOB}/results").json()
    assert body["pipeline"] == "wound_healing"
    assert body["candidates"][0]["amp_probability"] == 0.9
    assert "pec50" not in body["candidates"][0]


def test_results_pipeline_is_inferred_when_config_is_missing(service):
    client, sdk, job, objects, *_ = service
    objects[f"{RUN}/results.json"] = json.dumps({"status": "success", "n_final": 1})
    objects[f"{RUN}/candidates_final.json"] = json.dumps(
        [{"id": "m1", "sequence": "HAEG", "predictions": {"ec50_nm": {"GLP1R": 1.0}}}])
    assert client.get(f"/api/v1/jobs/{JOB}/results").json()["pipeline"] == "metabolic"


@pytest.mark.parametrize(
    ("brief_stage", "pipeline"),
    [
        ("s01_therapeutic_product_brief", "wound_healing"),
        ("s01_metabolic_therapeutic_product_brief", "metabolic"),
        ("s01_cosmetic_therapeutic_product_brief", "cosmetic"),
    ],
)
def test_an_unfinished_run_is_tagged_with_its_own_pipeline(service, brief_stage, pipeline):
    """A run that has not succeeded still answers, and the answer says which pipeline it is.

    This was an `if metabolic else wound_healing` ternary, so an unfinished COSMETIC run came
    back tagged "wound_healing". Nothing rendered wrongly, because the candidate list is empty
    here -- but the web app discriminates on this field to decide how to render a run, so a
    wrong tag is a wrong screen the moment these envelopes carry anything of their own.
    """
    client, sdk, job, objects, *_ = service
    objects[f"{RUN}/results.json"] = json.dumps({"status": "running", "n_final": None})
    objects[f"{RUN}/config.json"] = json.dumps({"stages": {brief_stage: {"enabled": True}}})
    body = client.get(f"/api/v1/jobs/{JOB}/results").json()
    assert body["pipeline"] == pipeline
    assert body["status"] == "running"
    assert body["candidates"] == []


def test_every_pipeline_has_an_unfinished_response_shape(service):
    """The map the ternary became. A fourth pipeline added to Pipeline without a row here is a
    KeyError on the next unfinished run, not a response quietly tagged wound_healing."""
    from backend.api_e2e.api import RESULTS_RESPONSE_FOR
    from typing import get_args

    from backend.api_e2e.api import Pipeline

    assert set(RESULTS_RESPONSE_FOR) == set(get_args(Pipeline))


def test_retrying_a_request_returns_the_first_job_and_starts_no_second_one(service):
    """The defect this guards is expensive: two paid GPU jobs, and one of them writing into the
    other's run directory.

    Every job for a request_id is launched with `--config <artifacts>/requests/<id>/config.json`
    -- one path, shared by every attempt -- and the worker waits up to 120s for it. A second
    create overwrote that file with the second run's config, so the first job read the SECOND
    run's run_id and wrote its results there, leaving the first run's directory empty forever.
    """
    client, sdk, job, objects, calls, payload = service
    first = client.post("/api/v1/jobs/create", json=payload)
    assert first.status_code == 202, first.text
    assert len(calls) == 1

    again = client.post("/api/v1/jobs/create", json=payload)
    assert again.status_code == 202, again.text
    # No second Vertex job, and the same job named back.
    assert len(calls) == 1
    assert again.json() == first.json()
    assert again.json()["job_id"] == JOB


def test_a_retry_whose_staging_config_is_unreadable_still_submits(service):
    """Half-written or corrupt staging config: there is no job name to return, so the request has
    to go through rather than fail. This is the one path that can still double-submit, which is
    why api.py logs it."""
    client, sdk, job, objects, calls, payload = service
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 202
    staging = "gs://test/artifacts/requests/" + payload["request_id"] + "/config.json"
    objects[staging] = "{not json"
    assert client.post("/api/v1/jobs/create", json=payload).status_code == 202
    assert len(calls) == 2


def test_an_unfinished_run_is_tagged_from_the_stage_when_the_config_is_missing(service):
    """The unfinished branch has no candidates to inspect, so without the worker's last stage it
    fell through to "wound_healing" however cosmetic the run was -- and the web app picks how to
    render a run from this field."""
    client, sdk, job, objects, *_ = service
    objects[f"{RUN}/results.json"] = json.dumps(
        {"status": "running", "stage": "s08_cosmetic_safety_developability"}
    )
    body = client.get(f"/api/v1/jobs/{JOB}/results").json()
    assert body["pipeline"] == "cosmetic"
    assert body["candidates"] == []


def test_a_shared_stage_says_nothing_about_the_pipeline(service):
    """s07_structure_mechanism is the same stage for all three, so it must not be read as a
    cosmetic or metabolic signal. Falls back to the default rather than guessing."""
    client, sdk, job, objects, *_ = service
    objects[f"{RUN}/results.json"] = json.dumps(
        {"status": "running", "stage": "s07_structure_mechanism"}
    )
    assert client.get(f"/api/v1/jobs/{JOB}/results").json()["pipeline"] == "wound_healing"


def test_gipr_is_flagged_on_a_run_that_never_recorded_the_flag(service):
    """Runs from before stage 6 recorded potency_unreliable_receptors have GIPR values and no
    flag. Four such candidates reached the web app showing GIPR numbers with nothing saying they
    cannot be ranked, so the API derives the flag when the run did not store one."""
    client, sdk, job, objects, *_ = service
    candidate = {
        "id": "old1",
        "sequence": "HAEGTFTSDV",
        "predictions": {
            # No potency_unreliable_receptors key at all: this is the old shape.
            "pec50": {"GLP1R": 10.5, "GIPR": 9.8},
            "ec50_nm": {"GLP1R": 0.03, "GIPR": 0.16},
            "ranking": {"status": "ranked", "final_score_rank": 1},
        },
    }
    _publish_run(objects, {"s01_metabolic_therapeutic_product_brief": {"enabled": True}}, [candidate])
    got = client.get(f"/api/v1/jobs/{JOB}/results").json()["candidates"][0]
    assert got["potency_unreliable_receptors"] == ["GIPR"]
    # The note is what says the values still count toward the gates and the score. A flag
    # without it is a worse half-truth than no flag.
    assert "still count toward the gates" in got["potency_unreliable_note"]
    # And the values are still returned, which is the whole contract.
    assert got["pec50"]["GIPR"] == 9.8


def test_a_run_that_did_not_target_gipr_is_not_flagged(service):
    client, sdk, job, objects, *_ = service
    candidate = {
        "id": "old2",
        "sequence": "HAEGTFTSDV",
        "predictions": {
            "pec50": {"GLP1R": 10.5, "GCGR": 8.1},
            "ranking": {"status": "ranked", "final_score_rank": 1},
        },
    }
    _publish_run(objects, {"s01_metabolic_therapeutic_product_brief": {"enabled": True}}, [candidate])
    got = client.get(f"/api/v1/jobs/{JOB}/results").json()["candidates"][0]
    assert got["potency_unreliable_receptors"] == []
    assert got["potency_unreliable_note"] is None


def test_an_empty_recorded_flag_is_kept_empty(service):
    """A run that DID record the flag and found nothing unreliable must stay empty. Treating the
    stored empty list as "not recorded" would re-derive it and flag a GIPR value the run had
    deliberately not targeted."""
    client, sdk, job, objects, *_ = service
    candidate = {
        "id": "new1",
        "sequence": "HAEGTFTSDV",
        "predictions": {
            "pec50": {"GIPR": 9.8},
            "potency_unreliable_receptors": [],
            "ranking": {"status": "ranked", "final_score_rank": 1},
        },
    }
    _publish_run(objects, {"s01_metabolic_therapeutic_product_brief": {"enabled": True}}, [candidate])
    got = client.get(f"/api/v1/jobs/{JOB}/results").json()["candidates"][0]
    assert got["potency_unreliable_receptors"] == []

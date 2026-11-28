"""The food protein library over the API: discovery routes, library mode, and an upload matched against the library without being rejected for not matching."""

from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.api_e2e import api
from food.a_proteins.library import library
from food.pipeline.run_config import FOOD_REQUIRED_WEIGHT_FILES, FoodRunConfig

BASE = "/api/v1/jobs/food"
L = library()
WHEY, CASEIN = (next(p for p in L.proteins if p.food_source == s) for s in ("Whey", "Casein"))
NOT_IN_LIBRARY = "MKVLAAGIVGLLLAQPSAFAEQ" * 3


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "DEV_MODE", True)
    monkeypatch.setattr(api, "job_client", lambda: pytest.fail("local mode must never touch Vertex"))
    monkeypatch.setattr(api.subprocess, "Popen", lambda command, **kwargs: SimpleNamespace(poll=lambda: None, terminate=lambda: None))
    monkeypatch.setattr(api, "_local_processes", {})
    return TestClient(api.app), tmp_path


def run_dir(tmp_path, body):
    return tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]


# --- discovery --------------------------------------------------------------------------------------------------

def test_the_summary_route_reports_the_release_and_counts(local):
    client, _ = local
    body = client.get("/api/v1/food/library/summary").json()
    assert body["total"] == 524 and body["manifest"]["uniprot_release"] == ["2026_04"] and body["by_food_source"]["Casein"] == 23 and body["families"]


def test_the_proteins_route_filters_pages_and_never_returns_a_sequence(local):
    client, _ = local
    body = client.get("/api/v1/food/library/proteins", params={"food_source": "Casein", "limit": 5}).json()
    assert body["total"] == 23 and len(body["proteins"]) == 5 and all(p["food_source"] == "Casein" for p in body["proteins"])
    assert all("sequence" not in p for p in body["proteins"]) and {"substrate_id", "protein_family", "allergen_source", "uniprot_release", "length"} <= set(body["proteins"][0])
    assert client.get("/api/v1/food/library/proteins", params={"q": CASEIN.accession}).json()["total"] >= 1
    assert client.get("/api/v1/food/library/proteins", params={"limit": 501}).status_code == 422 and client.get("/api/v1/food/library/proteins", params={"offset": -1}).status_code == 422


# --- library mode ----------------------------------------------------------------------------------------------------

def test_a_run_of_library_proteins_needs_no_fasta_and_carries_the_librarys_records(local):
    client, tmp_path = local
    response = client.post(f"{BASE}/create", json={"request_id": "lib-1", "library_proteins": [WHEY.substrate_id, CASEIN.substrate_id]})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["input_mode"] == "library" and body["accepted"] == 2 and body["rejected"] == [] and body["library_matches"] == []
    d = run_dir(tmp_path, body)
    config = FoodRunConfig.model_validate_json((d / "config.json").read_text())
    assert config.source_records_path.endswith("source_records.csv") and config.food_source == "" and config.allergen_source is None
    assert [r["accession"] for r in csv.DictReader((d / "source_records.csv").open(encoding="utf-8"))] == [WHEY.accession, CASEIN.accession]
    assert (d / "proteins.fasta").read_text(encoding="utf-8") == L.fasta_text([WHEY, CASEIN])
    record = json.loads((d / "library_match.json").read_text())
    assert record["mode"] == "library" and record["library_proteins"] == [WHEY.substrate_id, CASEIN.substrate_id] and record["library"]["uniprot_release"] == ["2026_04"]


def test_an_unknown_library_protein_is_a_422_that_names_it_and_starts_nothing(local):
    client, tmp_path = local
    response = client.post(f"{BASE}/create", json={"request_id": "lib-2", "library_proteins": [WHEY.substrate_id, "NOT:1-9"]})
    assert response.status_code == 422 and response.json()["detail"]["missing"] == ["NOT:1-9"] and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("body", [
    {"request_id": "r"},  # neither input
    {"request_id": "r", "fasta": ">a\nAAAAAA\n", "food_source": "x", "library_proteins": ["P02754:17-178"]},  # both
    {"request_id": "r", "library_proteins": ["P02754:17-178"], "food_source": "Whey"},  # the library supplies it
    {"request_id": "r", "library_proteins": ["P02754:17-178"], "allergen_source": True},
    {"request_id": "r", "library_proteins": []},
])
def test_the_request_names_exactly_one_input(local, body):
    client, _ = local
    assert client.post(f"{BASE}/create", json=body).status_code == 422


# --- upload mode: matched, never rejected for not matching ----------------------------------------------------------

def upload_body(*records):
    return {"request_id": "up-1", "fasta": "".join(f">{rid} FS=Whey\n{seq}\n" for rid, seq in records), "food_source": "Whey", "allergen_source": True}


def test_an_upload_that_is_exactly_a_library_protein_gets_its_record_and_one_that_is_not_is_kept(local):
    client, tmp_path = local
    body = client.post(f"{BASE}/create", json=upload_body(("MY-WHEY-1", WHEY.sequence), ("NEW-2", NOT_IN_LIBRARY))).json()
    assert body["input_mode"] == "upload" and body["accepted"] == 2  # the protein that is not in the library is accepted
    statuses = {m["id"]: m for m in body["library_matches"]}
    assert statuses["MY-WHEY-1:1-" + str(len(WHEY.sequence))]["status"] == "exact_sequence" and statuses["MY-WHEY-1:1-" + str(len(WHEY.sequence))]["library_substrate_id"] == WHEY.substrate_id
    unknown = statuses["NEW-2:1-66"]
    assert unknown["status"] == "not_in_library" and "family and allergen source are not established" in unknown["message"] and unknown["protein_family"] is None
    rows = list(csv.DictReader((run_dir(tmp_path, body) / "source_records.csv").open(encoding="utf-8")))
    assert [r["accession"] for r in rows] == ["MY-WHEY-1"] and rows[0]["protein_family"] == WHEY.protein_family  # keyed by the upload's accession
    config = FoodRunConfig.model_validate_json((run_dir(tmp_path, body) / "config.json").read_text())
    assert config.allergen_source is True and config.food_source == "Whey"  # for the proteins no record covers


def test_an_upload_with_nothing_in_the_library_writes_no_records_and_is_not_refused(local):
    client, tmp_path = local
    response = client.post(f"{BASE}/create", json=upload_body(("NEW-1", NOT_IN_LIBRARY)))
    assert response.status_code == 202 and [m["status"] for m in response.json()["library_matches"]] == ["not_in_library"]
    d = run_dir(tmp_path, response.json())
    assert not (d / "source_records.csv").exists() and FoodRunConfig.model_validate_json((d / "config.json").read_text()).source_records_path == ""


def test_the_results_report_how_the_proteins_relate_to_the_library(local):
    client, tmp_path = local
    body = client.post(f"{BASE}/create", json=upload_body(("NEW-1", NOT_IN_LIBRARY))).json()
    d = run_dir(tmp_path, body)
    (d / "candidates").mkdir()
    (d / "candidates" / "s07_food_ranking_decision.jsonl").write_text("")
    (d / "results.json").write_text(json.dumps({"status": "success", "n_final": 0}))
    library_record = client.get(f"{BASE}/{body['job_id']}/results").json()["library"]
    assert library_record["mode"] == "upload" and library_record["matches"][0]["status"] == "not_in_library" and library_record["library"]["name"]


# --- Vertex path ------------------------------------------------------------------------------------------------------

@pytest.fixture
def vertex(monkeypatch):
    for key, value in {"VERTEX_CLOUD_PROJECT": "123", "VERTEX_LOCATION": "us-central1", "VERTEX_ARTIFACTS_DIR": "gs://test/artifacts", "VERTEX_MODEL_STORE": "gs://test/models",
                       "WORKER_FOOD_IMAGE_URI": "img", "WORKER_SERVICE_ACCOUNT": "w@test", "MACHINE_TYPE": "g2-standard-4", "ACCELERATOR_TYPE": "NVIDIA_L4", "ACCELERATOR_COUNT": "1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(api, "DEV_MODE", False)
    objects = {f"gs://test/models/{f}": "{}" for f in FOOD_REQUIRED_WEIGHT_FILES}
    job = SimpleNamespace(name="projects/123/locations/us-central1/customJobs/77", state=SimpleNamespace(name="JOB_STATE_RUNNING"), error=SimpleNamespace(message=""))
    monkeypatch.setattr(api, "job_client", lambda: SimpleNamespace(create_custom_job=lambda **kw: job, cancel_custom_job=lambda **kw: None))
    monkeypatch.setattr(api.storage, "write_text", lambda path, data: objects.__setitem__(path, data))
    monkeypatch.setattr(api.storage, "read_text", lambda path: objects[path])
    monkeypatch.setattr(api.storage, "exists", lambda path: path in objects)
    return TestClient(api.app), objects


def test_a_library_run_on_vertex_stores_its_records_beside_the_fasta(vertex):
    client, objects = vertex
    body = client.post(f"{BASE}/create", json={"request_id": "lib-v", "library_proteins": [WHEY.substrate_id]}).json()
    run = "gs://test/artifacts/runs/77"
    assert body["input_mode"] == "library" and objects[f"{run}/proteins.fasta"] == L.fasta_text([WHEY])
    assert next(csv.DictReader(io.StringIO(objects[f"{run}/source_records.csv"])))["accession"] == WHEY.accession
    assert FoodRunConfig.model_validate_json(objects[f"{run}/config.json"]).source_records_path == f"{run}/source_records.csv"
    assert json.loads(objects[f"{run}/library_match.json"])["mode"] == "library"

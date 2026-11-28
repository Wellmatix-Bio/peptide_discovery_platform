"""POST /api/v1/jobs/food/create: DEV_MODE starts the food worker as a local process on the submitted FASTA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api_e2e import api
from food.pipeline.run_config import FoodRunConfig

URL = "/api/v1/jobs/food/create"
FASTA = ">P1 whey beta-lactoglobulin\nVAGTWYSLAMAASDISLLDAQSAPLR\n>P2 casein\nFFVAPFPEVFGK\n>BAD1 bad\nAXB\n"


@pytest.fixture
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("VERTEX_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "DEV_MODE", True)
    monkeypatch.setattr(api, "job_client", lambda: pytest.fail("local mode must never touch Vertex"))
    started = []

    class FakeProcess:
        code = None

        def poll(self):
            return self.code

    monkeypatch.setattr(api.subprocess, "Popen", lambda command, **kwargs: started.append((command, kwargs)) or FakeProcess())
    monkeypatch.setattr(api, "_local_processes", {})
    client = TestClient(api.app)
    client.started = started
    return client, tmp_path


def create(client, **over):
    body = {"request_id": "r1", "fasta": FASTA, "food_source": "Whey", "allergen_source": True, **over}
    return client.post(URL, json=body)


def test_create_starts_the_food_worker_on_the_submitted_proteins(local):
    client, tmp_path = local
    response = create(client)
    assert response.status_code == 202, response.text
    body = response.json()
    run_id = body["job_id"].rsplit("/", 1)[-1]
    assert body["job_id"].startswith("projects/local/locations/local/customJobs/") and body["accepted"] == 2
    assert [(r["id"], r["code"]) for r in body["rejected"]] == [("BAD1:1-3", "NONSTANDARD_RESIDUES")]  # reported, not dropped silently
    command, kwargs = client.started[0]
    assert command[1:] == ["-m", "backend.worker_food.worker", "--config", str(tmp_path / "runs" / run_id / "config.json")]
    assert kwargs["env"]["DEV_MODE"] == "true"


def test_the_config_the_api_writes_is_the_one_the_worker_reads(local):
    client, tmp_path = local
    run_id = create(client, allergen_source=None).json()["job_id"].rsplit("/", 1)[-1]
    run_dir = tmp_path / "runs" / run_id
    config = FoodRunConfig.model_validate_json((run_dir / "config.json").read_text())
    assert (config.run_id, config.food_source, config.allergen_source) == (run_id, "Whey", None)  # not established stays None
    assert (run_dir / "proteins.fasta").read_text() == FASTA and config.fasta_path.endswith("proteins.fasta")


def test_the_worker_applies_the_run_food_source_and_allergen_status(local):
    from backend.worker_food.worker import load_run_proteins

    client, tmp_path = local
    run_id = create(client).json()["job_id"].rsplit("/", 1)[-1]
    config = FoodRunConfig.model_validate_json((tmp_path / "runs" / run_id / "config.json").read_text())
    proteins, issues, notes = load_run_proteins(config)
    assert [p.substrate_id for p in proteins] == ["P1:1-26", "P2:1-12"]
    assert {p.food_source for p in proteins} == {"Whey"} and {p.allergen_source for p in proteins} == {True}
    assert [i.code for i in issues] == ["NONSTANDARD_RESIDUES"]


def test_a_fasta_with_no_usable_record_is_a_422_and_starts_nothing(local):
    client, _ = local
    response = create(client, fasta=">BAD1 bad\nAXB\n")
    assert response.status_code == 422 and response.json()["detail"]["rejected"][0]["code"] == "NONSTANDARD_RESIDUES"
    assert create(client, fasta="MKWVT\n").status_code == 422  # not FASTA: no '>' header
    assert client.started == []


@pytest.mark.parametrize("body", [{}, {"request_id": "r", "fasta": ">a\nAAAA\n"}, {"request_id": "r", "fasta": ">a\nAAAA\n", "food_source": "x", "extra": 1}])
def test_a_malformed_body_is_a_422(local, body):
    client, _ = local
    assert client.post(URL, json=body).status_code == 422


def test_status_follows_the_process_then_the_worker(local):
    client, tmp_path = local
    body = create(client).json()
    run_id = body["job_id"].rsplit("/", 1)[-1]
    status = client.get(f"/api/v1/jobs/food/{body['job_id']}/status").json()
    assert (status["vertex_state"], status["run_state"]) == ("JOB_STATE_RUNNING", "submitted")
    (tmp_path / "runs" / run_id / "results.json").write_text(json.dumps({"status": "running", "stage": "s05_food_screening"}))
    status = client.get(f"/api/v1/jobs/food/{body['job_id']}/status").json()
    assert (status["run_state"], status["stage"]) == ("running", "s05_food_screening")


def test_the_stage_is_finished_once_the_worker_reports_success_whether_or_not_it_wrote_a_stage(local):
    client, tmp_path = local
    body = create(client).json()
    results = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1] / "results.json"
    for written in ({"status": "success", "stage": None}, {"status": "success"}, {"status": "success", "stage": "finished"}):  # a worker of any age
        results.write_text(json.dumps(written))
        status = client.get(f"/api/v1/jobs/food/{body['job_id']}/status").json()
        assert (status["status"], status["stage"], status["run_state"]) == ("success", "finished", "succeeded")
    results.write_text(json.dumps({"status": "failed", "stage": None, "error": "x"}))
    assert client.get(f"/api/v1/jobs/food/{body['job_id']}/status").json()["stage"] == "pending"  # only a success is finished


def test_the_food_worker_writes_stage_finished_on_success():
    from pathlib import Path

    from backend.worker_food import worker

    assert '"stage": "finished"' in Path(worker.__file__).read_text(encoding="utf-8")


# --- the worker's run loop, with the runner faked ------------------------------------------------------------

def write_config(tmp_path, fasta):
    run_dir = tmp_path / "runs" / "w1"
    run_dir.mkdir(parents=True)
    (run_dir / "proteins.fasta").write_text(fasta)
    config = FoodRunConfig(run_id="w1", artifacts_dir=str(tmp_path), model_store="model_store", stages={},
                           fasta_path=str(run_dir / "proteins.fasta"), food_source="Whey")
    path = run_dir / "config.json"
    path.write_text(config.model_dump_json())
    return run_dir, str(path)


def test_the_worker_runs_the_stages_on_the_submitted_proteins_and_reports_success(tmp_path, monkeypatch):
    from dataclasses import dataclass

    from backend.worker_food import worker
    from food.pipeline import runner as food_runner

    @dataclass(frozen=True)
    class Ctx:
        proteins: list | None = None

    runners = []

    class FakeRunner:
        def __init__(self, config, status_path=None):
            self.ctx = Ctx()
            self.candidate_stages = []
            runners.append(self)

        def run(self):
            return []

    monkeypatch.setattr(food_runner, "FoodPipelineRunner", FakeRunner)
    run_dir, config_path = write_config(tmp_path, FASTA)
    worker.run_job(config_path)
    assert json.loads((run_dir / "results.json").read_text())["status"] == "success"
    assert [p.substrate_id for p in runners[0].ctx.proteins] == ["P1:1-26", "P2:1-12"]
    assert [i["code"] for i in json.loads((run_dir / "protein_qc.json").read_text())] == ["NONSTANDARD_RESIDUES"]


def test_the_worker_fails_the_run_when_no_protein_is_usable(tmp_path):
    from backend.worker_food import worker

    run_dir, config_path = write_config(tmp_path, ">BAD1 bad\nAXB\n")
    with pytest.raises(ValueError, match="no usable protein"):
        worker.run_job(config_path)
    assert json.loads((run_dir / "results.json").read_text())["status"] == "failed"


# --- results ------------------------------------------------------------------------------------------------

FIXTURE_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "food" / "run"  # four real ranked peptides from a real-weights run: two designed and shortlisted, one digested and not shortlisted, one digested with an exact evidence match


@pytest.fixture
def finished(local):
    client, tmp_path = local
    body = create(client).json()
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "candidates").mkdir()
    (run_dir / "candidates" / "s07_food_ranking_decision.jsonl").write_text((FIXTURE_RUN / "s07_food_ranking_decision.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps({"status": "success", "n_final": 4}))
    return client, run_dir, body["job_id"]


def results(client, job_id):
    return client.get(f"/api/v1/jobs/food/{job_id}/results")


def test_results_list_the_ranked_peptides_with_their_traceability_and_limits(finished):
    client, _, job_id = finished
    body = results(client, job_id).json()
    assert body["status"] == "success" and body["pipeline"] == "food" and body["n_final"] == 4
    assert [p["rank"] for p in body["peptides"]] == sorted(p["rank"] for p in body["peptides"])
    first = body["peptides"][0]
    assert first["parents"] and first["parents"][0]["protein_id"] and first["parent_clusters"]
    assert first["rationale"] and first["limitations"] and first["glp1_classifier_status"] == "Not validated"
    assert {"safety", "developability", "uncut_bond_fraction"} <= set(first["scores"]) and "gut_survival" not in first["scores"] and "overall" in first["applicability"]
    assert set(first["models"]) >= {"allergenicity", "bitterness"} and first["cleavage"]["rules"]["pepsin"] == "pepsin_ph1.3"
    assert set(first["cleavage"]["cleavage_positions"]) == {"pepsin", "trypsin", "chymotrypsin"} and "does not measure survival" in first["cleavage"]["statement"]
    assert first["evidence"]["evidence_type"] == "SEQUENCE_SIMILARITY" and first["evidence"]["evidence_applies_to"] == "CANDIDATE" and first["evidence"]["evidence_gate_passed"] is False
    assert first["evidence"]["template_evidence_applies_to"] == "TEMPLATE" and "evidence_gate" not in first["evidence"] and first["review"] is None
    assert first["digestion_product_type"] is None and body["peptides"][-1]["digestion_product_type"] == "FINAL_PRODUCT" and body["peptides"][-1]["evidence"]["evidence_gate_passed"] is True


def test_the_shortlist_is_the_shortlisted_peptides_in_shortlist_order(finished):
    client, _, job_id = finished
    body = results(client, job_id).json()
    assert [p["shortlist_position"] for p in body["shortlist"]] == [1, 2, 3]
    assert all(p["shortlist_position"] is None for p in body["peptides"] if p not in body["shortlist"])
    assert body["review_gate"]["status"] == "AWAITING_REVIEW" and len(body["review_gate"]["pending"]) == 3 and body["review_gate"]["approved"] == []


def test_gate_controls_qc_and_cluster_sources_come_from_the_run_folder(finished):
    client, run_dir, job_id = finished
    (run_dir / "ranking_report.json").write_text(json.dumps({"gate": {"selected": 3, "shortfall": 0}, "facts": {"n_clusters": 2}, "controls": [{"role": "positive_control", "sequence": "APDSGNFR"}]}))
    (run_dir / "protein_clusters.json").write_text(json.dumps({"notes": ["no sequence clustering tool (mmseqs, cd-hit) on PATH"],
                                                               "clusters": {"P1:1-26": {"cluster": "protein:P1", "source": "protein_id"}, "P2:1-12": {"cluster": "protein:P2", "source": "protein_id"}}}))
    (run_dir / "protein_qc.json").write_text(json.dumps([{"substrate_id": "BAD1:1-3", "severity": "error", "code": "NONSTANDARD_RESIDUES", "message": "x"}]))
    body = results(client, job_id).json()
    assert body["gate"]["shortfall"] == 0 and body["controls"][0]["role"] == "positive_control" and body["facts"] == {"n_clusters": 2}
    assert body["clusters"] == {"notes": ["no sequence clustering tool (mmseqs, cd-hit) on PATH"], "proteins_per_source": {"protein_id": 2}}
    assert body["protein_qc"][0]["code"] == "NONSTANDARD_RESIDUES"


def test_a_missing_number_in_a_run_file_becomes_null_not_invalid_json(finished):
    client, run_dir, job_id = finished
    path = run_dir / "candidates" / "s07_food_ranking_decision.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["predictions"]["ranking"]["evidence_similarity"] = float("nan")
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    response = results(client, job_id)
    assert response.status_code == 200 and "NaN" not in response.text


def test_an_unfinished_run_has_an_empty_envelope_and_an_unknown_run_is_a_404(local):
    client, tmp_path = local
    body = create(client).json()
    assert results(client, body["job_id"]).status_code == 404  # the worker has written nothing yet
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "results.json").write_text(json.dumps({"status": "running", "stage": "s05_food_screening"}))
    assert results(client, body["job_id"]).json()["peptides"] == []
    assert results(client, "projects/local/locations/local/customJobs/1").status_code == 404


def test_success_without_the_ranked_file_is_a_404(finished):
    client, run_dir, job_id = finished
    (run_dir / "candidates" / "s07_food_ranking_decision.jsonl").unlink()
    assert results(client, job_id).status_code == 404


def test_the_worker_saves_the_gate_facts_and_controls_the_route_returns(tmp_path):
    from backend.worker_food import worker

    class Stage:
        name = "s07_food_ranking_decision"
        gate, facts, controls = {"shortfall": 0}, {"n_clusters": 5}, [{"role": "negative_control", "sequence": "GGGG"}]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    worker.write_ranking_report(type("R", (), {"candidate_stages": [type("Other", (), {"name": "s06"})(), Stage()]})(), str(run_dir))
    saved = json.loads((run_dir / "ranking_report.json").read_text())
    assert {k: saved[k] for k in ("gate", "facts", "controls", "digestion_warnings")} == {"gate": {"shortfall": 0}, "facts": {"n_clusters": 5}, "controls": [{"role": "negative_control", "sequence": "GGGG"}], "digestion_warnings": []}
    assert saved["rules"]["max_per_cluster"] == 2 and saved["rules"]["max_per_parent_family"] == 3 and saved["rules"]["final_count"] == 25 and "tie_break" in saved["rules"]  # the settings the ranking ran under


# --- dossier and experiment sheet ----------------------------------------------------------------------------

def test_the_dossier_route_returns_json_html_and_a_404_for_a_peptide_that_was_not_ranked(finished):
    client, run_dir, job_id = finished
    (run_dir / "ranking_report.json").write_text(json.dumps({"controls": [{"role": "positive_control", "sequence": "APDSGNFR", "study": "PMID40509480", "assay": "STC-1", "n_observations": "7"}]}))
    ids = [json.loads(line)["id"] for line in (run_dir / "candidates" / "s07_food_ranking_decision.jsonl").read_text(encoding="utf-8").splitlines()]
    base = f"/api/v1/jobs/food/{job_id}/dossiers"
    body = client.get(f"{base}/{ids[0]}").json()
    assert body["peptide_id"] == ids[0] and body["rank"] == 1 and body["controls"][0]["sequence"] == "APDSGNFR"
    assert body["evidence"]["candidate"]["applies_to"] == "CANDIDATE" and body["evidence"]["template"]["applies_to"] == "TEMPLATE" and body["safety_screening"]["ige_binding"]["status"] == "RUN"
    assert body["gut_stability"]["predicted_cleavage_positions"] and body["comparison_materials"]["parent_proteins"] and body["digestion"]["observation_status"] == "PREDICTED"
    assert body["review"]["decision"] is None and all(v is None for v in body["assay_plan"]["set_by_biology_lead"].values())
    page = client.get(f"{base}/{ids[0]}?format=html")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html") and body["sequence"] in page.text
    assert client.get(f"{base}/PEP-NOT-RANKED").status_code == 404
    assert client.get(f"{base}/{ids[0]}?format=docx").status_code == 422


def test_the_experiment_sheet_is_a_csv_of_the_shortlist_and_the_controls(finished):
    import csv
    import io

    client, run_dir, job_id = finished
    (run_dir / "ranking_report.json").write_text(json.dumps({"controls": [{"role": "negative_control", "sequence": "DLVDK", "study": "PMID33396793", "assay": "STC-1", "n_observations": "2", "note": "proposed by rule"}]}))
    response = client.get(f"/api/v1/jobs/food/{job_id}/experiment-sheet")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv") and "attachment" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert [r["row_type"] for r in rows] == ["candidate"] * 3 + ["negative_control"] and [r["shortlist_position"] for r in rows[:3]] == ["1", "2", "3"]
    assert [r["evidence_type"] for r in rows[:3]] == ["SEQUENCE_SIMILARITY", "SEQUENCE_SIMILARITY", "EXACT_SEQUENCE"] and rows[0]["predicted_ige_binding"]
    assert rows[0]["concentration_range"] == "" and rows[0]["decision"] == "" and rows[3]["sequence"] == "DLVDK"


def test_no_dossier_or_sheet_before_the_run_has_succeeded(local):
    client, tmp_path = local
    body = create(client).json()
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "results.json").write_text(json.dumps({"status": "running"}))
    assert client.get(f"/api/v1/jobs/food/{body['job_id']}/experiment-sheet").status_code == 404
    assert client.get(f"/api/v1/jobs/food/{body['job_id']}/dossiers/PEP-X").status_code == 404


# --- provenance check ---------------------------------------------------------------------------------------

def test_the_worker_fails_the_run_when_a_peptide_is_not_where_it_says(tmp_path, monkeypatch):
    from backend.worker_food import worker
    from food.pipeline import runner as food_runner
    from food.shared.candidate import FoodCandidate, ParentOrigin

    liar = FoodCandidate(id="PEP-000000000000", sequence="AAAAAA", origin_type="DIGESTION_DERIVED",
                         parents=[ParentOrigin(protein_id="P1:1-26", start_in_product=1, end_in_product=6, start_in_precursor=1, end_in_precursor=6)],
                         predictions={"screening": {}, "models": {}, "cleavage": {}, "ranking": {"rank": 1}})

    class FakeRunner:
        def __init__(self, config, status_path=None):
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Ctx:
                proteins: list | None = None

            self.ctx, self.candidate_stages = Ctx(), []

        def run(self):
            return [liar]

    monkeypatch.setattr(food_runner, "FoodPipelineRunner", FakeRunner)
    run_dir, config_path = write_config(tmp_path, FASTA)
    with pytest.raises(RuntimeError, match="provenance check failed"):
        worker.run_job(config_path)
    assert json.loads((run_dir / "results.json").read_text())["status"] == "failed"
    saved = json.loads((run_dir / "provenance_check.json").read_text())
    assert saved["checked"] == 1 and {i["check"] for i in saved["issues"]} >= {"id_matches_sequence", "sequence_at_parent_position"}


def test_the_results_report_the_provenance_check(finished):
    client, run_dir, job_id = finished
    (run_dir / "provenance_check.json").write_text(json.dumps({"checked": 3, "issues": []}))
    assert results(client, job_id).json()["provenance"] == {"checked": 3, "issues": []}


# --- PDF dossiers --------------------------------------------------------------------------------------------

def _pdf_pages(content: bytes) -> list[str]:
    import io

    from pypdf import PdfReader

    return [page.extract_text() for page in PdfReader(io.BytesIO(content)).pages]


def test_one_peptide_dossier_is_a_pdf_with_its_overview_shared_page_and_one_peptide_page(finished):
    client, run_dir, job_id = finished
    ids = [json.loads(line)["id"] for line in (run_dir / "candidates" / "s07_food_ranking_decision.jsonl").read_text(encoding="utf-8").splitlines()]
    response = client.get(f"/api/v1/jobs/food/{job_id}/dossiers/{ids[0]}?format=pdf")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf" and "attachment" in response.headers["content-disposition"]
    overview, shared, how_made, page = _pdf_pages(response.content)
    assert "How the list was made" in how_made and ids[0] in page and "AETYNPR" in page and "Decision (approve / hold / reject)" in page and "Not established" in shared and "set by the biology lead" in shared
    assert "Shortlist for biological review" in overview and "1 peptide." in overview


def test_the_shortlist_pdf_has_one_page_per_shortlisted_peptide_in_shortlist_order(finished):
    client, _, job_id = finished
    response = client.get(f"/api/v1/jobs/food/{job_id}/shortlist-dossiers.pdf")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    shortlist = results(client, job_id).json()["shortlist"]
    overview, shared, how_made, *pages = _pdf_pages(response.content)
    assert len(pages) == len(shortlist) == 3 and "3 peptides." in overview and "Shared information" in shared and "How the list was made" in how_made
    assert all(p["sequence"] in overview for p in shortlist) and len(_pdf_pages(response.content)) == 6
    for page, peptide in zip(pages, shortlist):
        assert peptide["id"] in page and f"Shortlist position {peptide['shortlist_position']}" in page


def test_no_pdf_before_the_run_has_succeeded_and_none_for_a_run_without_a_shortlist(local, finished):
    client, tmp_path = local
    body = create(client).json()
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "results.json").write_text(json.dumps({"status": "running"}))
    assert client.get(f"/api/v1/jobs/food/{body['job_id']}/shortlist-dossiers.pdf").status_code == 404
    _, finished_dir, finished_id = finished
    ranked = finished_dir / "candidates" / "s07_food_ranking_decision.jsonl"
    rows = [json.loads(line) for line in ranked.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["predictions"]["ranking"]["shortlist_position"] = None
    ranked.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    assert client.get(f"/api/v1/jobs/food/{finished_id}/shortlist-dossiers.pdf").status_code == 404


# --- the biological approval gate -----------------------------------------------------------------------------

def _ranked_ids(run_dir):
    return {json.loads(line)["sequence"]: json.loads(line)["id"] for line in (run_dir / "candidates" / "s07_food_ranking_decision.jsonl").read_text(encoding="utf-8").splitlines()}


def _rank_table(client, job_id):
    return [(p["id"], p["rank"], p["shortlist_position"]) for p in results(client, job_id).json()["peptides"]]


def test_a_decision_is_saved_in_the_run_review_log_and_read_back_and_the_ranking_does_not_change(finished):
    import csv
    import io

    client, run_dir, job_id = finished
    ids, base = _ranked_ids(run_dir), f"/api/v1/jobs/food/{job_id}"
    before = _rank_table(client, job_id)
    saved = client.put(f"{base}/review/{ids['IPAVFK']}", json={"decision": "approve", "reason": "Exact match to a primary-assay peptide.", "reviewer": "Hazara"})
    assert saved.status_code == 200
    record = saved.json()["record"]
    assert (record["peptide_id"], record["sequence"], record["decision"], record["reviewer"]) == (ids["IPAVFK"], "IPAVFK", "approve", "Hazara") and record["decided_at"].endswith("+00:00")
    assert saved.json()["gate"]["status"] == "IN_REVIEW" and saved.json()["gate"]["approved"] == [ids["IPAVFK"]]
    log = (run_dir / "review" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 and json.loads(log[0])["reason"] == "Exact match to a primary-assay peptide."

    current = client.get(f"{base}/review").json()
    assert [d["decision"] for d in current["decisions"]] == ["approve"] and len(current["history"]) == 1 and current["gate"]["rule"].startswith("Ranks are computed before the review")
    # the same decision shows everywhere, and the computed rank and shortlist are exactly what they were
    assert _rank_table(client, job_id) == before
    body = results(client, job_id).json()
    assert next(p for p in body["peptides"] if p["id"] == ids["IPAVFK"])["review"]["decision"] == "approve" and body["review_gate"]["pending"] == [ids["AETYNPR"], ids["SQQPPFVQEQQPVL"]]
    dossier = client.get(f"{base}/dossiers/{ids['IPAVFK']}").json()
    assert dossier["review"]["decision"] == "approve" and dossier["review"]["reason"] == "Exact match to a primary-assay peptide." and dossier["rank"] == 24
    assert dossier["status"].startswith("approved by the biology lead")
    rows = {r["sequence"]: r for r in csv.DictReader(io.StringIO(client.get(f"{base}/experiment-sheet").text)) if r["row_type"] == "candidate"}
    assert (rows["IPAVFK"]["decision"], rows["IPAVFK"]["reviewer"], rows["IPAVFK"]["comments"]) == ("approve", "Hazara", "Exact match to a primary-assay peptide.") and rows["AETYNPR"]["decision"] == ""
    overview = _pdf_pages(client.get(f"{base}/shortlist-dossiers.pdf").content)[0]
    assert "Approved" in overview and "Pending" in overview


def test_a_later_decision_replaces_the_current_one_and_the_history_keeps_both(finished):
    client, run_dir, job_id = finished
    ids, base = _ranked_ids(run_dir), f"/api/v1/jobs/food/{job_id}"
    client.put(f"{base}/review/{ids['AETYNPR']}", json={"decision": "hold", "reason": "Wait for the IgE follow-up.", "reviewer": "Hazara"})
    client.put(f"{base}/review/{ids['AETYNPR']}", json={"decision": "reject", "reason": "IgE binding confirmed.", "reviewer": "Hazara"})
    current = client.get(f"{base}/review").json()
    assert [(d["decision"], d["reason"]) for d in current["decisions"]] == [("reject", "IgE binding confirmed.")]
    assert [h["decision"] for h in current["history"]] == ["hold", "reject"] and current["gate"]["rejected"] == [ids["AETYNPR"]] and current["gate"]["on_hold"] == []


def test_the_gate_is_complete_when_every_shortlisted_peptide_has_a_decision_and_ignores_other_ranked_peptides(finished):
    client, run_dir, job_id = finished
    ids, base = _ranked_ids(run_dir), f"/api/v1/jobs/food/{job_id}"
    client.put(f"{base}/review/{ids['LKPTPEGD']}", json={"decision": "approve", "reviewer": "Hazara"})  # ranked but not shortlisted
    assert client.get(f"{base}/review").json()["gate"]["status"] == "AWAITING_REVIEW"
    for sequence in ("AETYNPR", "SQQPPFVQEQQPVL", "IPAVFK"):
        client.put(f"{base}/review/{ids[sequence]}", json={"decision": "hold", "reason": "", "reviewer": "Hazara"})
    gate = client.get(f"{base}/review").json()["gate"]
    assert gate["status"] == "REVIEW_COMPLETE" and gate["pending"] == [] and len(gate["on_hold"]) == 3 and gate["approved"] == [] and gate["shortlisted"] == 3


@pytest.mark.parametrize("body", [{"decision": "maybe", "reviewer": "H"}, {"decision": "approve"}, {"decision": "approve", "reviewer": ""}, {"reviewer": "H"}])
def test_a_malformed_decision_is_a_422_and_saves_nothing(finished, body):
    client, run_dir, job_id = finished
    ids = _ranked_ids(run_dir)
    response = client.put(f"/api/v1/jobs/food/{job_id}/review/{ids['IPAVFK']}", json=body)
    assert response.status_code == 422 and not (run_dir / "review" / "decisions.jsonl").exists()


def test_a_decision_on_an_unknown_peptide_or_an_unfinished_run_is_a_404(local, finished):
    client, _, job_id = finished
    assert client.put(f"/api/v1/jobs/food/{job_id}/review/PEP-NOT-RANKED", json={"decision": "approve", "reviewer": "H"}).status_code == 404
    other, tmp_path = local
    body = create(other).json()
    run_dir = tmp_path / "runs" / body["job_id"].rsplit("/", 1)[-1]
    (run_dir / "results.json").write_text(json.dumps({"status": "running"}))
    assert other.put(f"/api/v1/jobs/food/{body['job_id']}/review/PEP-X", json={"decision": "approve", "reviewer": "H"}).status_code == 404
    assert other.get(f"/api/v1/jobs/food/{body['job_id']}/review").status_code == 404

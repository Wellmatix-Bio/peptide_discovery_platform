# worker_immuno: run() scoring, failure reporting, and the refusal to overwrite a finished run.
import json

import pytest

from backend.worker_immuno import worker


def write_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"run_id": "7", "artifacts_dir": str(tmp_path / "art"), "model_store": str(tmp_path), "peptides": ["ALPMHIR"]})
    )
    return str(config), tmp_path / "art" / "runs" / "7"


class FakePredictor:
    def __init__(self, p):
        self.p = p

    def predict_batch(self, sequences, extractor):
        return [{"probability": self.p, "label": int(self.p >= 0.5), "members": {"m": self.p}, "in_length_range": len(s) <= 10} for s in sequences]


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(worker.ImmunoWorker, "_feature_extractor", lambda self: object())
    monkeypatch.setattr(
        worker.ImmunoWorker, "_predictors", lambda self: {c: FakePredictor(0.2 + 0.1 * i) for i, c in enumerate(worker.CYTOKINE_MODELS)}
    )


def test_run_scores_every_peptide_on_every_cytokine_in_order(tmp_path, fake_models):
    config, _ = write_config(tmp_path)
    cfg = worker.load_job_config(config)
    cfg.peptides = ["ALPMHIR", "AMKPWIQPKAMKPWIQPK"]
    results = worker.ImmunoWorker(cfg).run()
    assert [r["sequence"] for r in results] == cfg.peptides
    assert list(results[0]["predictions"]) == ["TNF-alpha", "IL-6", "IL-10", "IL-4", "IL-13", "IFN-gamma"]
    il6 = results[1]["predictions"]["IL-6"]
    assert il6["model"] == "il6_induction_predictor_v1" and il6["in_length_range"] is False and il6["label"] == 0


class CountingPredictor(FakePredictor):
    def __init__(self, p, sizes):
        super().__init__(p)
        self.sizes = sizes

    def predict_batch(self, sequences, extractor):
        self.sizes.append(len(sequences))
        return super().predict_batch(sequences, extractor)


def test_peptides_are_scored_in_batches_and_come_back_in_order(tmp_path, monkeypatch):
    config, _ = write_config(tmp_path)
    cfg = worker.load_job_config(config)
    cfg.peptides = [f"ALPMH{c}R" for c in "ACDEFGHIK"]  # 9 peptides
    cfg.batch_size = 4
    sizes, built = [], []
    monkeypatch.setattr(worker.ImmunoWorker, "_feature_extractor", lambda self: object())

    def predictors(self):
        built.append(1)
        return {c: CountingPredictor(0.3, sizes) for c in worker.CYTOKINE_MODELS}

    monkeypatch.setattr(worker.ImmunoWorker, "_predictors", predictors)
    progress = []
    results = worker.ImmunoWorker(cfg, on_progress=lambda *args: progress.append(args)).run()
    assert [r["sequence"] for r in results] == cfg.peptides
    assert set(sizes) == {4, 1} and sizes.count(4) == 6 * 2  # six models x two full batches, plus a final batch of one
    assert len(built) == 1  # the models are loaded once, not per batch
    assert progress == [(1, 3, 4), (2, 3, 8), (3, 3, 9)]


def test_status_stage_reports_the_batch_while_running(tmp_path, fake_models, monkeypatch):
    config, run_dir = write_config(tmp_path)
    seen = []
    real = worker.storage.write_text
    monkeypatch.setattr(worker.storage, "write_text", lambda path, text: (seen.append(text), real(path, text))[1])
    worker.run_job(config)
    running = [json.loads(t) for t in seen if '"running"' in t and "batch" in t]
    assert running and running[-1]["stage"] == "batch 1 of 1 (1/1 peptides)" and running[-1]["progress"] == 1.0


def test_a_model_failure_marks_the_run_failed(tmp_path, monkeypatch):
    config, run_dir = write_config(tmp_path)
    monkeypatch.setattr(worker.ImmunoWorker, "_feature_extractor", lambda self: object())

    def boom(self):
        raise FileNotFoundError("weights missing")

    monkeypatch.setattr(worker.ImmunoWorker, "_predictors", boom)
    with pytest.raises(FileNotFoundError):
        worker.run_job(config)
    status = json.loads((run_dir / "results.json").read_text())
    assert status["status"] == "failed" and "weights missing" in status["error"]
    assert not (run_dir / "immuno_results.json").exists()


def test_cytokine_models_are_real_packages():
    import importlib

    for package, cls in worker.CYTOKINE_MODELS.values():
        assert hasattr(importlib.import_module(f"model_store.{package}"), cls)


def test_a_successful_run_writes_results_and_status(tmp_path, monkeypatch):
    config, run_dir = write_config(tmp_path)
    monkeypatch.setattr(worker.ImmunoWorker, "run", lambda self: [{"sequence": s} for s in self.config.peptides])
    assert worker.run_job(config) == [{"sequence": "ALPMHIR"}]
    assert json.loads((run_dir / "results.json").read_text())["status"] == "success"
    assert json.loads((run_dir / "immuno_results.json").read_text()) == [{"sequence": "ALPMHIR"}]


def test_a_finished_run_is_not_overwritten_without_the_flag(tmp_path, monkeypatch):
    config, run_dir = write_config(tmp_path)
    monkeypatch.setattr(worker.ImmunoWorker, "run", lambda self: [])
    worker.run_job(config)
    with pytest.raises(worker.CompletedRunExists):
        worker.run_job(config)
    assert worker.run_job(config, overwrite=True) == []


def test_an_empty_peptide_list_is_rejected(tmp_path):
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"run_id": "7", "artifacts_dir": "a", "model_store": "m", "peptides": []}))
    with pytest.raises(ValueError):
        worker.load_job_config(str(config))


# --- where the weights come from: DEV_MODE decides ---------------------------------------------------------

def test_dev_mode_uses_local_weights_and_downloads_nothing(tmp_path, monkeypatch):
    from backend.worker_immuno.worker import weights_source
    from common import model_sync, storage
    from common.cytokine_ensemble import CytokineEnsemble

    monkeypatch.setattr(model_sync, "DEV_MODE", True)
    monkeypatch.setattr(model_sync, "VERTEX_MODEL_STORE", "gs://bucket/models")
    monkeypatch.setattr(model_sync, "_synced", set())
    monkeypatch.setattr(storage, "download_dir", lambda *a: pytest.fail("DEV_MODE must not download"))
    assert "DEV_MODE" in weights_source()

    code_dir = tmp_path / "pkg_v1"
    weights = model_sync.weights_dir_for(code_dir)
    weights.mkdir(parents=True)
    (weights / "info.json").write_text(json.dumps({"training": {"length_min": 8, "length_max": 20}}))

    class Local(CytokineEnsemble):
        MEMBERS = ()

    Local(code_dir, weights)._load()


def test_production_downloads_each_package_from_the_bucket(tmp_path, monkeypatch):
    from backend.worker_immuno.worker import weights_source
    from common import model_sync, storage
    from common.cytokine_ensemble import CytokineEnsemble

    monkeypatch.setattr(model_sync, "DEV_MODE", False)
    monkeypatch.setattr(model_sync, "VERTEX_MODEL_STORE", "gs://bucket/models")
    monkeypatch.setattr(model_sync, "_synced", set())
    fetched = []

    def fake_download(prefix, destination):
        fetched.append((prefix, destination))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "info.json").write_text(json.dumps({"training": {"length_min": 8, "length_max": 20}}))

    monkeypatch.setattr(storage, "download_dir", fake_download)
    assert weights_source() == "weights downloaded from gs://bucket/models"

    code_dir = tmp_path / "il6_induction_predictor_v1"

    class Remote(CytokineEnsemble):
        MEMBERS = ()

    Remote(code_dir, model_sync.weights_dir_for(code_dir))._load()
    assert fetched == [("gs://bucket/models/il6_induction_predictor_v1", model_sync.weights_dir_for(code_dir))]


def test_production_without_a_real_bucket_fails_before_any_scoring(monkeypatch):
    from backend.worker_immuno.worker import weights_source
    from common import model_sync

    monkeypatch.setattr(model_sync, "DEV_MODE", False)
    monkeypatch.setattr(model_sync, "VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store")
    with pytest.raises(RuntimeError, match="VERTEX_MODEL_STORE is not set"):
        weights_source()


def test_run_checks_the_weights_source_before_loading_anything(tmp_path, monkeypatch):
    from common import model_sync

    config, run_dir = write_config(tmp_path)
    monkeypatch.setattr(model_sync, "DEV_MODE", False)
    monkeypatch.setattr(model_sync, "VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store")
    monkeypatch.setattr(worker.ImmunoWorker, "_feature_extractor", lambda self: pytest.fail("must fail before loading models"))
    with pytest.raises(RuntimeError):
        worker.run_job(config)
    assert json.loads((run_dir / "results.json").read_text())["status"] == "failed"

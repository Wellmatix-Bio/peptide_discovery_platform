"""Exercise real runner/boundary I/O with lightweight stages and an in-memory GCS client."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from backend.worker_e2e import worker
from common import storage
from schemas.candidate import Candidate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    # Avoid eagerly importing GPU predictors; execute the actual base and runner.
    package = ModuleType("pipeline")
    package.__path__ = [str(ROOT / "src/pipeline")]
    monkeypatch.setitem(sys.modules, "pipeline", package)
    features = ModuleType("pipeline.feature_extractor")

    class FeatureExtractor:
        def save(self, path):
            storage.write_text(storage.join(path, "test.json"), "{}")

    features.FeatureExtractor = FeatureExtractor
    monkeypatch.setitem(sys.modules, features.__name__, features)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    base = load("pipeline.base", "src/pipeline/base.py")
    seen = []

    class Setup(base.SetupStage):
        name = "s01_therapeutic_product_brief"

        def run(self, config, ctx):
            seen.append(self.name)
            return SimpleNamespace(model_dump=lambda **kw: config.params["brief"])

    class Disabled(base.SetupStage):
        name = "s02_wound_biology_and_targets"

        def run(self, config, ctx):
            raise AssertionError("Stage 2 must not run")

    class Enrich(base.CandidateStage):
        name = "fake_enrich"
        requires = {"sequence"}
        produces = {"score"}

        def run(self, candidates, config, ctx):
            seen.append(self.name)
            assert ctx.objectives is None
            assert ctx.seed == 0
            for c in candidates:
                c.predictions["score"] = 0.9
            return candidates

    class Finish(base.CandidateStage):
        name = "fake_finish"
        requires = {"score"}

        def run(self, candidates, config, ctx):
            seen.append(self.name)
            if config.params.get("fail"):
                raise RuntimeError("test failure")
            return [] if config.params.get("empty") else candidates

    registry = ModuleType("registry")
    registry.build_stages = lambda config: (
        [c() for c in (Setup, Disabled) if config.for_stage(c.name).enabled],
        [c() for c in (Enrich, Finish) if config.for_stage(c.name).enabled],
    )
    monkeypatch.setitem(sys.modules, "registry", registry)
    load("runner", "src/runner.py")
    # Only seeding is required; model inference isn't part of these tests.
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None)
    )
    from common import env, model_sync

    monkeypatch.setattr(env, "DEV_MODE", env.DEV_MODE)
    monkeypatch.setattr(model_sync, "DEV_MODE", model_sync.DEV_MODE)
    monkeypatch.setattr(model_sync, "VERTEX_MODEL_STORE", model_sync.VERTEX_MODEL_STORE)
    monkeypatch.setattr(model_sync, "_synced", set())
    monkeypatch.setenv("DEV_MODE", "false")
    monkeypatch.setenv("VERTEX_MODEL_STORE", "local")
    return seen


def config_file(tmp_path, *, artifacts=None, finish=None):
    seeds = tmp_path / "seeds.fasta"
    seeds.write_text(">one\nKLLKLLKK\n", encoding="utf-8")
    data = {
        "run_id": "test",
        "seed": 0,
        "seed_candidates_path": str(seeds),
        "artifacts_dir": artifacts or str(tmp_path / "artifacts"),
        "model_store": str(tmp_path / "models"),
        "stages": {
            "s01_therapeutic_product_brief": {
                "params": {
                    "brief": json.loads(
                        (ROOT / "src/backend/api_e2e/example_request.json").read_text()
                    )["stages"]["s01_therapeutic_product_brief"]["brief"]
                }
            },
            "fake_finish": {"params": finish or {}},
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("empty", [False, True])
def test_complete_local_run(runtime, tmp_path, empty):
    path = config_file(tmp_path, finish={"empty": empty})
    result = worker.run_job(path)
    run = tmp_path / "artifacts/runs/test"
    assert len(result) == (0 if empty else 1)
    assert json.loads((run / "candidates_final.json").read_text()) == [
        c.model_dump() for c in result
    ]
    assert json.loads((run / "run_context.json").read_text())["objectives"] is None
    assert json.loads((run / "results.json").read_text())["status"] == "success"
    assert (run / "stats_test.txt").exists()
    assert (run / "config_snapshot.yaml").exists()
    assert (run / "feature_cache/test.json").exists()
    assert runtime == ["s01_therapeutic_product_brief", "fake_enrich", "fake_finish"]


def test_failure_propagates_and_preserves_boundary(runtime, tmp_path):
    with pytest.raises(RuntimeError, match="test failure"):
        worker.run_job(config_file(tmp_path, finish={"fail": True}))
    run = tmp_path / "artifacts/runs/test"
    assert json.loads((run / "results.json").read_text())["status"] == "failed"
    assert (run / "candidates/fake_enrich.jsonl").exists()
    assert not (run / "candidates_final.json").exists()


def test_gcs_config_and_outputs(runtime, tmp_path, monkeypatch):
    objects = {}

    class Blob:
        def __init__(self, key):
            self.key = key

        def exists(self):
            return self.key in objects

        def upload_from_string(self, data):
            objects[self.key] = data

        def download_as_text(self, encoding="utf-8"):
            return objects[self.key].decode(encoding)

    client = SimpleNamespace(
        bucket=lambda bucket: SimpleNamespace(
            blob=lambda name: Blob(f"gs://{bucket}/{name}")
        )
    )
    monkeypatch.setattr(storage, "_gcs_client", lambda: client)
    config = config_file(tmp_path, artifacts="gs://bucket/artifacts")
    storage.write_text("gs://bucket/config.json", Path(config).read_text())
    worker.run_job("gs://bucket/config.json")
    prefix = "gs://bucket/artifacts/runs/test/"
    for suffix in (
        "config_snapshot.yaml",
        "audit_log.jsonl",
        "candidates_final.json",
        "stats_test.txt",
        "run_context.json",
        "results.json",
        "candidates/fake_finish.jsonl",
    ):
        assert prefix + suffix in objects


def test_stage2_default_disabled(tmp_path):
    config = worker.load_job_config(config_file(tmp_path))
    assert not config.for_stage("s02_wound_biology_and_targets").enabled


def test_yaml_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("run_id: test\nstages: {}\n")
    with pytest.raises(json.JSONDecodeError):
        worker.load_job_config(str(path))


def test_wait_for_config_publication(tmp_path, monkeypatch):
    body = Path(config_file(tmp_path)).read_text()
    reads = []

    def read(path):
        reads.append(path)
        if len(reads) == 1:
            raise FileNotFoundError(path)
        return body

    monkeypatch.setattr(worker.storage, "read_text", read)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    assert worker.load_job_config("pending.json", wait_seconds=10).run_id == "test"
    assert len(reads) == 2


def test_wait_for_config_timeout(monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(worker.storage, "read_text", missing)
    with pytest.raises(FileNotFoundError):
        worker.load_job_config("pending.json", wait_seconds=0)


# ----------------------------------------------------------------------
# Refusing to overwrite a run that already finished.
#
# A config carries run_id AND artifacts_dir, so replaying one writes into that run's directory.
# On 2026-10-02 a worker started outside Vertex from an existing config did exactly that: the
# "running" status it wrote at startup replaced a completed run's results.json, and its 11 final
# candidates stopped being readable through the API although candidates_final.json was untouched
# beside them. docs/BASELINE.md has the investigation.
# ----------------------------------------------------------------------


def test_a_second_run_refuses_to_overwrite_a_finished_one(runtime, tmp_path):
    path = config_file(tmp_path)
    worker.run_job(path)
    run = tmp_path / "artifacts/runs/test"
    before = (run / "results.json").read_text()
    final_before = (run / "candidates_final.json").read_text()

    with pytest.raises(worker.CompletedRunExists) as refused:
        worker.run_job(path)

    # The refusal must name what it found and what to do, not merely fail.
    assert "candidates_final.json" in str(refused.value)
    assert "--overwrite" in str(refused.value)
    # And it must have changed NOTHING. This is the whole point: the first run survives intact.
    assert (run / "results.json").read_text() == before
    assert (run / "candidates_final.json").read_text() == final_before


def test_the_refusal_does_not_write_a_failed_status(runtime, tmp_path):
    """The guard sits before the try block on purpose. That block's handler writes a "failed"
    status to the same results.json, so a guard raised inside it would destroy the very run it
    exists to protect."""
    path = config_file(tmp_path)
    worker.run_job(path)
    run = tmp_path / "artifacts/runs/test"

    with pytest.raises(worker.CompletedRunExists):
        worker.run_job(path)

    assert json.loads((run / "results.json").read_text())["status"] == "success"


def test_overwrite_is_still_possible_when_asked_for(runtime, tmp_path):
    path = config_file(tmp_path)
    worker.run_job(path)
    result = worker.run_job(path, overwrite=True)
    run = tmp_path / "artifacts/runs/test"
    assert json.loads((run / "results.json").read_text())["status"] == "success"
    assert len(result) == 1


def test_an_unfinished_run_is_not_blocked(runtime, tmp_path):
    """A Vertex retry restarts a run that did NOT finish, so neither signal is present and the
    worker must proceed. Blocking this would turn a recoverable restart into a dead job."""
    path = config_file(tmp_path, finish={"fail": True})
    with pytest.raises(RuntimeError, match="test failure"):
        worker.run_job(path)
    run = tmp_path / "artifacts/runs/test"
    assert json.loads((run / "results.json").read_text())["status"] == "failed"
    assert not (run / "candidates_final.json").exists()

    # The retry succeeds, which it could not do if a failed run counted as a finished one.
    worker.run_job(config_file(tmp_path))
    assert json.loads((run / "results.json").read_text())["status"] == "success"


def test_a_fresh_run_directory_is_not_blocked(runtime, tmp_path):
    worker.run_job(config_file(tmp_path))
    assert json.loads(
        (tmp_path / "artifacts/runs/test/results.json").read_text()
    )["status"] == "success"


def test_a_success_status_alone_is_enough_to_refuse(tmp_path):
    """The two signals are independent. A run whose candidates_final.json was removed, but whose
    own results.json still says it succeeded, is still a finished run."""
    run = tmp_path / "artifacts/runs/test"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"run_id": "test", "status": "success"}))
    assert "success" in (worker.completed_run_reason(str(run)) or "")


def test_an_unreadable_status_is_not_treated_as_a_finished_run(tmp_path):
    """Refusing on a file that cannot be parsed would block recovery from a half-written run."""
    run = tmp_path / "artifacts/runs/test"
    run.mkdir(parents=True)
    (run / "results.json").write_text("{not json")
    assert worker.completed_run_reason(str(run)) is None


def test_the_cli_exposes_the_override(tmp_path):
    assert worker._parse_args(["--config", "c.json"]).overwrite is False
    assert worker._parse_args(["--config", "c.json", "--overwrite"]).overwrite is True

"""The three worker entry points are entry points, not three workers.

worker_metabolic/worker.py and worker_cosmetic/worker.py were byte-identical to each other once
you swapped the pipeline word, and both were worker_e2e/worker.py minus its `--overwrite` flag --
which is what a copy costs: a fix landed in one and not the others, and nothing noticed.

These tests pin the shape that replaced them: one implementation, three modules that name their
pipeline. Each assertion is about the thing a future copy-paste would break.
"""

from __future__ import annotations

import pytest

from backend.worker import worker as shared
from backend.worker_cosmetic import worker as cosmetic
from backend.worker_e2e import worker as wound
from backend.worker_metabolic import worker as metabolic

ENTRY_POINTS = ((wound, "wound_healing"), (metabolic, "metabolic"), (cosmetic, "cosmetic"))


@pytest.mark.parametrize(("module", "pipeline"), ENTRY_POINTS)
def test_each_entry_point_names_its_own_pipeline(module, pipeline: str) -> None:
    assert module.PIPELINE == pipeline
    assert pipeline in shared.PIPELINES


@pytest.mark.parametrize(("module", "pipeline"), ENTRY_POINTS)
def test_every_entry_point_offers_overwrite(module, pipeline: str) -> None:
    """The flag that went missing from two of the three copies. It guards a finished run's
    results, so a pipeline without it has no way to say "yes, replace that run"."""
    assert module._parse_args(["--config", "c.json"]).overwrite is False
    assert module._parse_args(["--config", "c.json", "--overwrite"]).overwrite is True


@pytest.mark.parametrize(("module", "pipeline"), ENTRY_POINTS)
def test_the_run_loop_is_shared_not_reimplemented(module, pipeline: str) -> None:
    """Identity, not behaviour: these names must BE the shared ones, so monkeypatching
    `worker.storage` in a test still reaches the code that runs."""
    assert module.storage is shared.storage
    assert module.time is shared.time
    assert module.completed_run_reason is shared.completed_run_reason
    assert module.CompletedRunExists is shared.CompletedRunExists


@pytest.mark.parametrize(("module", "pipeline"), ENTRY_POINTS)
def test_the_pipeline_reaches_the_config_validator(monkeypatch, module, pipeline: str) -> None:
    """The one thing each entry point exists to do. Without it a metabolic config would be
    validated against the wound-healing stage names and rejected."""
    seen: list[str] = []
    monkeypatch.setattr(
        shared, "validate_job_config", lambda payload, pipeline: seen.append(pipeline)
    )
    monkeypatch.setattr(shared.storage, "read_text", lambda path: "{}")
    module.load_job_config("anywhere.json")
    assert seen == [pipeline]


def test_overriding_the_pipeline_on_the_command_line_wins(monkeypatch) -> None:
    """--pipeline exists so one image can run another pipeline's config without a rebuild. The
    entry point's own PIPELINE is the default, not a lock."""
    seen: list[str] = []
    monkeypatch.setattr(
        shared, "run_job", lambda config, wait, overwrite, pipeline: seen.append(pipeline)
    )
    shared.main(["--config", "c.json", "--pipeline", "cosmetic"], pipeline="metabolic")
    assert seen == ["cosmetic"]
    seen.clear()
    shared.main(["--config", "c.json"], pipeline="metabolic")
    assert seen == ["metabolic"]

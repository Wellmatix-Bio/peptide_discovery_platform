"""Stage 1: turning a brief into a `Brief`, and the DEV_MODE fork in how it finds one.

Stage 1 is the first thing a run does and the only stage that reads the client's brief, so every
later stage inherits whatever it accepts. Two things here are worth pinning:

- **The DEV_MODE fork is resolved at IMPORT time.** `stage.py` does `from common.env import
  DEV_MODE`, binding the value into the module, so setting the environment variable after import
  changes nothing. A test that only set `os.environ` would pass against either branch while
  exercising one; these patch the module attribute, which is what the code actually reads.
- **An empty brief is treated as no brief.** `if not brief` is falsy for `{}`, so a request
  carrying an empty object is refused with the same error as one carrying nothing. That is
  defensible -- an empty brief cannot produce a `Brief`, whose length fields are required -- and
  it is not obvious from the message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.base import StageError  # noqa: E402
from pipeline.s01_brief import stage as s01  # noqa: E402
from schemas.brief import Brief  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

VALID = {
    "wound_context": ["chronic", "infected"],
    "desired_functions": ["antimicrobial"],
    "pathogens": ["Staphylococcus_aureus"],
    "min_length": 8,
    "max_length": 30,
}


def run(config_params: dict, *, dev_mode: bool, monkeypatch) -> Brief:
    """DEV_MODE is read from the module, not the environment -- see the module docstring."""
    monkeypatch.setattr(s01, "DEV_MODE", dev_mode)
    return s01.Stage1().run(StageConfig(params=config_params), ctx=None)


# -- production path: the brief arrives inline -------------------------------


def test_an_inline_brief_becomes_a_brief_object(monkeypatch):
    got = run({"brief": VALID}, dev_mode=False, monkeypatch=monkeypatch)
    assert isinstance(got, Brief)
    assert got.wound_context == ["chronic", "infected"]
    assert got.min_length == 8 and got.max_length == 30


def test_a_missing_brief_is_refused_by_name(monkeypatch):
    with pytest.raises(StageError, match="brief is not specified"):
        run({}, dev_mode=False, monkeypatch=monkeypatch)


def test_an_empty_brief_is_refused_the_same_way(monkeypatch):
    """`if not brief` is falsy for {}. Pinned because the message says "not specified" for a brief
    that WAS specified, just emptily -- and because a `Brief` cannot be built from {} anyway."""
    with pytest.raises(StageError, match="brief is not specified"):
        run({"brief": {}}, dev_mode=False, monkeypatch=monkeypatch)


def test_an_invalid_brief_fails_validation_rather_than_being_accepted(monkeypatch):
    """min_length has gt=0, so the schema -- not this stage -- rejects it. The point is that it
    IS rejected: the stage does not sanitise a bad value into a plausible one."""
    with pytest.raises(Exception) as refused:
        run({"brief": {**VALID, "min_length": 0}}, dev_mode=False, monkeypatch=monkeypatch)
    assert not isinstance(refused.value, StageError), "validation should surface, not be swallowed"


def test_the_production_path_ignores_brief_path(monkeypatch, tmp_path):
    """A brief_path in a production request must not be read. It would be a file path chosen by
    the client, pointing at the server's filesystem."""
    planted = tmp_path / "brief.json"
    planted.write_text(json.dumps({**VALID, "min_length": 42}))
    with pytest.raises(StageError, match="brief is not specified"):
        run({"brief_path": str(planted)}, dev_mode=False, monkeypatch=monkeypatch)


# -- DEV_MODE path: the brief is read from a file ----------------------------


def test_dev_mode_loads_the_brief_from_disk(monkeypatch, tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(VALID))
    got = run({"brief_path": str(path)}, dev_mode=True, monkeypatch=monkeypatch)
    assert isinstance(got, Brief)
    assert got.pathogens == ["Staphylococcus_aureus"]


def test_dev_mode_without_a_path_is_refused_by_name(monkeypatch):
    with pytest.raises(StageError, match="brief_path is not specified"):
        run({}, dev_mode=True, monkeypatch=monkeypatch)


def test_dev_mode_with_a_missing_file_says_so(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="Brief file not found"):
        run({"brief_path": str(tmp_path / "nope.json")}, dev_mode=True, monkeypatch=monkeypatch)


def test_dev_mode_ignores_an_inline_brief(monkeypatch, tmp_path):
    """The two paths are exclusive. In DEV_MODE an inline brief is not a fallback, so a config
    carrying only `brief` fails rather than quietly using it."""
    with pytest.raises(StageError, match="brief_path is not specified"):
        run({"brief": VALID}, dev_mode=True, monkeypatch=monkeypatch)


def test_the_two_paths_produce_the_same_brief(monkeypatch, tmp_path):
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(VALID))
    from_disk = run({"brief_path": str(path)}, dev_mode=True, monkeypatch=monkeypatch)
    inline = run({"brief": VALID}, dev_mode=False, monkeypatch=monkeypatch)
    assert from_disk == inline

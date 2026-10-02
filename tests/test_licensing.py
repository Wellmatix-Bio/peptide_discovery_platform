"""The project's licence posture, as checks rather than as prose.

This project is Apache-2.0. Two dependencies it can use are copyleft -- propy3 is GPL-2.0-only and
s4pred is GPL-3.0 -- and they are mutually incompatible as well as incompatible with shipping an
Apache-2.0 work that requires them. They are therefore OPTIONAL: not required, not vendored, not
installed by default.

That is easy to undo by accident. Someone adds `propy3` back to requirements.txt to make a test
pass, or moves an optional import to the top of a module, and the repository quietly stops being
distributable under the licence it claims. These tests fail when that happens.

They do not check that the code is correct. They check that it is still legal to publish.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COPYLEFT = ("propy3", "propy", "s4pred")


def test_the_project_states_a_licence():
    """With no LICENCE file, published code is 'all rights reserved' and nobody may use it."""
    licence = ROOT / "LICENSE"
    assert licence.is_file(), "LICENSE is missing"
    text = licence.read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text
    assert (ROOT / "NOTICE").is_file(), "NOTICE is missing"


def test_no_copyleft_dependency_is_required():
    """requirements.txt is what the Docker images install. A copyleft entry here makes every
    built image a combined work subject to that licence."""
    required = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            required.append(line.lower())
    for name in COPYLEFT:
        assert not any(
            entry.startswith(name) for entry in required
        ), f"{name} is copyleft and must not be a required dependency; see docs/LICENSING.md"


def test_copyleft_is_an_optional_extra_not_a_core_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = [d.lower() for d in data["project"].get("dependencies", [])]
    for name in COPYLEFT:
        assert not any(d.startswith(name) for d in core), f"{name} must not be a core dependency"

    extras = data["project"].get("optional-dependencies", {})
    offered = [d.lower() for group in extras.values() for d in group]
    assert any(
        d.startswith("propy3") for d in offered
    ), "propy3 should still be offered as an extra, so the feature remains reachable"


@pytest.mark.parametrize(
    "module,flag",
    [
        ("pipeline.feature_extractor", "PROPY_AVAILABLE"),
        ("pipeline.s05_physchem_screening.stage", "S4PRED_AVAILABLE"),
    ],
)
def test_the_optional_import_does_not_break_the_module(module: str, flag: str):
    """The whole point: these modules must import whether or not the copyleft package is there.

    s4pred previously did not, and because pipeline/__init__.py imports stage 5 eagerly, a missing
    s4pred made the entire pipeline package unimportable and 47 tests uncollectable.
    """
    sys.path.insert(0, str(ROOT / "src"))
    imported = __import__(module, fromlist=[flag])
    assert hasattr(imported, flag), f"{module} should expose {flag}"
    assert isinstance(getattr(imported, flag), bool)


def test_an_unavailable_screen_is_never_reported_as_a_pass():
    """Stage 8's rollup. A null aggregation score used to read `else "pass"`, so a screen that
    never ran -- a peptide too short for the model, or propy3 absent -- was recorded as having
    passed it. The verdict must say the check did not run."""
    source = (ROOT / "src/pipeline/s08_safety_developability/stage.py").read_text(encoding="utf-8")
    assert "not_screened" in source, "stage 8 must distinguish an unrun screen from a pass"
    assert (
        'if aggregation_score is None:\n            properties["aggregation_tendency"] = "not_screened"'
        in source
    ), "a null aggregation score must not fall through to a pass"
    assert (
        '"flag" in properties.values() or "not_screened" in properties.values()' in source
    ), "an unrun screen must make the overall verdict a flag, not a pass"


def test_the_licensing_document_exists_and_names_both_dependencies():
    text = (ROOT / "docs/LICENSING.md").read_text(encoding="utf-8")
    for name in ("propy3", "s4pred", "GPL-2.0-only", "GPL-3.0", "Apache-2.0"):
        assert name in text, f"docs/LICENSING.md should mention {name}"

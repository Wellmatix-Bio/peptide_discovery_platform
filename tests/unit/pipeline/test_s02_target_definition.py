"""Stage 2: scoring wound-biology deficits from the brief, by rules loaded from a YAML file.

Stage 2 is force-disabled on the deployed path, so these are tests of a component rather than of
something a user currently reaches. They are worth having anyway: the stage is the one place a
brief's words are turned into weighted severities, and the rule file that drives it is editable
data, so the failure modes are the file's as much as the code's.

Two sharp edges are pinned below rather than fixed, because fixing either changes scoring
behaviour and that is the model owner's call:

- `contains` on a STRING field does substring matching, so a rule looking for "acute" fires on
  "subacute". Brief vocabulary fields are lists, where `in` is exact, so this is only reachable
  through a rule naming a scalar field -- but nothing refuses such a rule.
- Severity is capped above and NOT below. A rule file with negative deltas can drive a deficit
  below zero and the cap will not catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.base import StageError  # noqa: E402
from pipeline.s02_target_definition.stage import Stage2  # noqa: E402
from schemas.brief import Brief  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

BRIEF = Brief(
    wound_context=["chronic", "infected"],
    desired_functions=["antimicrobial"],
    pathogens=["Staphylococcus_aureus"],
    min_length=8,
    max_length=30,
)

RULES = {
    "deficits": {
        "infection_burden": {
            "base": 0.1,
            "cap": 1.0,
            "contributions": [
                {"op": "contains", "field": "wound_context", "value": "infected",
                 "delta": 0.5, "evidence": "strong"},
                {"op": "contains", "field": "wound_context", "value": "necrotic",
                 "delta": 0.4, "evidence": "moderate"},
                {"op": "nonempty", "field": "pathogens", "delta": 0.2, "evidence": "weak"},
            ],
        },
        "untouched": {"base": 0.3, "cap": 1.0},
    }
}


class FakeCtx:
    def __init__(self, brief=BRIEF):
        self.brief = brief


def rules_file(tmp_path, rules=None) -> str:
    path = tmp_path / "deficits.yaml"
    path.write_text(yaml.safe_dump(rules if rules is not None else RULES))
    return str(path)


def run(tmp_path, rules=None, brief=BRIEF):
    config = StageConfig(params={"deficit_rules_path": rules_file(tmp_path, rules)})
    return Stage2().run(config, FakeCtx(brief))


# -- the operators -----------------------------------------------------------


@pytest.mark.parametrize(
    "contribution,context,expected",
    [
        ({"op": "contains", "field": "f", "value": "a"}, {"f": ["a", "b"]}, True),
        ({"op": "contains", "field": "f", "value": "z"}, {"f": ["a", "b"]}, False),
        ({"op": "contains_any", "field": "f", "value": ["z", "b"]}, {"f": ["a", "b"]}, True),
        ({"op": "contains_any", "field": "f", "value": ["y", "z"]}, {"f": ["a", "b"]}, False),
        ({"op": "nonempty", "field": "f"}, {"f": ["a"]}, True),
        ({"op": "nonempty", "field": "f"}, {"f": []}, False),
    ],
)
def test_each_operator_decides_as_documented(contribution, context, expected):
    assert Stage2().condition_met(contribution, context) is expected


def test_a_field_the_brief_does_not_have_is_absent_not_an_error():
    """`context.get(field, [])`. A rule naming a field that no longer exists stops firing rather
    than crashing the run -- quiet, but the alternative is a rule file taking down every run."""
    assert Stage2().condition_met({"op": "nonempty", "field": "gone"}, {}) is False
    assert Stage2().condition_met({"op": "contains", "field": "gone", "value": "x"}, {}) is False


def test_an_unknown_operator_is_refused_rather_than_ignored():
    """A typo in a rule file must not silently mean "never fires"."""
    with pytest.raises(StageError, match="Unknown op"):
        Stage2().condition_met({"op": "contians", "field": "f", "value": "a"}, {"f": ["a"]})


def test_contains_on_a_string_field_matches_substrings():
    """PINNED, NOT ENDORSED. On a list, `in` is exact. On a string it is a substring test, so a
    rule for "acute" fires on "subacute". Brief vocabulary fields are lists, so this needs a rule
    naming a scalar field -- but nothing rejects such a rule, and the result would be a deficit
    scored on a word that is not there."""
    assert Stage2().condition_met(
        {"op": "contains", "field": "note", "value": "acute"}, {"note": "subacute wound"}
    ) is True


# -- loading the rules -------------------------------------------------------


def test_a_missing_rules_path_is_refused_by_name(tmp_path):
    with pytest.raises(StageError, match="deficit_rules_path is not specified"):
        Stage2().run(StageConfig(params={}), FakeCtx())


def test_an_unreadable_rules_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(StageError, match=str(missing)):
        Stage2().run(StageConfig(params={"deficit_rules_path": str(missing)}), FakeCtx())


def test_malformed_yaml_is_refused_rather_than_half_read(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("deficits: [unclosed\n")
    with pytest.raises(StageError, match="Failed to load deficit rules"):
        Stage2().run(StageConfig(params={"deficit_rules_path": str(path)}), FakeCtx())


def test_running_without_a_brief_says_stage_1_must_run_first(tmp_path):
    with pytest.raises(StageError, match="Stage 1"):
        run(tmp_path, brief=None)


# -- scoring -----------------------------------------------------------------


def test_severities_accumulate_the_contributions_that_fired(tmp_path):
    got = run(tmp_path)
    # base 0.1 + "infected" 0.5 + non-empty pathogens 0.2; "necrotic" is absent.
    assert got.deficits["infection_burden"].severity == pytest.approx(0.8)


def test_a_deficit_with_no_contributions_keeps_its_base(tmp_path):
    assert run(tmp_path).deficits["untouched"].severity == pytest.approx(0.3)


def test_the_cap_limits_a_deficit(tmp_path):
    capped = {"deficits": {"d": {"base": 0.9, "cap": 1.0, "contributions": [
        {"op": "contains", "field": "wound_context", "value": "infected",
         "delta": 0.5, "evidence": "strong"}]}}}
    assert run(tmp_path, capped).deficits["d"].severity == pytest.approx(1.0)


def test_severity_has_no_floor(tmp_path):
    """PINNED, NOT ENDORSED. `min(severity, cap)` bounds above only, so negative deltas can drive
    a deficit below zero and nothing notices. A rule file is data; this is reachable by editing
    one."""
    negative = {"deficits": {"d": {"base": 0.1, "cap": 1.0, "contributions": [
        {"op": "contains", "field": "wound_context", "value": "infected",
         "delta": -0.9, "evidence": "weak"}]}}}
    assert run(tmp_path, negative).deficits["d"].severity < 0


def test_only_the_contributions_that_fired_are_described(tmp_path):
    drivers = run(tmp_path).deficits["infection_burden"].drivers
    assert len(drivers) == 2, "infected and non-empty pathogens fired; necrotic did not"
    assert any("infected" in d for d in drivers)
    assert not any("necrotic" in d for d in drivers)


def test_a_driver_states_its_condition_weight_and_evidence(tmp_path):
    drivers = run(tmp_path).deficits["infection_burden"].drivers
    infected = next(d for d in drivers if "infected" in d)
    assert "wound_context contains 'infected'" in infected
    assert "+0.50" in infected and "strong" in infected


@pytest.mark.parametrize(
    "op,expected",
    [("nonempty", "pathogens is non-empty"), ("contains_any", "contains any of")],
)
def test_each_operator_describes_itself_readably(op, expected):
    rendered = Stage2().describe(
        {"op": op, "field": "pathogens", "value": ["a"], "delta": 0.2, "evidence": "weak"}
    )
    assert expected in rendered


def test_severity_is_rounded_for_reporting(tmp_path):
    thirds = {"deficits": {"d": {"base": 0.3333333, "cap": 1.0, "contributions": []}}}
    assert run(tmp_path, thirds).deficits["d"].severity == 0.33

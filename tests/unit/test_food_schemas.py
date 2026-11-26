"""The food pipeline's candidate, `FoodCandidate`, a subclass of `Candidate`.

The shared `Candidate` is left exactly as it was, so the other pipelines are untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schemas.candidate import Candidate  # noqa: E402
from schemas.food import FoodCandidate, ParentOrigin  # noqa: E402


def parent(**kw) -> ParentOrigin:
    return ParentOrigin(**{"protein_id": "P02754:17-178", **kw})


def candidate(**kw) -> FoodCandidate:
    return FoodCandidate(**{"id": "PEP-1", "sequence": "AAK", "origin_type": "DIGESTION_DERIVED", "parents": [parent()], **kw})


class TestFoodCandidate:
    def test_it_is_a_candidate(self):
        c = candidate()
        assert isinstance(c, Candidate)
        assert (c.id, c.sequence, c.predictions) == ("PEP-1", "AAK", {})

    def test_defaults_are_theoretical_peptide(self):
        c = candidate()
        assert (c.evidence_class, c.material_class) == ("THEORETICAL", "peptide")

    @pytest.mark.parametrize("field,value", [("origin_type", "MADE_UP"), ("evidence_class", "PREDICTED"), ("material_class", "powder")])
    def test_unknown_labels_are_refused(self, field, value):
        with pytest.raises(ValidationError):
            candidate(**{field: value})

    def test_a_digested_peptide_must_name_a_parent(self):
        with pytest.raises(ValidationError, match="parent"):
            candidate(parents=[])

    def test_a_designed_peptide_may_have_no_parent_yet(self):
        assert candidate(origin_type="SYNTHETIC_DESIGN", parents=[]).parents == []

    @pytest.mark.parametrize(
        "flags,expected",
        [
            ([True], True),
            ([False], False),
            ([None], None),  # not assessed is not "not an allergen source"
            ([False, True], True),
            ([False, None], None),
            ([False, False], False),
        ],
    )
    def test_allergen_flag_is_true_false_or_unknown(self, flags, expected):
        c = candidate(parents=[parent(protein_id=f"P{i}", allergen_source=f) for i, f in enumerate(flags)])
        assert c.any_parent_allergen_source is expected

    def test_no_parents_means_unknown(self):
        assert candidate(origin_type="LITERATURE_KNOWN", parents=[]).any_parent_allergen_source is None

    def test_it_survives_a_save_and_reload_as_a_food_candidate(self):
        c = candidate(parents=[parent(allergen_source=True, enzymes=["trypsin"], ph=7.5)], predictions={"x": 1})
        back = FoodCandidate.model_validate_json(c.model_dump_json())
        assert back == c
        assert back.parents[0].enzymes == ["trypsin"]
        assert back.any_parent_allergen_source is True

    def test_its_fields_count_as_present_for_stage_preconditions(self):
        assert {"origin_type", "parents"} <= candidate().present_fields()


class TestSharedCandidateIsUntouched:
    def test_candidate_keeps_its_three_fields(self):
        assert set(Candidate.model_fields) == {"id", "sequence", "predictions"}

    def test_other_pipelines_candidates_serialise_as_before(self):
        c = Candidate(id="x", sequence="GIGKFLHSAK", predictions={"mic": 1.0})
        assert c.model_dump() == {"id": "x", "sequence": "GIGKFLHSAK", "predictions": {"mic": 1.0}}

"""s04_food_generation on the real Phase 1 substrates, limited to a few templates."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.pipeline.stages.digestion import FoodDigestion  # noqa: E402
from food.c_discovery.generation import settings as S  # noqa: E402
from food.pipeline.stages.generation import LINEAGE, FoodGeneration  # noqa: E402
from food.a_proteins.protein import Protein  # noqa: E402
from food.a_proteins.fasta import read_fasta  # noqa: E402
from food.shared.candidate import FoodCandidate  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"


@pytest.fixture(scope="module")
def run_state():
    subs, _ = read_fasta(FIX / "digestion_substrates.fasta")
    proteins = [Protein(**vars(s), allergen_source=True) for s in subs]
    ctx = SimpleNamespace(proteins=proteins, seed=0)
    digested = FoodDigestion().run([], StageConfig(), ctx)
    return ctx, digested


def test_generated_peptides_are_added_tagged_and_traceable(run_state, monkeypatch):
    ctx, digested = run_state
    monkeypatch.setattr(S, "N_TEMPLATES", 3)
    out = FoodGeneration().run(digested, StageConfig(), ctx)
    assert out[: len(digested)] == digested  # the digestion candidates are untouched
    new = out[len(digested):]
    assert new and all(isinstance(c, FoodCandidate) for c in new)
    for c in new:
        assert (c.origin_type, c.evidence_class, c.generation_method) == ("SYNTHETIC_DESIGN", "THEORETICAL", "reference_guided_edit")
        assert c.parents and all(p.protein_id and p.pathway for p in c.parents)
        assert set(c.predictions["generation"]) == set(LINEAGE)
        assert c.predictions["generation"]["template_origin_type"] == "DIGESTION_DERIVED"
        assert c.predictions["generation"]["guide_origin_type"] == "LITERATURE_KNOWN"
        assert c.any_parent_allergen_source is True  # inherited from the parent protein
    assert len({c.sequence for c in out}) == len(out)


def test_it_needs_the_proteins_for_the_parent_sequences():
    with pytest.raises(ValueError, match="proteins"):
        FoodGeneration().run([], StageConfig(), SimpleNamespace(proteins=None))

"""s03_food_digestion: proteins in the run context become FoodCandidates with every origin kept."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food.digestion import DEFAULT_SETTINGS, run  # noqa: E402
from pipeline.food.digestion.stage import FoodDigestion  # noqa: E402
from pipeline.food.protein_library.protein import Protein  # noqa: E402
from schemas.candidate import Candidate  # noqa: E402
from schemas.food import FoodCandidate  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402


def protein(seq: str, sid: str, **kw) -> Protein:
    acc = sid.split(":")[0]
    return Protein(substrate_id=sid, accession=acc, precursor_start=1, precursor_end=len(seq), sequence=seq, food_source="Whey", **kw)


def run_stage(proteins):
    return FoodDigestion().run([], StageConfig(), SimpleNamespace(proteins=proteins))


def test_one_candidate_per_unique_peptide_matching_the_engine():
    proteins = [protein("AAKGGRCCFLK", "P1:1-11"), protein("GGRCCKAAK", "P2:1-9")]
    _, peptides, _ = run(proteins, DEFAULT_SETTINGS)
    out = run_stage(proteins)
    assert [c.id for c in out] == [p["peptide_id"] for p in peptides]
    assert all(isinstance(c, FoodCandidate) and isinstance(c, Candidate) for c in out)
    assert len({c.sequence for c in out}) == len(out)


def test_a_peptide_from_two_proteins_keeps_both_origins():
    out = {c.sequence: c for c in run_stage([protein("AAKGGRCC", "P1:1-8"), protein("AAKTTRDD", "P2:1-8")])}
    assert {p.protein_id for p in out["AAK"].parents} == {"P1:1-8", "P2:1-8"}


def test_origin_fields_and_protein_provenance_are_copied():
    p = protein("AAKGGRCC", "P1:1-8", version="2", checksum="abc", allergen_source=True, cluster="c7")
    c = next(c for c in run_stage([p]) if c.sequence == "AAK")
    assert (c.origin_type, c.evidence_class, c.generation_method) == ("DIGESTION_DERIVED", "THEORETICAL", "in_silico_digestion")
    assert c.parent_clusters == ["c7"] and c.duplicate_family_id == c.id
    o = next(o for o in c.parents if o.pathway == "trypsin_only")
    assert (o.protein_version, o.protein_checksum, o.allergen_source, o.enzymes, o.ph) == ("2", "abc", True, ["trypsin"], 7.5)
    assert (o.start_in_product, o.end_in_product, o.is_limit_product) == (1, 3, True)
    assert c.any_parent_allergen_source is True and c.mono_mass > 0


def test_an_unassessed_protein_is_unknown_not_safe():
    assert all(c.any_parent_allergen_source is None for c in run_stage([protein("AAKGGRCC", "P1:1-8")]))


def test_a_run_with_no_proteins_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="protein library"):
        run_stage([])


def test_the_stage_contract_names_what_it_produces():
    c = FoodDigestion.contract()
    assert (c["name"], c["requires"], c["produces"]) == ("s03_food_digestion", [], ["origin_type", "parents"])

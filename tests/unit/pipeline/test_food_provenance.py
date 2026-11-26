"""Provenance checks: a clean digestion and generation run passes, and each kind of corruption is caught by the check that names it."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food import settings  # noqa: E402
from pipeline.food.digestion.engine import peptide_id  # noqa: E402
from pipeline.food.digestion.stage import FoodDigestion  # noqa: E402
from pipeline.food.protein_library.clusters import assign_clusters  # noqa: E402
from pipeline.food.protein_library.fasta import read_fasta  # noqa: E402
from pipeline.food.protein_library.protein import Protein  # noqa: E402
from pipeline.food.provenance import check_provenance  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"


@pytest.fixture(scope="module")
def run():
    subs, _ = read_fasta(FIX / "digestion_substrates.fasta")
    proteins = [Protein(**vars(s), allergen_source=(i % 2 == 0), version="1", checksum=f"c{i}") for i, s in enumerate(subs[:12])]
    proteins, _ = assign_clusters(proteins, easy_cluster=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("clustering off")))  # each protein its own cluster, whatever is installed
    candidates = [c for c in FoodDigestion().run([], StageConfig(), SimpleNamespace(proteins=proteins)) if settings.MIN_LENGTH <= len(c.sequence) <= settings.MAX_LENGTH][:40]  # what screening lets through
    for rank, c in enumerate(candidates, 1):  # what the later stages add
        c.predictions.update({"screening": {}, "models": {}, "cleavage": {}, "ranking": {"rank": rank}})
    return proteins, candidates


def issues_of(run, mutate):
    proteins, candidates = run
    copies = [c.model_copy(deep=True) for c in candidates]
    mutate(copies)
    return check_provenance(copies, proteins)


def test_a_real_digestion_of_twelve_proteins_has_no_issue(run):
    proteins, candidates = run
    assert len(candidates) == 40 and check_provenance(candidates, proteins) == []


def test_a_sequence_that_is_not_at_its_stated_position_is_caught(run):
    def mutate(cs):
        cs[3].parents[0].start_in_product += 1
        cs[3].parents[0].end_in_product += 1

    assert {i["check"] for i in issues_of(run, mutate)} >= {"sequence_at_parent_position", "precursor_coordinates"}


def test_an_id_that_does_not_match_its_sequence_and_a_duplicate_are_caught(run):
    def mutate(cs):
        cs[0].id = "PEP-000000000000"
        cs[5].id, cs[5].sequence = cs[6].id, cs[6].sequence

    assert {"id_matches_sequence", "no_duplicate_sequence"} <= {i["check"] for i in issues_of(run, mutate)}


def test_a_parent_that_is_not_in_the_run_or_has_the_wrong_provenance_is_caught(run):
    def mutate(cs):
        cs[1].parents[0].protein_id = "NOT-IN-RUN:1-9"
        cs[2].parents[0].protein_checksum = "tampered"
        cs[2].parents[0].allergen_source = None

    found = {(i["peptide_id"], i["check"]) for i in issues_of(run, mutate)}
    assert (run[1][1].id, "parent_is_in_the_run") in found and (run[1][2].id, "parent_protein_checksum") in found and (run[1][2].id, "parent_allergen_source") in found


def test_a_wrong_cluster_a_missing_stage_and_a_gap_in_the_ranks_are_caught(run):
    def mutate(cs):
        cs[4].parent_clusters = ["protein:ELSEWHERE"]
        del cs[7].predictions["cleavage"]
        cs[9].predictions["ranking"]["rank"] = 99

    assert {"parent_clusters", "stages_complete", "ranks_are_1_to_n"} <= {i["check"] for i in issues_of(run, mutate)}


def test_a_designed_peptide_is_checked_against_its_template_not_its_own_sequence(run):
    proteins, candidates = run
    donor = candidates[0]
    designed = donor.model_copy(deep=True, update={"origin_type": "SYNTHETIC_DESIGN", "generation_method": "reference_guided_edit"})
    template = donor.sequence
    designed.sequence = template[:-1] + ("A" if template[-1] != "A" else "G")
    designed.id = peptide_id(designed.sequence)
    designed.predictions["generation"] = {"template_sequence": template}
    designed.predictions["ranking"] = {"rank": 1}
    assert check_provenance([designed], proteins) == []  # the parent position holds the template
    designed.predictions["generation"] = {"template_sequence": "AAAA" + template}
    assert any(i["check"] == "sequence_at_parent_position" for i in check_provenance([designed], proteins))
    del designed.predictions["generation"]
    assert any(i["check"] == "design_has_template" for i in check_provenance([designed], proteins))

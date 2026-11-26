"""The food batch run end to end with the platform's runner: real substrates, real stages, only the heavy models faked.

Digestion, generation, screening, gut stability and ranking run in order from an empty start; each stage writes its candidates; the final list is ranked,
traceable to its parent proteins, and keeps the food fields through a save and reload.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline.food import registry, settings  # noqa: E402
from pipeline.food.digestion.stage import FoodDigestion  # noqa: E402
from pipeline.food.generation import settings as S  # noqa: E402
from pipeline.food.generation.stage import FoodGeneration  # noqa: E402
from pipeline.food.gut_stability.stage import FoodGutStability  # noqa: E402
from pipeline.food.io import load_food_candidates_jsonl  # noqa: E402
from pipeline.food.protein_library.fasta import read_fasta  # noqa: E402
from pipeline.food.protein_library.protein import Protein  # noqa: E402
from pipeline.food.ranking.stage import FoodRankingDecision  # noqa: E402
from pipeline.food.runner import FoodPipelineRunner  # noqa: E402
from pipeline.food.screening.model_registry import ScreeningModel  # noqa: E402
from pipeline.food.screening.stage import FoodScreening  # noqa: E402
from schemas.food import FoodCandidate  # noqa: E402
from schemas.run_config import RunConfig  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "food" / "phase1"


class FakeModel:
    def __init__(self, score):
        self.score = score

    def predict(self, seqs):
        return {s: {"score": self.score, "label": "x", "applicability": "in_domain", "detail": ""} for s in seqs}


class Screening(FoodScreening):
    def build_models(self, ctx):
        slots = []
        for name, objective, score in (("allergenicity", "safety", 0.8), ("solubility", "developability", 0.6)):
            m = ScreeningModel(name, {"objective": objective})
            m.load_with(lambda score=score: FakeModel(score))
            slots.append(m)
        return slots


STAGES = [FoodDigestion, FoodGeneration, Screening, FoodGutStability, FoodRankingDecision]


def make_proteins():
    subs, _ = read_fasta(FIX / "digestion_substrates.fasta")
    return [Protein(**vars(s), allergen_source=(i % 2 == 0)) for i, s in enumerate(subs[:25])]


def run_once(artifacts: Path, run_id: str):
    proteins = make_proteins()
    cfg = RunConfig(seed_candidates_path="", artifacts_dir=str(artifacts), model_store=str(artifacts), stages={}, run_id=run_id)
    runner = FoodPipelineRunner(cfg)
    runner.ctx = replace(runner.ctx, proteins=proteins)
    return runner.run(), artifacts / "runs" / run_id


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "FOOD_CANDIDATE_STAGE_CLASSES", STAGES)
        mp.setattr(S, "N_TEMPLATES", 3)
        yield run_once(tmp_path_factory.mktemp("food"), "food_e2e")


def test_the_run_ends_with_a_ranked_list_of_food_candidates(run):
    out, _ = run
    assert 0 < len(out) <= settings.FINAL_COUNT
    assert all(isinstance(c, FoodCandidate) for c in out)
    assert [c.predictions["ranking"]["rank"] for c in out] == list(range(1, len(out) + 1))


def test_every_ranked_peptide_is_traceable_and_carries_what_was_not_validated(run):
    out, _ = run
    for c in out:
        assert c.parents and all(p.protein_id for p in c.parents)
        assert c.origin_type in ("DIGESTION_DERIVED", "SYNTHETIC_DESIGN") and c.evidence_class == "THEORETICAL"
        r = c.predictions["ranking"]
        assert r["glp1_classifier_status"] == "Not validated"
        assert r["rationale"] and r["limitations"]
        assert c.predictions["cleavage"]["rules"]["pepsin"] == "pepsin_ph1.3" and 0.0 <= c.predictions["cleavage"]["uncut_bond_fraction"] <= 1.0
        assert "bitterness" in r["predictions_not_run"]  # no bitterness model yet: reported as not run


def test_every_ranked_peptide_passes_the_provenance_check(run):
    from pipeline.food.provenance import check_provenance

    out, _ = run
    assert out and check_provenance(out, make_proteins()) == []  # the real runner's final list, designed peptides included


def test_each_stage_wrote_its_candidates(run):
    _, run_dir = run
    names = ["s03_food_digestion", "s04_food_generation", "s05_food_screening", "s06_food_gut_stability", "s07_food_ranking_decision"]
    for n in names:
        assert (run_dir / "candidates" / f"{n}.jsonl").exists(), n
    assert (run_dir / "audit_log.jsonl").exists()
    audited = [json.loads(line).get("stage") for line in (run_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert set(names) <= set(audited)


def test_the_final_list_reloads_with_its_food_fields(run):
    out, run_dir = run
    back = load_food_candidates_jsonl(str(run_dir / "candidates" / "s07_food_ranking_decision.jsonl"))
    assert [c.id for c in back] == [c.id for c in out]
    assert [c.origin_type for c in back] == [c.origin_type for c in out]
    assert [len(c.parents) for c in back] == [len(c.parents) for c in out]
    assert [c.any_parent_allergen_source for c in back] == [c.any_parent_allergen_source for c in out]


def test_the_run_is_repeatable(run, tmp_path_factory):
    out, _ = run
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry, "FOOD_CANDIDATE_STAGE_CLASSES", STAGES)
        mp.setattr(S, "N_TEMPLATES", 3)
        again, _ = run_once(tmp_path_factory.mktemp("again"), "food_e2e_2")
    assert [c.sequence for c in again] == [c.sequence for c in out]


def test_the_ranked_peptides_carry_their_digestion_detail_alignment_and_mass_and_the_digestion_agrees_with_a_new_simulation(run):
    from common.food_digestion import masses
    from pipeline.food.digestion.explain import explain_candidates

    out, run_dir = run
    back = load_food_candidates_jsonl(str(run_dir / "candidates" / "s07_food_ranking_decision.jsonl"))  # what was saved, not what the stage holds in memory
    assert back and any(c.origin_type == "DIGESTION_DERIVED" for c in back)
    designed = [c for c in load_food_candidates_jsonl(str(run_dir / "candidates" / "s04_food_generation.jsonl")) if c.origin_type == "SYNTHETIC_DESIGN"]
    assert designed and all(c.mono_mass == round(masses.mono_mass(c.sequence), 5) and c.avg_mass == masses.avg_mass(c.sequence) and "free N- and C-termini" in c.chemistry_notes for c in designed)
    for c in back:
        screening = c.predictions["screening"]
        if screening["evidence_peptide"]:
            assert screening["evidence_alignment"]["candidate"] == c.sequence and screening["evidence_alignment"]["reference"] == screening["evidence_peptide"]
        if c.origin_type == "SYNTHETIC_DESIGN":
            assert c.predictions["generation"]["template_digestion"]
        else:
            assert all(p.digestion_detail and p.digestion_detail["stages"] for p in c.parents)
    assert explain_candidates(back, make_proteins()) == []  # the saved origins and fragment status are what the digestion gives again

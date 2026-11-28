"""s05_food_screening: stage A fields against the Phase 1 sample, the funnel to the model stage, and models that cannot run."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.c_discovery import settings  # noqa: E402
from food.c_discovery.screening.model_registry import ScreeningModel  # noqa: E402
from food.pipeline.stages.screening import FoodScreening  # noqa: E402
from food.shared.candidate import FoodCandidate, ParentOrigin  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"
with open(FIX / "screening_stage_a_sample.csv", encoding="utf-8") as f:
    SAMPLE = list(csv.DictReader(f))


def candidate(row: dict, allergen: bool | None = None) -> FoodCandidate:
    """A FoodCandidate with the digestion facts Phase 1 recorded for this peptide."""
    parents = [ParentOrigin(protein_id="P1:1-9", allergen_source=allergen, pathway="trypsin_only", is_limit_product=row["is_limit_product_any"] == "True")]
    return FoodCandidate(id=row["peptide_id"], sequence=row["sequence"], origin_type="DIGESTION_DERIVED", parents=parents,
                         parent_clusters=[f"c{i}" for i in range(int(row["n_parent_clusters"] or 0))])


class Fake:
    def __init__(self, score=0.8, app="in_domain", skip=()):
        self.score, self.app, self.skip = score, app, set(skip)

    def predict(self, seqs):
        return {s: {"score": self.score, "label": "x", "applicability": self.app, "detail": ""} for s in seqs if s not in self.skip}


class Screening(FoodScreening):
    def __init__(self, models):
        self._models = models

    def build_models(self, ctx):
        out = []
        for name, objective, impl in self._models:
            m = ScreeningModel(name, {"objective": objective})
            m.load_with(lambda impl=impl: impl)
            out.append(m)
        return out


def run_stage(rows, models=(("allergenicity", "safety", Fake()),), **kw):
    stage = Screening(models)
    return stage, stage.run([candidate(r, **kw) for r in rows], StageConfig(), SimpleNamespace())


def test_stage_a_fields_match_phase_1_for_the_sample():
    rows = [r for r in SAMPLE if not r["exclusion"]]
    stage, out = run_stage(rows)
    assert 0 < len(out) < len(rows)  # near-duplicates are dropped before the models, as in Phase 1
    assert stage._rejection_reasons == {"not_in_model_shortlist": len(rows) - len(out)}
    by_seq = {c.sequence: c.predictions["screening"] for c in out}
    for r in (r for r in rows if r["sequence"] in by_seq):
        got = by_seq[r["sequence"]]
        for k in ("net_charge_pH7_4", "gravy", "liabilities", "n_liabilities", "evidence_match", "evidence_grade", "evidence", "nearest_positive_sim",
                  "digestion_support", "prior_art_alert"):
            if k in r:
                assert str(got[k]) == r[k] or float(got[k]) == pytest.approx(float(r[k]), abs=1e-9), (r["sequence"], k, got[k], r[k])
        assert got["developability_rule"] == pytest.approx(float(r["developability_rule"]))


def test_a_peptide_that_breaks_a_hard_rule_is_removed_with_the_rule_as_the_reason():
    bad = [r for r in SAMPLE if r["exclusion"]]
    assert bad
    stage, out = run_stage(bad)
    assert out == []
    assert stage._rejection_reasons == {k: sum(r["exclusion"] == k for r in bad) for k in {r["exclusion"] for r in bad}}


def test_only_the_best_few_reach_the_slow_models_and_the_rest_are_counted(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_TOP", 5)
    rows = [r for r in SAMPLE if not r["exclusion"]]
    stage, out = run_stage(rows)
    assert len(out) <= 5
    assert stage._rejection_reasons["not_in_model_shortlist"] == len(rows) - len(out)
    assert all(c.predictions["models"]["allergenicity"]["score"] == 0.8 for c in out)


def test_a_model_that_cannot_run_is_reported_not_run_and_scores_nothing():
    class Broken:
        def __init__(self):
            raise FileNotFoundError("weights not found")

    stage = FoodScreening()
    stage.build_models = lambda ctx: Screening([("allergenicity", "safety", Fake()), ("cytotoxicity", "safety", None)]).build_models(ctx)
    rows = [r for r in SAMPLE if not r["exclusion"]][:5]
    out = stage.run([candidate(r) for r in rows], StageConfig(), SimpleNamespace())
    assert stage.model_status["allergenicity"]["status"] == "ready"
    assert stage.model_status["cytotoxicity"]["status"] == "error"
    assert all(set(c.predictions["models"]) == {"allergenicity"} for c in out)


def test_a_peptide_a_model_skips_has_no_score_from_it():
    rows = [r for r in SAMPLE if not r["exclusion"]][:4]
    skipped = rows[0]["sequence"]
    _, out = run_stage(rows, models=(("allergenicity", "safety", Fake(skip=[skipped])),))
    by_seq = {c.sequence: c.predictions["models"] for c in out}
    assert by_seq[skipped] == {}
    assert all("allergenicity" in v for s, v in by_seq.items() if s != skipped)


@pytest.mark.parametrize("flag,expected", [(True, "allergen_source"), (False, "no_flag"), (None, "not_available")])
def test_the_parent_allergen_flag_comes_from_the_protein_library(flag, expected):
    row = next(r for r in SAMPLE if not r["exclusion"])
    _, (c,) = run_stage([row], allergen=flag)
    assert c.predictions["screening"]["parent_allergen_flag"] == expected


def test_a_generated_peptide_is_looked_up_as_itself_and_its_template_evidence_is_shown_but_never_transferred():
    from food.c_discovery.evidence import EvidenceLookup, load_evidence

    lookup = EvidenceLookup(load_evidence())
    row = next(r for r in SAMPLE if not r["exclusion"] and r["evidence_match"] == "near")
    template = lookup.lookup(row["sequence"])
    assert template["evidence_type"] == "SEQUENCE_SIMILARITY"
    c = candidate(row)
    gen = FoodCandidate(id="G1", sequence="AAKVLE", origin_type="SYNTHETIC_DESIGN", parents=c.parents,
                        predictions={"generation": {"template_sequence": row["sequence"], "edits": "A1>S", "guide_peptide": "LKPT", "sim_to_guide": 0.8}})
    stage = Screening([("allergenicity", "safety", Fake())])
    (out,) = stage.run([gen], StageConfig(), SimpleNamespace())
    s, own = out.predictions["screening"], lookup.lookup("AAKVLE")
    assert (s["evidence_applies_to"], s["template_evidence_applies_to"], s["template_sequence"], s["guide_peptide"]) == ("CANDIDATE", "TEMPLATE", row["sequence"], "LKPT")
    # the designed sequence's own lookup is what is scored and labelled ...
    assert (s["evidence_type"], s["evidence_peptide"], s["evidence"]) == (own["evidence_type"], own["evidence_peptide"], own["evidence"])
    # ... and the template's evidence is only reported next to it
    assert (s["template_evidence_type"], s["template_evidence_peptide"]) == (template["evidence_type"], template["evidence_peptide"])
    assert "evidence_basis" not in s


def test_a_designed_peptide_that_equals_an_evidence_peptide_is_an_exact_sequence_match_of_its_own():
    from food.c_discovery.evidence import load_evidence

    known = next(e for e in load_evidence() if e["direction"] == "positive")["sequence"]
    row = next(r for r in SAMPLE if not r["exclusion"])
    gen = FoodCandidate(id="G2", sequence=known, origin_type="SYNTHETIC_DESIGN", parents=candidate(row).parents,
                        predictions={"generation": {"template_sequence": row["sequence"], "edits": "A1>S", "guide_peptide": "LKPT", "sim_to_guide": 0.8}})
    (out,) = Screening([("allergenicity", "safety", Fake())]).run([gen], StageConfig(), SimpleNamespace())
    assert out.predictions["screening"]["evidence_type"] == "EXACT_SEQUENCE" and out.predictions["screening"]["evidence_applies_to"] == "CANDIDATE"

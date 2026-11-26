"""The ported food screening and ranking code against the Phase 1 outputs it came from (tests/fixtures/food/phase1).

The fixtures are real Phase 1 results: a sample of `screening_stage_a.csv`, the 400-peptide scored pool, the 25 ranked peptides, the
biology shortlist and the reference controls. The code is run on the same inputs and must give the same answers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food import settings  # noqa: E402
from pipeline.food.evidence import EvidenceLookup, load_evidence  # noqa: E402
from pipeline.food.ranking import ranking, shortlist  # noqa: E402
from pipeline.food.screening import checks, screening  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"
#: md5 of labeled_peptides_v1.csv recorded by the last Phase 1 screening run (screening_run.json).
EVIDENCE_MD5 = "a3ede6f56006836c384657b3dc680304"


def read(name: str) -> list[dict]:
    with open(FIX / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def same(a, b) -> bool:
    """CSV text against a recomputed value: numbers compare by value, everything else as text."""
    if isinstance(b, bool):
        return str(b) == a
    if isinstance(b, (int, float, np.floating)):
        return a != "" and float(a) == pytest.approx(float(b), abs=1e-9)
    return str(b) == a


# ---------------------------------------------------------------- evidence data
def test_the_evidence_file_is_the_one_phase_1_screened_against():
    assert hashlib.md5(settings.EVIDENCE_DB.read_bytes()).hexdigest() == EVIDENCE_MD5
    ev = load_evidence()
    assert len(ev) == 81
    assert sorted({e["direction"] for e in ev}) == ["mixed", "negative", "positive"]


# ---------------------------------------------------------------- stage A
STAGE_A = read("screening_stage_a_sample.csv")
PHYSCHEM = ["net_charge_pH7_4", "isoelectric_point", "gravy", "hydrophobic_fraction", "aromaticity", "liabilities", "n_liabilities"]
EVIDENCE = ["evidence_match", "evidence_direction", "evidence_peptide", "evidence_similarity", "evidence_study", "evidence_assay",
            "evidence_food_derivation", "evidence_tier", "evidence_origin_type", "evidence", "nearest_positive", "nearest_positive_sim",
            "nearest_negative", "nearest_negative_sim", "evidence_grade"]


def test_the_sample_covers_every_kind_of_evidence_match_and_the_exclusions():
    assert {r["evidence_match"] for r in STAGE_A if not r["exclusion"]} == {"exact", "contains", "near", "none"}
    assert {r["exclusion"] for r in STAGE_A} >= {""} and any(r["exclusion"] for r in STAGE_A)


@pytest.mark.parametrize("row", [r for r in STAGE_A if r["exclusion"]], ids=lambda r: r["sequence"][:12])
def test_exclusion_reasons_match(row):
    assert screening.exclusion_reason(row["sequence"]) == row["exclusion"]


def test_physchem_and_liabilities_match_phase_1():
    for row in (r for r in STAGE_A if not r["exclusion"]):
        got = screening.physchem(row["sequence"])
        for k in PHYSCHEM:
            assert same(row[k], got[k]), (row["sequence"], k, row[k], got[k])
        # Phase 1 overwrote `developability` with the combined model score for peptides that reached the model stage; the rule score is kept apart.
        assert same(row["developability_rule"], got["developability"]), (row["sequence"], row["developability_rule"], got["developability"])
        assert screening.exclusion_reason(row["sequence"]) == ""


def test_evidence_lookup_matches_phase_1():
    lookup = EvidenceLookup(load_evidence())
    for row in (r for r in STAGE_A if not r["exclusion"]):
        got = lookup.lookup(row["sequence"])
        for k in EVIDENCE:
            assert same(row[k], got[k]), (row["sequence"], k, row[k], got[k])


def test_digestion_support_matches_phase_1():
    for row in (r for r in STAGE_A if not r["exclusion"]):
        assert float(row["digestion_support"]) == pytest.approx(screening.digestion_support(row), abs=1e-9)


def test_prior_art_alerts_match_phase_1():
    prior = checks.PriorArt(load_evidence())
    for row in (r for r in STAGE_A if not r["exclusion"]):
        assert prior.alert(row["sequence"]) == row["prior_art_alert"], row["sequence"]


# ---------------------------------------------------------------- ranking
POOL = read("screening_scored.csv")
RANKED = read("ranked_candidates.csv")
RUN = json.loads((FIX / "screening_run.json").read_text(encoding="utf-8"))


def numeric(rows: list[dict]) -> list[dict]:
    """The pool as the stage held it: the objective and tie-break columns as numbers, everything else as text."""
    cols = set(RUN["objectives_used"]) | set(settings.TIEBREAK)
    # the gate flag is Phase 1's own gate (any match with a positive or mixed effect), so the unchanged Pareto selection is compared like for like
    return [{**{k: (float(v) if k in cols and v != "" else v) for k, v in r.items()}, "evidence_gate_passed": r["evidence_gate"] == "direct"} for r in rows]


def test_ranking_picks_the_same_peptides_in_the_same_order_as_phase_1():
    pool = numeric(POOL)
    objs = RUN["objectives_used"]
    clusters = ranking.cluster([c["sequence"] for c in pool])
    parents = [shortlist.first_parent_cluster(c) for c in pool]
    chosen, gate, per = ranking.pareto_pick_selection(pool, objs, clusters, parents, settings.FINAL_COUNT, settings.MIN_EXACT_EVIDENCE)
    assert [pool[i]["sequence"] for i in chosen] == [r["sequence"] for r in RANKED]
    assert gate["exact_evidence_in_final"] == RUN["selection"]["direct_in_final"]
    assert gate["shortfall"] == RUN["selection"]["shortfall"]
    assert per["n_fronts"] == RUN["n_pareto_fronts"]
    assert int(clusters.max()) == RUN["n_clusters"]
    for rank, i in enumerate(chosen):
        assert int(clusters[i]) == int(RANKED[rank]["cluster"])
        assert per["pareto_front"][i] == int(RANKED[rank]["pareto_front"])


def test_the_rationale_keeps_phase_1_wording_apart_from_the_evidence_sentences():
    from schemas.food_dossier_pdf import split_outside_brackets

    def kept(text):
        bits = [b.split("; ", 1)[-1] if b.startswith("evidence gate:") else b for b in split_outside_brackets(text)]
        return [b for b in bits if b.startswith(("Pareto front", "near pool-best", "weak on", "applicability:", "prior-art alert", "liabilities"))]

    pool = numeric(POOL)
    objs = RUN["objectives_used"]
    clusters = ranking.cluster([c["sequence"] for c in pool])
    parents = [shortlist.first_parent_cluster(c) for c in pool]
    chosen, _, per = ranking.pareto_pick_selection(pool, objs, clusters, parents, settings.FINAL_COUNT, settings.MIN_EXACT_EVIDENCE)
    best = {o: max(c[o] for c in pool) for o in objs}
    for k, c in enumerate(pool):
        c["cluster"], c["n_fronts"], c["pareto_front"] = int(clusters[k]), per["n_fronts"], per["pareto_front"][k]
    for rank, i in enumerate(chosen):
        got = ranking.rationale({k: ("" if (isinstance(v, float) and np.isnan(v)) else v) for k, v in pool[i].items()}, objs, best)
        assert kept(got) == kept(RANKED[rank]["rationale"]), RANKED[rank]["sequence"]


def test_dominator_counts_and_fronts_agree_on_a_small_case():
    X = np.array([[1.0, 1.0], [0.5, 0.5], [1.0, 0.2], [0.2, 1.0]])
    assert ranking.pareto_fronts(X).tolist() == [1, 2, 2, 2]
    assert ranking.dominator_counts(X).tolist() == [0, 1, 1, 1]


# ---------------------------------------------------------------- shortlist and controls
EVIDENCE_TYPE_OF_MATCH = {"exact": "EXACT_SEQUENCE", "contains": "CONTAINS_REFERENCE", "near": "SEQUENCE_SIMILARITY", "none": "NO_EVIDENCE"}


def test_the_biology_shortlist_rule_changed_on_purpose_so_it_no_longer_equals_phase_1_but_keeps_its_properties():
    """Phase 1 took one peptide per family, at most 4 designed and at most 2 'exploratory' ones, 8 in all. The platform lists every peptide whose evidence is an exact, contains or
    contained-in relationship, leaves off those a model calls unfavourable, and keeps one per family and at most 4 designed, up to 10. Here Phase 1's ranked list is run through it."""
    rows = [{**r, "evidence_type": EVIDENCE_TYPE_OF_MATCH[r["evidence_match"]], "evidence_gate_passed": r["evidence_gate"] == "direct"} for r in RANKED]
    got = shortlist.make_shortlist(rows)
    by_id = {r["peptide_id"]: r for r in rows}
    assert 0 < len(got) <= settings.SHORTLIST_SIZE == 10
    assert [g["rank"] for g in got] == sorted(g["rank"] for g in got) and [g["shortlist_position"] for g in got] == list(range(1, len(got) + 1))
    assert len({g["parent_family"] for g in got}) == len(got) and sum(g["origin_type"] == "SYNTHETIC_DESIGN" for g in got) <= settings.SHORTLIST_MAX_SYNTHETIC
    for g in got:
        r = by_id[g["peptide_id"]]
        assert r["evidence_type"] in settings.SHORTLIST_EVIDENCE and r["evidence_direction"] in ("positive", "mixed") and not shortlist.unfavourable_calls(r)
    reasons = {d["peptide_id"]: d for d in shortlist.shortlist_decisions(rows)}  # a peptide the rule left off has a stated reason
    assert all(d["reason"] for d in reasons.values()) and {g["peptide_id"] for g in got} == {i for i, d in reasons.items() if d["shortlisted"]}


def test_the_reference_controls_match_phase_1():
    got = shortlist.reference_controls()
    want = read("reference_controls.csv")
    assert [(g["role"], g["sequence"]) for g in got] == [(w["role"], w["sequence"]) for w in want]


# ---------------------------------------------------------------- checks that changed on purpose
@pytest.mark.parametrize("flag,expected", [(True, "allergen_source"), (False, "no_flag"), (None, "not_available")])
def test_parent_allergen_flag_comes_from_the_protein_library_and_is_never_guessed(flag, expected):
    assert checks.parent_allergen_flag(flag) == expected

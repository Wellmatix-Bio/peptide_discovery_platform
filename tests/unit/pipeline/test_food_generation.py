"""Food candidate generation: edit operations and filters, and the loop against the Phase 1 generator on the real substrates.

The substrate FASTA, the Phase 1 generated peptides and its summary are in tests/fixtures/food/phase1. The full run (all 150 templates, 889 peptides) was
also checked once outside the suite and matched Phase 1 exactly: every peptide, every field, the filter failure counts and the 1,225-row filter log.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.c_discovery import settings  # noqa: E402
from food.b_digestion import DEFAULT_SETTINGS, run  # noqa: E402
from food.c_discovery.evidence import load_evidence  # noqa: E402
from food.c_discovery.generation import edits as E  # noqa: E402
from food.c_discovery.generation import filters as Fl  # noqa: E402
from food.c_discovery.generation import generate as G  # noqa: E402
from food.c_discovery.generation import settings as S  # noqa: E402
from food.a_proteins.fasta import read_fasta  # noqa: E402
from food.c_discovery.screening.screening import physchem  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"


# ---------------------------------------------------------------- edits
def test_edit_operations():
    t = "RQGDVIALPAGIVHW"
    assert E.apply_edits(t, (("sub", 0, "R", "K", "conservative"),)) == "KQGDVIALPAGIVHW"
    assert E.apply_edits(t, (("del", 14, "W", "", "deletion"),)) == "RQGDVIALPAGIVH"
    assert E.apply_edits(t, (("del", 0, "R", "", "deletion"), ("sub", 3, "D", "E", "conservative"))) == "QGEVIALPAGIVHW"
    assert E.describe((("sub", 0, "R", "K", "c"), ("del", 14, "W", "", "d"))) == "R1>K;W15>del"
    assert all(a not in E.CONSERVATIVE[a] for a in E.AA)
    v = E.enumerate_variants("LKPTA", [])
    assert "LKPTA" not in v
    assert all(len(ed) <= S.MAX_EDITS and sum(e[0] == "del" for e in ed) <= 1 for ed in v.values())


# ---------------------------------------------------------------- filters
def test_length_composition_and_synthesis_filters_say_why():
    assert Fl.length_ok("AKV") == (False, "length")
    assert Fl.length_ok("A" * 21) == (False, "length")
    assert Fl.length_ok("AKVL") == (True, "")
    assert Fl.composition_ok("AAAAAAKV", physchem("AAAAAAKV")) == (False, "composition_dominant_residue")
    assert Fl.composition_ok("KKKKKRRRR", physchem("KKKKKRRRR"))[1] == "composition_charge"
    assert Fl.synthesis_ok(physchem("AKCVLE")) == (False, "synthesis_cysteine")


def test_novelty_filter():
    assert Fl.novelty_ok("AKVL", {"AKVL"}, set()) == (False, "novelty_digestion_fragment")
    assert Fl.novelty_ok("AKVL", set(), {"AKVL"}) == (False, "novelty_known_peptide")
    assert Fl.novelty_ok("AKVL", set(), set()) == (True, "")


def test_an_edit_must_move_toward_the_guide_and_not_toward_a_negative():
    near = {"nearest_positive_sim": 0.8, "nearest_negative_sim": 0.1}
    assert Fl.guidance_ok(near, 0.6, 0.1) == (True, "")
    assert Fl.guidance_ok({**near, "nearest_positive_sim": 0.5}, 0.6, 0.1) == (False, "guidance_away_from_positive")
    assert Fl.guidance_ok({**near, "nearest_positive_sim": 0.6}, 0.6, 0.1) == (False, "guidance_no_improvement")
    assert Fl.guidance_ok({"nearest_positive_sim": 0.8, "nearest_negative_sim": 0.9}, 0.6, 0.1) == (False, "guidance_toward_negative")


# ---------------------------------------------------------------- the loop against Phase 1
@pytest.fixture(scope="module")
def digested():
    subs, issues = read_fasta(FIX / "digestion_substrates.fasta")
    assert not issues
    origins, peptides, _ = run(subs, DEFAULT_SETTINGS)
    return subs, origins, peptides


def test_digestion_of_the_real_substrates_gives_the_phase_1_registry_size(digested):
    subs, origins, peptides = digested
    assert (len(subs), len(origins), len(peptides)) == (524, 295406, 43898)


def test_templates_and_generated_peptides_match_phase_1(digested):
    subs, origins, peptides = digested
    evidence = load_evidence()
    _, guide_scorer, evidence_seqs = G.guides_and_scorers(evidence)
    summary = json.loads((FIX / "generation_summary.json").read_text(encoding="utf-8"))
    templates = G.select_templates(peptides, evidence_seqs, guide_scorer, settings_n := S.N_TEMPLATES)
    assert len(templates) == summary["n_templates"] == settings_n
    assert [t["sequence"] for t in templates] == [p["template"] for p in summary["per_template"]]

    first = templates[:8]
    by_peptide: dict[str, list] = {}
    for o in origins:
        by_peptide.setdefault(o["peptide_id"], []).append(o)
    passed, log_rows, per_template, _ = G.generate(
        first, lambda pid: G.pick_origin(by_peptide[pid]), {s.substrate_id: s.sequence for s in subs},
        {p["sequence"] for p in peptides}, evidence, seed=S.SEED, evidence_md5=hashlib.md5(settings.EVIDENCE_DB.read_bytes()).hexdigest())

    with open(FIX / "generated_peptides.csv", encoding="utf-8") as f:
        want = [r for r in csv.DictReader(f) if r["template_sequence"] in {t["sequence"] for t in first}]
    assert [r["sequence"] for r in passed] == [r["sequence"] for r in want]
    skip = {"digestion_peptides_md5", "parent_clusters", "n_parent_clusters"}  # clusters need the library's cluster table
    for got, exp in zip(passed, want):
        for k, v in exp.items():
            if k in skip:
                continue
            assert str(got[k]) == v or float(got[k]) == pytest.approx(float(v), abs=1e-9), (got["sequence"], k, got[k], v)
    assert [p["n_passed"] for p in per_template] == [p["n_passed"] for p in summary["per_template"][:8]]
    assert {r["decision"] for r in log_rows} <= {"pass", "fail_digestion_plausibility", "fail_duplicate_generated"}


def test_every_generated_peptide_is_tagged_and_traceable(digested):
    subs, origins, peptides = digested
    evidence = load_evidence()
    _, guide_scorer, evidence_seqs = G.guides_and_scorers(evidence)
    templates = G.select_templates(peptides, evidence_seqs, guide_scorer, 3)
    by_peptide: dict[str, list] = {}
    for o in origins:
        by_peptide.setdefault(o["peptide_id"], []).append(o)
    passed, _, _, _ = G.generate(templates, lambda pid: G.pick_origin(by_peptide[pid]), {s.substrate_id: s.sequence for s in subs},
                                 {p["sequence"] for p in peptides}, evidence)
    assert passed
    for r in passed:
        assert (r["origin_type"], r["template_origin_type"], r["guide_origin_type"]) == ("SYNTHETIC_DESIGN", "DIGESTION_DERIVED", "LITERATURE_KNOWN")
        assert r["plausible_pathways"] and r["template_sequence"] and r["guide_peptide"]

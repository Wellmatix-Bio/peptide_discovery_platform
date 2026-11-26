"""The generation loop: pick templates, enumerate edits, filter, and check digestion plausibility. Ported from Phase 1 `6. Candidate Generation/run.py`."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from ..digestion.engine import peptide_id
from ..evidence import EvidenceLookup
from ..ranking.ranking import diverse_shortlist
from ..screening.screening import physchem
from . import edits as E
from . import filters as Fl
from . import settings as G


def guides_and_scorers(evidence: list[dict]) -> tuple[EvidenceLookup, EvidenceLookup, set[str]]:
    """(scorer over all evidence, guide scorer over positives only, evidence sequences). Guides steer the edits; all evidence guards novelty and negatives."""
    guides = [e for e in evidence if e["direction"] == "positive" and len(e["sequence"]) <= G.GUIDE_MAX_LENGTH
              and e["food_derivation"] not in G.GUIDE_EXCLUDE_FOOD_DERIVATION]
    return EvidenceLookup(evidence), EvidenceLookup(guides), {e["sequence"] for e in evidence}


def select_templates(peptides: Sequence[Mapping], evidence_seqs: set[str], guide_scorer: EvidenceLookup, n: int = G.N_TEMPLATES) -> list[dict]:
    """Digestion fragments near a known positive, best guide similarity first, near-duplicates removed.

    `peptides` rows need peptide_id, sequence, length and is_limit_product_any; parent_clusters and n_parent_clusters are carried through when present."""
    cand = []
    for p in peptides:
        seq = p["sequence"]
        if not (G.TEMPLATE_MIN_LENGTH <= int(p["length"]) <= G.TEMPLATE_MAX_LENGTH):
            continue
        if any(r in seq for r in G.EXCLUDE_TEMPLATE_RESIDUES):
            continue
        look = guide_scorer.lookup(seq)
        sim = look["nearest_positive_sim"]
        if G.TEMPLATE_MIN_GUIDE_SIM <= sim < 1.0 and seq not in evidence_seqs:
            cand.append({**p, "guide_sim": sim, "guide": look["nearest_positive"], "limit": int(str(p["is_limit_product_any"]) == "True")})
    cand.sort(key=lambda c: (-c["guide_sim"], -c["limit"], c["peptide_id"]))
    keep = diverse_shortlist([c["sequence"] for c in cand], np.arange(len(cand)), n)
    return [cand[i] for i in keep]


def pick_origin(origins: Sequence[Mapping]) -> Mapping:
    """The one origin a template is edited in: a limit product if there is one, then the first substrate id."""
    return min(origins, key=lambda o: (0 if str(o["is_limit_product"]) == "True" else 1, o["substrate_id"]))


def generate(
    templates: Sequence[Mapping],
    origin_of: Callable[[str], Mapping],
    parent_seq: Mapping[str, str],
    digestion_seqs: set[str],
    evidence: list[dict],
    seed: int = G.SEED,
    evidence_md5: str = "",
    log: Callable[[str], None] = lambda m: None,
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    """(generated peptides that passed every filter, filter log, per-template counts, failures per filter).

    `origin_of(peptide_id)` gives the template's origin: substrate_id, accession, food_source, start_in_product (1-based)."""
    scorer, guide_scorer, evidence_seqs = guides_and_scorers(evidence)

    def look_up(seq: str) -> dict:
        d = guide_scorer.lookup(seq)
        d["nearest_negative_sim"] = scorer.lookup(seq)["nearest_negative_sim"]
        return d

    plaus = Fl.Plausibility()
    seen: set[str] = set()
    reasons: dict[str, int] = {}
    per_template, passed_rows, log_rows = [], [], []

    def count(r: str, n: int = 1) -> None:
        reasons[r] = reasons.get(r, 0) + n

    for ti, t in enumerate(templates):
        o = origin_of(t["peptide_id"])
        parent = parent_seq[o["substrate_id"]]
        s0 = int(o["start_in_product"]) - 1
        tseq = t["sequence"]
        assert parent[s0:s0 + len(tseq)] == tseq, f"template {t['peptide_id']} does not match its parent"
        tlook = look_up(tseq)
        guide = tlook["nearest_positive"]
        guided = E.guided_positions(guide_scorer.al, tseq, guide)
        variants = E.enumerate_variants(tseq, guided)
        fails: dict[str, int] = {}
        survivors = []
        for seq, ed in variants.items():
            ok_, why = Fl.length_ok(seq)
            phys = look = None
            if ok_:
                phys = physchem(seq)
                for fn in (lambda: Fl.composition_ok(seq, phys), lambda: Fl.synthesis_ok(phys),
                           lambda: Fl.novelty_ok(seq, digestion_seqs, evidence_seqs)):
                    ok_, why = fn()
                    if not ok_:
                        break
            if ok_:
                look = look_up(seq)
                ok_, why = Fl.guidance_ok(look, tlook["nearest_positive_sim"], tlook["nearest_negative_sim"])
            if ok_ and seq in seen:
                ok_, why = False, "duplicate_generated"
            if not ok_:
                fails[why] = fails.get(why, 0) + 1
                count(why)
                continue
            survivors.append((seq, ed, phys, look))
        n_survived = len(survivors)
        rng = np.random.default_rng([seed, int(t["peptide_id"][4:12], 16)])
        if n_survived > G.MAX_CANDIDATES_PER_TEMPLATE:
            pick = sorted(rng.choice(n_survived, G.MAX_CANDIDATES_PER_TEMPLATE, replace=False).tolist())
            fails["not_sampled_cap"] = n_survived - len(pick)
            count("not_sampled_cap", n_survived - len(pick))
            survivors = [survivors[i] for i in pick]
        n_pass = 0
        for seq, ed, phys, look in survivors:
            res = plaus.check(parent, s0, len(tseq), seq)
            row = {"peptide_id": peptide_id(seq), "sequence": seq, "template_peptide_id": t["peptide_id"],
                   "template_sequence": tseq, "edits": E.describe(ed), "n_edits": len(ed),
                   "edit_kinds": ";".join(sorted({e[4] for e in ed})), "sim_to_guide": look["nearest_positive_sim"],
                   "template_sim_to_guide": tlook["nearest_positive_sim"], "digestion_plausible": res["plausible"],
                   "plausible_pathways": res["pathways"]}
            if not res["plausible"]:
                row["decision"] = "fail_digestion_plausibility"
                fails["digestion_plausibility"] = fails.get("digestion_plausibility", 0) + 1
                count("digestion_plausibility")
                log_rows.append(row)
                continue
            if seq in seen:
                row["decision"] = "fail_duplicate_generated"
                fails["duplicate_generated"] = fails.get("duplicate_generated", 0) + 1
                count("duplicate_generated")
                log_rows.append(row)
                continue
            row["decision"] = "pass"
            log_rows.append(row)
            seen.add(seq)
            n_pass += 1
            passed_rows.append({
                "peptide_id": row["peptide_id"], "sequence": seq, "length": len(seq),
                "origin_type": G.ORIGIN_TYPE, "generation_method": G.GENERATION_METHOD,
                "template_peptide_id": t["peptide_id"], "template_sequence": tseq, "template_origin_type": G.TEMPLATE_ORIGIN_TYPE,
                "parent_substrate": o["substrate_id"], "parent_accession": o["accession"], "food_source": o["food_source"],
                "template_start_in_parent": s0 + 1, "template_end_in_parent": s0 + len(tseq),
                "edits": row["edits"], "n_edits": len(ed), "edit_kinds": row["edit_kinds"],
                "guide_peptide": guide, "guide_origin_type": G.GUIDE_ORIGIN_TYPE,
                "guide_study": guide_scorer.by_seq[guide]["study"], "guide_food_derivation": guide_scorer.by_seq[guide]["food_derivation"],
                "sim_to_guide": look["nearest_positive_sim"], "template_sim_to_guide": tlook["nearest_positive_sim"],
                "nearest_negative_sim": look["nearest_negative_sim"], "evidence_match": look["evidence_match"],
                "plausible_pathways": res["pathways"], "is_limit_product_any": str(bool(res["is_limit_product"])),
                "missed_cleavages_final": res["missed_cleavages_final"],
                "net_charge_pH7_4": phys["net_charge_pH7_4"], "liabilities": phys["liabilities"],
                "parent_clusters": t.get("parent_clusters", ""), "n_parent_clusters": t.get("n_parent_clusters", 0),
                "pathways": res["pathways"], "parent_substrates": o["substrate_id"],
                "seed": seed, "generator_version": G.VERSION, "evidence_db_md5": evidence_md5})
        per_template.append({"template": tseq, "guide": guide, "n_enumerated": len(variants),
                             "n_survived_cheap_filters": n_survived, "n_passed": n_pass, "filter_failures": fails})
        if (ti + 1) % 25 == 0:
            log(f"  {ti + 1}/{len(templates)} templates, {len(passed_rows)} peptides passed so far")
    return passed_rows, log_rows, per_template, reasons

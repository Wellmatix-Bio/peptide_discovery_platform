"""Evidence lookup against the labelled peptides: what has been measured for a peptide or its closest relatives."""
from __future__ import annotations

import csv

from Bio import Align
from Bio.Align import substitution_matrices

from .. import settings


def load_evidence() -> list[dict]:
    """The labelled peptides of the evidence database (positives, negatives and the one mixed peptide)."""
    with open(settings.EVIDENCE_DB, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [{"sequence": r["sequence"], "direction": r["direction"], "study": r["studies"],
             "assay": r["cell_models"], "food_derivation": r["food_derivation"], "tier": r["evidence_tier"]}
            for r in rows if r["sequence"] and set(r["sequence"]) <= settings.STANDARD_AA]


class EvidenceLookup:
    """What has been measured for this peptide or its closest relatives.

    exact      same sequence as an evidence peptide
    contains   an evidence peptide (>= MIN_FUZZY_LENGTH aa) sits inside the candidate, or the reverse
    near       local alignment (BLOSUM62, normalised by the smaller self-score) >= NEAR_THRESHOLD
    none       nothing close. This means "no data", never "inactive".
    Labels are reported, not transferred: only an exact match scores like the evidence peptide itself.

    `evidence_type` names the same relationship for the reports. These are sequence relationships, not proof of biological activity:
    EXACT_SEQUENCE            the candidate equals the evidence peptide
    CONTAINS_REFERENCE        the candidate contains the whole evidence peptide (4 residues or more)
    CONTAINED_IN_REFERENCE    the candidate is a subsequence of the evidence peptide
    SEQUENCE_SIMILARITY       alignment score of NEAR_THRESHOLD or more, with no exact or containment match
    NO_EVIDENCE               nothing qualifies
    """

    def __init__(self, evidence: list[dict]):
        al = Align.PairwiseAligner()
        al.mode = "local"
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.open_gap_score, al.extend_gap_score = settings.ALIGN_OPEN, settings.ALIGN_EXTEND
        self.al, self.ev = al, evidence
        self.by_seq = {e["sequence"]: e for e in evidence}
        self.fuzzy = [e for e in evidence if len(e["sequence"]) >= settings.MIN_FUZZY_LENGTH]
        self._self: dict[str, float] = {}

    def _selfscore(self, s: str) -> float:
        if s not in self._self:
            self._self[s] = self.al.score(s, s)
        return self._self[s]

    @staticmethod
    def _fields(e: dict, match: str, sim: float) -> dict:
        return {"evidence_match": match, "evidence_direction": e["direction"] if match != "none" else "none",
                "evidence_peptide": e["sequence"] if match != "none" else "", "evidence_similarity": round(sim, 3),
                "evidence_study": e["study"] if match != "none" else "", "evidence_assay": e["assay"] if match != "none" else "",
                "evidence_food_derivation": e["food_derivation"] if match != "none" else "",
                "evidence_tier": e["tier"] if match != "none" else "",
                "evidence_origin_type": "LITERATURE_KNOWN" if match != "none" else ""}

    def alignment(self, seq: str, reference: str) -> dict | None:
        """How the candidate and one evidence peptide line up, for the reports only (it feeds no score). An exact or containment match is the contained sequence inside the other; any other
        pair uses the same local alignment (BLOSUM62, the same gap scores) as the similarity score. `identity` is identical aligned positions over aligned columns; `candidate_coverage` and
        `reference_coverage` are the aligned residues of each sequence over its length."""
        if not seq or not reference:
            return None
        if seq == reference or reference in seq or seq in reference:
            inner, outer = (reference, seq) if reference in seq else (seq, reference)
            i = outer.find(inner)
            c0, r0 = (i, 0) if reference in seq else (0, i)
            a_cand, a_ref = inner, inner
            c_len = r_len = len(inner)
            score = None
        else:
            a = self.al.align(seq, reference)[0]
            (cs, ce), (rs, re_) = a.aligned[0][0], a.aligned[1][0]
            c0, r0, a_cand, a_ref, score = int(cs), int(rs), a[0], a[1], float(a.score)
            c_len, r_len = int(sum(int(e - s) for s, e in a.aligned[0])), int(sum(int(e - s) for s, e in a.aligned[1]))
        matrix = self.al.substitution_matrix
        mid = "".join("|" if x == y else "+" if "-" not in (x, y) and matrix[x][y] > 0 else " " for x, y in zip(a_cand, a_ref))
        identical = mid.count("|")
        relationship = "identical" if seq == reference else "reference inside candidate" if reference in seq else "candidate inside reference" if seq in reference else "similar"
        return {"relationship": relationship, "candidate": seq, "reference": reference, "candidate_length": len(seq), "reference_length": len(reference),
                "aligned_candidate": a_cand, "aligned_reference": a_ref, "midline": mid, "candidate_start": c0 + 1, "candidate_end": c0 + c_len, "reference_start": r0 + 1,
                "reference_end": r0 + r_len, "aligned_columns": len(a_cand), "identical_positions": identical, "identity": round(identical / len(a_cand), 3),
                "candidate_coverage": round(c_len / len(seq), 3), "reference_coverage": round(r_len / len(reference), 3), "alignment_score": score}

    def lookup(self, seq: str) -> dict:
        cap = settings.NON_EXACT_CAP
        ex = self.by_seq.get(seq)
        if ex:
            out = self._fields(ex, "exact", 1.0)
            out["evidence"] = {"positive": 1.0, "mixed": 0.5, "negative": 0.0}[ex["direction"]]
            out.update(nearest_positive=seq if ex["direction"] != "negative" else "",
                       nearest_positive_sim=1.0 if ex["direction"] != "negative" else 0.0,
                       nearest_negative=seq if ex["direction"] != "positive" else "",
                       nearest_negative_sim=1.0 if ex["direction"] != "positive" else 0.0)
            out["evidence_grade"] = f"exact_{ex['direction']}"
            out["evidence_type"] = "EXACT_SEQUENCE"
            return out
        best = {"positive": (0.0, None), "negative": (0.0, None)}      # per side: (similarity, evidence row)
        top = (0.0, None, False, False)                               # overall nearest: (sim, row, contains, reference sits inside the candidate)
        for e in self.fuzzy:
            r = e["sequence"]
            reference_inside = r in seq
            contains = reference_inside or seq in r
            sim = 1.0 if contains else self.al.score(seq, r) / min(self._selfscore(seq), self._selfscore(r))
            sim = min(sim, 1.0)
            if e["direction"] in ("positive", "mixed") and sim > best["positive"][0]:
                best["positive"] = (sim, e)
            if e["direction"] in ("negative", "mixed") and sim > best["negative"][0]:
                best["negative"] = (sim, e)
            if sim > top[0] or (sim == top[0] and contains and not top[2]):
                top = (sim, e, contains, reference_inside)
        sim, e, contains, reference_inside = top
        match = "none" if e is None else "contains" if contains else "near" if sim >= settings.NEAR_THRESHOLD else "none"
        out = self._fields(e, match, sim) if e else self._fields({"sequence": "", "direction": "none", "study": "", "assay": "",
                                                                  "food_derivation": "", "tier": ""}, "none", 0.0)
        sp, sn = min(best["positive"][0], cap), min(best["negative"][0], cap)
        out["evidence"] = round(max(0.0, sp - max(0.0, sn - sp)), 3)      # nearest positive, reduced when a known negative is closer
        out.update(nearest_positive=best["positive"][1]["sequence"] if best["positive"][1] else "",
                   nearest_positive_sim=round(best["positive"][0], 3),
                   nearest_negative=best["negative"][1]["sequence"] if best["negative"][1] else "",
                   nearest_negative_sim=round(best["negative"][0], 3))
        out["evidence_grade"] = "none" if match == "none" else f"{match}_{out['evidence_direction']}"
        out["evidence_type"] = ("NO_EVIDENCE" if match == "none" else "SEQUENCE_SIMILARITY" if match == "near"
                                else "CONTAINS_REFERENCE" if reference_inside else "CONTAINED_IN_REFERENCE")
        return out

"""Constants for food screening and ranking, taken from the Phase 1 screening settings (Models/Food Peptide GLP1 Discovery/7. AI Screening and Ranking/settings.py)."""
from __future__ import annotations

from pathlib import Path

EVIDENCE_DB = Path(__file__).resolve().parent / "evidence" / "data" / "labeled_peptides_v1.csv"  # 81 labelled peptides, v1

# Gut proteases and the Expasy rule each one uses. Pepsin and trypsin are the rules the default digestion pathways apply (digestion/settings.py); chymotrypsin uses the
# low-specificity rule for survival (docs/validation/food/protease_validation.md), while the digestion pathways keep the high-specificity one.
GUT_ENZYMES = {"pepsin": "pepsin_ph1.3", "trypsin": "trypsin", "chymotrypsin": "chymotrypsin_low"}  # chymotrypsin_low: finds 97% of observed cuts, the narrow rule 62%

# Hard exclusion rules.
MIN_LENGTH = 4  # shortest reference GLP-1 peptide in BIOPEP
MAX_LENGTH = 40
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Digestion support from the digestion run.
DIGESTION_WEIGHTS = {"limit_product": 0.5, "parent_clusters": 0.5}
PARENT_CLUSTER_CAP = 5  # parent clusters beyond this add nothing

# Evidence lookup: exact, containment, then BLOSUM62 local alignment normalised by the smaller self-score.
ALIGN_OPEN, ALIGN_EXTEND = -10.0, -1.0
NEAR_THRESHOLD = 0.75  # heuristic, uncalibrated
MIN_FUZZY_LENGTH = 4  # shorter evidence peptides match exactly only
NON_EXACT_CAP = 0.9  # containment and near matches never score as high as an exact match

# Developability liabilities (sequence-only heuristic): score = 1 - min(1, n_liabilities / LIABILITY_CAP).
LIABILITY_CAP = 4
HYDROPHOBIC = set("AVILMFWC")
HYDROPHOBIC_FRACTION_LIMIT = 0.6
LONG_PEPTIDE = 25

# Funnel and ranking.
MODEL_TOP = 400  # peptides sent to the slow model stage
SHORTLIST_SIMILARITY = 0.6  # 3-mer Jaccard above which a shortlist candidate is a near-duplicate
CLUSTER_DISTANCE = 0.6  # 1 - 3-mer Jaccard; closer peptides share a cluster
FINAL_COUNT = 25  # 20-30 prioritised peptides
MAX_PER_CLUSTER = 2
OBJECTIVES = ["digestion_plausibility", "coverage", "developability_rule"]  # evidence is appended last by the stage
TIEBREAK = ("model_allergenicity_score", "model_cytotoxicity_score", "evidence")  # fixed order, higher first, then sequence
# Coverage: mean over the assessment dimensions of these weights; the weights are a judgement call.
COVERAGE_WEIGHTS = {"in_domain": 1.0, "limited": 0.5, "out_of_domain": 0.0, "not_run": 0.0}
SCORING_APPLICABILITY = ("in_domain", "limited")  # an out-of-domain model does not veto; it is listed as flagged
PATENT_PREFIXES = ("EP", "WO", "CN", "US")  # study ids that mark patent-sourced evidence peptides

# Applicability (reported per dimension, never averaged into a score).
LIMITED_SIMILARITY = 0.6  # nearest-evidence similarity at or above this is "limited"

# Selection (defaults awaiting sign-off).
MIN_EXACT_EVIDENCE = 15  # of FINAL_COUNT: ranked candidates needing an EXACT_SEQUENCE match to an evidence peptide with a positive or mixed effect; a judgement call, not derived from data
MAX_PER_PARENT_CLUSTER = 3  # protein-family diversity

# Biology shortlist and reference controls.
SHORTLIST_SIZE = 10  # the most peptides on the biology shortlist (spec: 5 to 10); Phase 1 used 8
SHORTLIST_MAX_SYNTHETIC = 4
#: A peptide is eligible for the shortlist when it equals, contains or is a fragment of an evidence peptide whose reported effect is positive or mixed. There is no cap on how many
#: eligible peptides lack an exact match. Sequence-similarity-only and no-evidence peptides are not eligible. (Decided 2026-11-25.)
SHORTLIST_EVIDENCE = ("EXACT_SEQUENCE", "CONTAINS_REFERENCE", "CONTAINED_IN_REFERENCE")
N_CONTROLS = 3  # positive and negative reference controls each


from food.shared.dossier import UNCUT_BOND_FRACTION_STATEMENT  # noqa: E402,F401  (what `uncut_bond_fraction` is and is not; one copy, shown wherever the value is shown)

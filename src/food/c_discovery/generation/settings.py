"""Reference-guided generation settings, from the Phase 1 generator (Models/Food Peptide GLP1 Discovery/6. Candidate Generation/gen_settings.py)."""
from __future__ import annotations

VERSION = "gen_v1"
SEED = 0
ORIGIN_TYPE = "SYNTHETIC_DESIGN"  # every edited peptide; the fragment it was edited from stays DIGESTION_DERIVED
GENERATION_METHOD = "reference_guided_edit"
TEMPLATE_ORIGIN_TYPE = "DIGESTION_DERIVED"
GUIDE_ORIGIN_TYPE = "LITERATURE_KNOWN"

# Templates: food-derived fragments that get edited.
TEMPLATE_MIN_LENGTH, TEMPLATE_MAX_LENGTH = 4, 15  # the labelled evidence peptides are short (median 7)
TEMPLATE_MIN_GUIDE_SIM = 0.5  # nearest known positive must be at least this similar (and < 1.0)
N_TEMPLATES = 150  # diverse templates, best guide similarity first
EXCLUDE_TEMPLATE_RESIDUES = "C"  # no variant of a cysteine fragment can pass the synthesis filter

# Guides: known positives that steer the edits. Long guides inflate the similarity of a short fragment; non-food hormones are not food-derived.
GUIDE_MAX_LENGTH = 15
GUIDE_EXCLUDE_FOOD_DERIVATION = ("non_food_hormone",)

# Edits.
MAX_EDITS = 2  # per peptide
CONSERVATIVE_MIN_BLOSUM = 1  # substitution allowed when BLOSUM62(old, new) >= this
REQUIRE_STRICT_IMPROVEMENT = True  # the edit must make the peptide MORE similar to its guide than the template is
ALLOW_DELETION = True  # one residue, at most one per peptide
MAX_CANDIDATES_PER_TEMPLATE = 25  # cheap-filter survivors sampled (seeded) before the digestion check

# Constraints.
MIN_LENGTH, MAX_LENGTH = 4, 20  # generated peptides (screening accepts 4-40)
CHARGE_LIMIT = 4.0  # |net charge at pH 7.4|
DOMINANT_RESIDUE_LIMIT = 0.4  # no residue above this fraction
SYNTHESIS_BLOCKING = ("cysteine", "homopolymer_run", "long_for_synthesis", "hydrophobic_solubility_risk")  # these liabilities fail a candidate
MAX_LIABILITIES = 2
PLAUSIBILITY_PAD = 10  # residues of parent context digested on each side

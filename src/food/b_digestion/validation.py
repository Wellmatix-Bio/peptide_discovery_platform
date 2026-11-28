"""What the benchmark says about each protease rule, for the digestion registry. The figures are the balanced accuracy and its lower 95% bound from
docs/validation/food/protease_validation.md (`protease_validation.json`); a test keeps these strings equal to that file. The bar was a lower bound of 0.80 on at least 50 proteins."""

RULE_VALIDATION = {
    "trypsin": "validated: balanced accuracy 0.936, lower bound 0.933, on 7,525 observed MEROPS sites",
    "chymotrypsin_low": "validated: balanced accuracy 0.897, lower bound 0.884, on 150 observed MEROPS sites",
    "chymotrypsin_high": "not validated: balanced accuracy 0.780, lower bound 0.737 (bar 0.80); misses 38% of observed sites",
    "pepsin_ph1.3": "not validated: balanced accuracy 0.711, lower bound 0.700, on yeast peptidomics sites; MEROPS has too few pepsin proteins to test",
    "pepsin_ph2.0": "not validated: balanced accuracy 0.739, lower bound 0.728, on yeast peptidomics sites; MEROPS has too few pepsin proteins to test",
}

LIMITATIONS = [
    "A cleavage site is a hypothesis from a rule. It says nothing about abundance, kinetics or intestinal exposure.",
    "These fragments are theoretical until LC-MS/MS observes them.",
    "The pepsin rules look at two residues each side of a bond, so a peptide of 4 residues has no bond pepsin can cut, and one of 6 has 2 of 5.",
    "A predicted cleavage probability of zero would not guarantee survival in the gut; these rules give no probability.",
]

"""Job-config constants for the cosmetic worker: the cosmetic stages replace their wound-healing peers."""

#: Cosmetic stages, in pipeline order; each is forced on.
COSMETIC_STAGES = (
    "s01_cosmetic_therapeutic_product_brief",
    "s04_cosmetic_candidate_generation",
    "s05_cosmetic_physchem_screening",
    "s06_cosmetic_functional_models",
    "s08_cosmetic_safety_developability",
    "s09_cosmetic_synthesis_cmc",
    "s11_cosmetic_ranking",
)

#: Wound-healing peers the cosmetic stages replace; each is forced off. Stage 7 is shared and stays on.
COSMETIC_REPLACED_STAGES = (
    "s01_therapeutic_product_brief",
    "s02_wound_biology_and_targets",
    "s03_data_integration",
    "s04_candidate_generation",
    "s05_physchem_screening",
    "s06_functional_models",
    "s08_safety_developability",
    "s09_synthesis_cmc",
    "s11_ranking",
)

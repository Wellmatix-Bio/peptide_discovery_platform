"""The self-contained run config of a food run: the platform's `RunConfig` plus where the submitted proteins are; written by the API, read by the food worker."""

from __future__ import annotations

from schemas.run_config import RunConfig


#: One file per weight folder the food worker needs in the model store, relative to it. Checked when a job is created, so a bucket that lacks a model fails the
#: request instead of a GPU job minutes after it starts. (ESM2 embeds the peptides; each model after it has the file that proves its folder is there.)
FOOD_REQUIRED_WEIGHT_FILES = [
    "esm2_t30_150M/config.json",
    "allergenicity_predictor_v1/ensemble.json",
    "cytotoxicity_predictor_v1/cytotoxicity_predictor_fold0.pt",
    "solubility_predictor_v1/solvent_descriptors.json",
    "aggregation_predictor_v1/feature_columns.json",
    "synthesis_feasibility_predictor_v1/svm.pkl",
    "bitterness_predictor_v1/classification_thresholds.json",
]


class FoodRunConfig(RunConfig):
    seed_candidates_path: str = ""  # a food run starts from its proteins, not from seed candidates
    fasta_path: str  # the submitted FASTA, stored in the run directory
    food_source: str = ""  # applies to every record whose header carries no FS= (empty for a run of library proteins, whose headers carry it)
    allergen_source: bool | None = None  # for a protein with no source record; None: not established, never False by default
    source_records_path: str = ""  # source records (family, allergen-source flag, version, release) for the proteins the library matched or supplied

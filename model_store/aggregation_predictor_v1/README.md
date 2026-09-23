---
tags:
  - peptide
  - aggregation
  - xgboost
  - aaindex
library_name: xgboost
---

# aggregation-predictor-v1

Aggregation-propensity classifier for Stage 8 (Safety & Developability) — flags candidates likely to self-aggregate/precipitate, a manufacturability and formulation risk independent of binding affinity.

## Architecture

Single tuned XGBoost classifier (`binary:logistic`) over the top-300 (by gain importance) of ~1490 sequence descriptors: biopython global descriptors, propy3 CTD/autocorrelation/QSO/SOCN/PAAC/APAAC descriptors, and AAindex1-scale summary statistics (mean/std/min/max over 19 curated physicochemical scales). Predicts probability of the `seed_bh` aggregation label.

Held-out test performance (10,306 rows): ROC-AUC 0.700, PR-AUC 0.446, balanced accuracy 0.642, F1 0.438, MCC 0.280.

## Files

- `xgb_seed_bh.joblib` — the trained XGBoost classifier (300 input features, tuned via `RandomizedSearchCV`)
- `feature_columns.json` — the exact 300 feature names and order the model expects
- `aaindex1_scales.json` — the 19 AAindex1 physicochemical scales (accession-keyed, one value per standard amino acid) needed by the `ps_*` features, extracted from the source project's `aaindex1` database file so this package is self-contained
- `model_card.json` — feature sources, XGBoost hyperparameters and search space, test metrics

## Usage

```python
from aggregation_predictor_v1 import AggregationPredictor

predictor = AggregationPredictor()
prob = predictor.predict_aggregation("KLAIVLLKLAIVL")
```

Requires `propy3` (imported as `propy`) and `biopython` in addition to `xgboost`/`joblib`/`pandas`/`numpy`. Sequences must be standard amino acids only, length >= 5 (QSO/SOCN/PAAC/APAAC lag=4 requirement).

## Provenance

Exported from `Models/Aggregation/models/A` (pipeline: `pipelineA.ipynb`). Trained on `seed_bh` labels from `data/adt5111_data_files_s1_to_s7.xlsx` (sheet `SD_2`). See `performance_improvements.md` in the source project for the tuning history (feature selection top-300, `RandomizedSearchCV` over XGBoost hyperparameters, programmatic model selection).

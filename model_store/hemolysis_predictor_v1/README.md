---
tags:
  - peptide
  - hemolysis
  - random-forest
license: mit
---

# hemolysis-predictor-v1

Random Forest hemolysis regressor (pHC50) over hand-computed sequence descriptors. It is the hemolysis model used by the wound-healing pipeline:

- Stage 8 (`s08_safety_developability`)
- Route A's GA fitness in Stage 4 (`s04_generation/routeA.py`)
- the shared `FeatureExtractor`, which caches this model's 1167-column descriptor rows

The metabolism pipeline does **not** use it; `s08_metabolic_safety_developability` and `s11_metabolic_ranking` use `hemolysis_predictor_v2` (an ESM2-based ensemble with a classifier and a regressor).

## What this is

An in-process, native Python/sklearn model, ported verbatim from `Models/Hemolysis prediction/paper_replicated.ipynb`. It reproduces the published HemoPI2 regression approach (same model family and feature set) but is trained independently on the published cross-validation/test splits, so there is no subprocess and no vendored GPL code.

**Validated against the paper's reported benchmark** on the independent test set: Pearson r = 0.740, R² = 0.544 (paper's targets: r = 0.739, R² = 0.543). That evaluation is on the paper's own split; the later `training.ipynb` work found that kind of random split leaky and reports lower, audited scores for its own models, so do not read 0.74 as expected accuracy on new peptide families.

## Architecture

`Pipeline([('imputer', SimpleImputer(strategy='median')), ('rf', RandomForestRegressor(n_estimators=800, max_depth=30, ...))])`, trained on 1167 hand-computed descriptors per sequence: molecular weight, length, AAC, DPC, ATC, BTC, PCP, RRI, PRI, DDR, SER, SEP, CTC, CeTD, PAAC, APAAC, QSO, SOC, the same descriptor groups the original feature-extraction script computes.

`predictor.py`'s `Extractor` class reimplements each descriptor group's exact arithmetic (including a couple of faithfully preserved quirks/bugs in the original author script, see the inline comments on `_feature_atc` and `_feature_rri`, since the released model was trained on that exact output).

## pHC50 conversion

The model predicts `y = -log10(HC50 in uM)` directly (not this codebase's `pHC50 = -log10(HC50 in M)` convention). `predictor.py` converts:

    pHC50 = y_pred + 6.0

HIGH pHC50 = hemolytic at LOW concentration = worse.

## Files

- `model.joblib` (~26MB) — the trained `Pipeline`, copied from `Models/Hemolysis prediction/models/regressor/v2/model.joblib` (already trained and cached by the notebook; not retrained here). Lives in `model_store/model_weights/hemolysis_predictor_v1/`.
- `property_tables.py` — the seven embedded CSV/TSV property tables the descriptor extractor needs (amino-acid attribute groups, atom/bond composition, physicochemical index, Schneider-Wrede and Grantham distance matrices), verbatim from the notebook.
- `predictor.py` — `Extractor` (descriptor computation), `normalize_sequences`, and `ModifiedHemolyticPredictor` (`predict_phc50(sequence, feature_extractor)` / `predict_phc50_batch(sequences, feature_extractor)`).
- `model_card.json` — metrics and training data summary.

## Usage

```python
from model_store.hemolysis_predictor_v1 import ModifiedHemolyticPredictor

predictor = ModifiedHemolyticPredictor()
# feature_extractor is the shared FeatureExtractor; it caches the descriptor rows.
phc50_values = predictor.predict_phc50_batch(sequences, feature_extractor)  # one extraction pass + one model.predict() call
```

Valid input is 6 to 50 standard residues.

## Provenance

- Source notebook: `Models/Hemolysis prediction/paper_replicated.ipynb`
- Training data: the published HemoPI2 `cross_val_dataset.csv` / `independent_dataset.csv` (Rathore et al., Commun Biol 8, 176 (2025); data on Zenodo, https://doi.org/10.5281/zenodo.14676712)
- Trained/cached 2026-09-18 with scikit-learn 1.9.0. Loading under the platform's scikit-learn 1.6.1 emits an `InconsistentVersionWarning`; the model card's metrics were not re-verified under 1.6.1.

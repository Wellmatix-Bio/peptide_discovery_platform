---
tags:
  - peptide
  - antimicrobial-peptide
  - xgboost
  - esm2
library_name: xgboost
---

# amp-classifier-v1

Antimicrobial-peptide (AMP) probability classifier used by Stage 6 (Functional AI Models) and as the default `amp_predictor` in Route A's GA (`src/pipeline/s04_generation/routeA.py`).

## Architecture

5-fold XGBoost ensemble over ESM2 (`facebook/esm2_t33_650M_UR50D`) mean-pooled embeddings (1280 dims) concatenated with modlAMP global/moment descriptors (21 dims), stacked through a meta-model (`meta_model.joblib`) that calibrates the fold-averaged raw score into a probability in `[0, 1]`.

## Files

- `model_card.json` — feature columns, XGBoost hyperparameters, training data paths, ESM2 model name
- `xgb_fold{0-4}.json` — the 5 fold models
- `meta_model.joblib` — probability calibrator over fold-averaged predictions
- `oof_predictions.npz` — out-of-fold predictions used to fit the meta-model

## Usage

Loaded lazily by `RealAMPPredictor` in `routeA.py`; see `predict_amp(sequence)`.

## Provenance

Exported from `Models/AMP classification/models/model1`.

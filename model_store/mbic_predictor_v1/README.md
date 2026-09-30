---
tags:
  - peptide
  - antibiofilm
  - mbic
  - esm2
  - svr
  - random-forest
library_name: scikit-learn
---

# mbic-predictor-v1

Minimum Biofilm Inhibitory Concentration (MBIC) potency regressor, for use in Stage 6 (Functional AI Models) antimicrobial/antibiofilm scoring.

**Note:** this model outputs `pMBIC`, NOT a raw concentration and NOT a `[0, 1]` probability. `pMBIC = 6 - log10(activity_uM)` (a pIC50-style log-scale transform; higher pMBIC = more potent, i.e. a lower MBIC). To recover a concentration:

```
activity_uM = 10 ** (6 - pMBIC)
```

Downstream consumers must apply this inverse transform (and/or a probability calibrator, analogous to `hemolysis_predictor_v1`) before combining this score with `[0, 1]`-scaled objectives in multi-objective ranking (Stage 11) — do not feed raw `pMBIC` into a weighted sum alongside probabilities.

## Architecture

SVR + RandomForest ensemble (unweighted mean of the two regressors' `pMBIC` predictions), each over a 58-dim concatenated feature vector:

1. **Species one-hot** (13 dims) — `OneHotEncoder(handle_unknown="ignore")` over `target_species`. A species outside the training vocabulary (see `model_card.json`) does **not** raise an error — it silently becomes an all-zero vector (no species signal). Treat predictions for out-of-vocabulary species with extra caution; this is a real model input, not an optional one, so `predict_pmbic()` requires it.
2. **modlAMP physicochemical descriptors** (14 dims, standardized) — charge/charge_density (pH 7.4, Bjellqvist), gravy, Eisenberg hydrophobic moment (18-residue/100° window, max modality), a custom amphipathicity (`hydrophobic_moment / mean(|H|)`), helical propensity (levitt_alpha), aliphatic index, aromaticity, Boman index, instability index, isoelectric point, hydrophobic ratio, length, and `d_residue_fraction` (fraction of lowercase/D-amino-acid positions in the raw sequence).
3. **ESM2-150M embeddings** (`facebook/esm2_t30_150M_UR50D`), mean-pooled excluding BOS/EOS/padding, standardized, then reduced via `PCA(n_components=0.95)` — 27 components on the full 147-row fit.

- `SVR(kernel=rbf, C=10.0, epsilon=0.1, gamma=scale)`
- `RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42)`

Each regressor carries its own fitted preprocessing bundle (they are fit identically to each other in this export, but `predictor.py` applies them independently rather than assuming that always holds). Preprocessing order at inference (must match training exactly): physchem descriptors → `physchem_scaler` → ESM2 embedding → `esm2_scaler` → `esm2_pca` → species one-hot → concatenate `[species_onehot, physchem_scaled, esm2_pcs]` → `<model>.predict`.

## Files

- `svr.joblib` — dict with keys `model` (fitted `SVR`) and `preprocessing` (`species_ohe`, `physchem_scaler`, `esm2_scaler`, `esm2_pca`, `physchem_cols`).
- `random_forest.joblib` — same structure, `model` is a fitted `RandomForestRegressor`.
- Both are self-contained; do not call `.predict()` on `model` without first applying `preprocessing`.
- `model_card.json` — feature list/order, ESM2 model name, PCA variance target, species vocabulary, CV metrics.

## Usage

```python
from mbic_predictor_v1 import MBICPredictor

predictor = MBICPredictor()
pmbic = predictor.predict_pmbic("GLLDIIKKIAESF", species="Pseudomonas aeruginosa")
activity_uM = 10 ** (6 - pmbic)
```

Loaded lazily; nothing loads until the first `predict_pmbic()` call.

## Provenance

Exported from `Models/MBIC prediction/pipeline_ESM35M.ipynb` (despite the filename, this run embeds with `facebook/esm2_t30_150M_UR50D`, not the 35M checkpoint).

Trained on 147 sequence/species rows reconciled from BAAMP (`activity == "Formation (3-24h)"`, `method_activity == "> 90%"`) and DBAASP (`antibiofilm_activities` exploded, `Activity Measure == "MBIC"`), unit-standardized to µM (direct µM/mM pass-through or `ug/mL -> uM` via modlAMP molecular weight; `nmol/cm^2` surface-density rows had no valid concentration equivalent and were excluded).

**Both models are deployed as an ensemble.** Random Forest and SVR were compared under 10-fold CV (`KFold(shuffle=True, random_state=42)` over all 147 rows; species one-hot encoder, both `StandardScaler`s, and PCA refit inside each fold on that fold's training rows only, to avoid leakage):

| Model | RMSE | MAE | R² | Spearman | Pearson |
|---|---|---|---|---|---|
| Random Forest | 0.525 ± 0.125 | 0.385 ± 0.078 | 0.415 ± 0.288 | 0.653 ± 0.175 | 0.671 ± 0.194 |
| SVR | 0.525 ± 0.112 | 0.382 ± 0.078 | 0.418 ± 0.257 | 0.646 ± 0.157 | 0.677 ± 0.194 |

RF and SVR are statistically indistinguishable under CV. Unlike the prior export (which shipped SVR alone), both are now averaged at inference to combine their independent error profiles rather than picking one.

Both models were refit on all 147 rows (not just the CV-reference train split) before being cached, alongside preprocessing objects refit the same way — this is the exact object graph saved in `svr.joblib`/`random_forest.joblib`, `54` feature columns before the ESM2 block is PCA-reduced (`Final feature matrix: (147, 54) (27 ESM2 PCs)` per the source notebook's own cache-writing cell).

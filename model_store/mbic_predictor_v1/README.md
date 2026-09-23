---
tags:
  - peptide
  - antibiofilm
  - mbic
  - esm2
  - svr
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

Single SVR (`kernel=rbf, C=10.0, epsilon=0.1, gamma=scale`) over a 58-dim concatenated feature vector:

1. **Species one-hot** (13 dims) — `OneHotEncoder(handle_unknown="ignore")` over `target_species`. A species outside the training vocabulary (see `model_card.json`) does **not** raise an error — it silently becomes an all-zero vector (no species signal). Treat predictions for out-of-vocabulary species with extra caution; this is a real model input, not an optional one, so `predict_pmbic()` requires it.
2. **modlAMP physicochemical descriptors** (14 dims, standardized) — charge/charge_density (pH 7.4, Bjellqvist), gravy, Eisenberg hydrophobic moment (18-residue/100° window, max modality), a custom amphipathicity (`hydrophobic_moment / mean(|H|)`), helical propensity (levitt_alpha), aliphatic index, aromaticity, Boman index, instability index, isoelectric point, hydrophobic ratio, length, and `d_residue_fraction` (fraction of lowercase/D-amino-acid positions in the raw sequence).
3. **ESM2-650M embeddings** (`facebook/esm2_t33_650M_UR50D`), mean-pooled excluding BOS/EOS/padding, standardized, then reduced via `PCA(n_components=0.95)` — 31 components on the full 147-row fit.

Preprocessing order at inference (must match training exactly): physchem descriptors → `physchem_scaler` → ESM2 embedding → `esm2_scaler` → `esm2_pca` → species one-hot → concatenate `[species_onehot, physchem_scaled, esm2_pcs]` → `SVR.predict`.

## Files

- `svr.joblib` — dict with keys `model` (fitted `SVR`) and `preprocessing` (`species_ohe`, `physchem_scaler`, `esm2_scaler`, `esm2_pca`, `physchem_cols`). Self-contained; do not call `.predict()` on `model` without first applying `preprocessing`.
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

Exported from `Models/MBIC prediction` (authoritative notebook: `pipeline.ipynb`, confirmed via its `joblib.dump` calls — `pipeline copy.ipynb` and `pipeline_bose_paper.ipynb` are divergent experiments that produced no saved artifacts and were not used).

Trained on 147 sequence/species rows reconciled from BAAMP (`activity == "Formation (3-24h)"`, `method_activity == "> 90%"`) and DBAASP (`antibiofilm_activities` exploded, `Activity Measure == "MBIC"`), unit-standardized to µM (direct µM/mM pass-through or `ug/mL -> uM` via modlAMP molecular weight; `nmol/cm^2` surface-density rows had no valid concentration equivalent and were excluded).

**Model selection:** Random Forest and SVR were compared under 10-fold CV (`KFold(shuffle=True, random_state=42)` over all 147 rows; species one-hot encoder, both `StandardScaler`s, and PCA refit inside each fold on that fold's training rows only, to avoid leakage):

| Model | RMSE | MAE | R² | Spearman | Pearson |
|---|---|---|---|---|---|
| Random Forest | 0.513 ± 0.112 | 0.382 ± 0.074 | 0.447 ± 0.241 | 0.652 ± 0.150 | 0.692 ± 0.164 |
| **SVR (exported)** | 0.516 ± 0.112 | 0.385 ± 0.077 | 0.439 ± 0.248 | 0.627 ± 0.168 | 0.690 ± 0.174 |

RF and SVR were statistically indistinguishable under CV — RF had marginally better numbers on every metric, but the gap is within noise at this sample size. **SVR was chosen per explicit decision** (full ESM2 + descriptors + species pipeline), not because it outperformed RF. `random_forest.joblib` exists alongside `svr.joblib` in the source project but was intentionally not exported.

Both models were refit on all 147 rows (not just the 115-row CV-reference train split) before being cached, alongside preprocessing objects refit the same way — this is the exact object graph saved in `svr.joblib`.

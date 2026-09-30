---
tags:
  - peptide
  - solubility
  - xgboost
  - esm2
library_name: xgboost
---

# solubility-predictor-v1

Peptide-in-solvent solubility probability classifier. Predicts `P(soluble)` for a peptide sequence in one of 7 known solvents, for use in Stage 5 (Sequence/Physicochemical Screening) and Stage 9 (Delivery/Formulation Co-Design).

## Architecture

Single XGBoost classifier (`xgboost.XGBClassifier`) over a 1286-dim flat feature vector: 1280-dim mean-pooled ESM2 (`facebook/esm2_t33_650M_UR50D`) embedding (per-residue hidden states, BOS/EOS excluded from the mean) concatenated with a 6-dim solvent descriptor block (`pH_value`, `pH_missing_flag`, `ionic_strength`, `dielectric_constant`, `protic_aprotic`, `polarity_index`) looked up per solvent. No PCA — this is the "raw" feature variant (Pipeline B, `B_raw`).

## Files

- `pair_stratified_87a389cf6ee76d08.joblib` — the deployed XGBoost model (see Provenance for why this specific split was chosen)
- `solvent_descriptors.json` — bundled 7-solvent lookup table (pH, ionic strength, dielectric constant, protic/aprotic, polarity index), self-contained copy of `solvent_descriptors.csv`
- `model_card.json` — feature columns, XGBoost hyperparameters, solvent vocabulary, deployment artifact rationale

## Usage

```python
from solubility_predictor_v1 import SolubilityPredictor

predictor = SolubilityPredictor()
p_soluble = predictor.predict_proba("SGLEQLESIINFEKL", solvent="1X DPBS")
```

`solvent` is validated against the bundled 7-solvent vocabulary (case/spacing-normalized); an unrecognized solvent raises `ValueError`.

## Provenance

Exported from `Models/Solubility` (`pipelineB.ipynb`, raw feature variant, `model_cache/B_raw/`).

**Important caveats carried over from the prior investigation of this project:**

- `interp.md` in the source project describes an unrelated continuous "difficulty_coefficient" regression problem and does **not** describe what was actually built. Ignore it. The real pipelines train binary classifiers on `data/long_peptide_solvent.csv` (long-format, one row per peptide-solvent pair, 4405 rows / 1337 unique peptides / 7 solvents) predicting `y` (0/1 = Insoluble/Soluble).
- Reported performance for this pipeline B raw variant: **LOSO (leave-one-solvent-out) macro AUC = 0.773, macro accuracy = 0.775**.
- **DMSO predictions are near chance-level** (LOSO AUC ~0.56) even in this best-performing variant — treat DMSO solubility predictions from this model with low confidence.
- **3 of 7 solvents have zero LOSO validation coverage**: 0.2 M acetic acid, 3% ammonia water (v/v), and HCOOH are 100% single-class (all-soluble) in the training data, so they were skipped in leave-one-solvent-out evaluation entirely. The model can still be queried for these solvents (solvent descriptors are defined and the model was trained on some in-solvent rows for them via the other splits), but there is no held-out evidence of generalization for them.

**Deployment artifact selection — read before trusting this as a "trained on everything" model:**

`pipelineB.ipynb` was read in full to determine whether a final refit-on-all-data model exists. It does not. The notebook's B_raw evaluation loop produces exactly 6 cached files, all evaluation/validation folds:
- `pair_stratified_<hash>.joblib` — random 64/16/20 stratified train/val/test split
- `seq_disjoint_<hash>.joblib` — `GroupShuffleSplit` on `sequence_id`, held-out peptides never seen in training
- `loso_<solvent>_<hash>.joblib` × 4 — leave-one-solvent-out, for the 4 solvents with both classes present (DMSO, 1X DPBS, 0.1 M PBS, Ultrapure water)

The notebook ends immediately after the LOSO summary cell (cell 20 of 21); there is no subsequent cell that refits on the full dataset with no held-out portion, cached or otherwise.

Per the documented fallback rule, **`pair_stratified_87a389cf6ee76d08.joblib` was selected as the deployment artifact.** It was not chosen because it is ideal — it still holds out 20% test + a further validation slice — but because among the six evaluation-fold models it uses the largest training portion and is a standard random split rather than a group-held-out one (unlike `seq_disjoint` and the `loso_*` files, which deliberately withhold entire peptides or an entire solvent from training). This is the closest available approximation to a general-purpose model, but it is explicitly a **fallback choice, not a true full-data-refit deployment model**. If a full-data refit is later produced, this export should be superseded.

Not used: `model_cache/A` (pipeline A, hand-crafted descriptor variant) and `model_cache/B_pca` (PCA-reduced embedding variant — the fitted PCA transform was never persisted alongside the cached model, so its features cannot be reproduced at inference time; unresolved reproducibility gap, do not use).

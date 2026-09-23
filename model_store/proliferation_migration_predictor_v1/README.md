---
tags:
  - peptide
  - proliferation
  - migration
  - voting-classifier
  - random-forest
  - xgboost
  - logistic-regression
library_name: scikit-learn
---

# proliferation-migration-predictor-v1

Cell-proliferation-dominant / migration-dominant probability classifier used by Stage 6 (Functional AI Models) to score the regeneration function axis (cell proliferation/migration) of the five target peptide functions.

## Architecture

Two independent soft-voting `VotingClassifier` ensembles (`RandomForestClassifier` + `XGBClassifier` + `StandardScaler`→`LogisticRegression`), one per binary target: `migration_dominant` and `proliferation_dominant`. Each ensemble consumes a 15-dimensional hand-built physicochemical feature vector (length, net charge, Kyte-Doolittle hydrophobicity mean, Eisenberg hydrophobic moment, aromatic/cationic/anionic/polar/aliphatic/proline/glycine/cysteine fractions, charge density, isoelectric proxy, hydrophobic fraction) — no ESM2 embeddings, no modlAMP calls.

**Ensemble hyperparameters (identical for both targets, `random_state=0`):**
- `RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=0)`
- `XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=0)`
- `Pipeline(StandardScaler() -> LogisticRegression(max_iter=1000, C=1.0, random_state=0))`
- `VotingClassifier(estimators=[rf, xgb, logreg], voting="soft")`

## Files

- `migration_dominant_ensemble.joblib` — fitted VotingClassifier for `migration_dominant`
- `proliferation_dominant_ensemble.joblib` — fitted VotingClassifier for `proliferation_dominant`
- `model_card.json` — feature list, hyperparameters, notebook LOO-CV metrics

## Usage

Loaded lazily by `ProliferationMigrationPredictor`; see `predict(sequence) -> {"migration_dominant": float, "proliferation_dominant": float}`.

## Provenance

Exported from `Models/Proliferation + Migration prediction/pipeline.ipynb`, trained on `data/peptide_activity_unique_clean.csv` (103 clean sequences after filtering to standard amino acids).

**Important — full-data refit, not the notebook's evaluated object.** The notebook itself never fits or saves a persistent model: `build_ensemble()` is instantiated fresh inside every Leave-One-Out CV fold, fit on that fold, used once to predict the held-out row, and discarded. No `joblib.dump`/`pickle` call exists anywhere in the notebook.

The two `.joblib` artifacts here were produced by a separate script (not part of the notebook, not checked into the source project) that reproduces the notebook's feature engineering and `build_ensemble()` function exactly, then fits each ensemble **once on the full 103-row dataset** (no LOO holdout) so it can be persisted and served. This is a deliberate deviation, authorized for deployment purposes — the exact same feature code and hyperparameters were used, so the only difference from any single notebook LOO-fold model is training-set size (full data vs. n-1 rows).

Because the deployed object is a full-data refit, it has **no held-out evaluation of its own**. The best available generalization estimate is the notebook's own Leave-One-Out CV, run per-fold with fresh `build_ensemble()` instances (103 folds each target):

| target | ACC | MCC | F1 | AUC |
|---|---|---|---|---|
| migration_dominant | 0.6505 | 0.2934 | 0.6087 | 0.6797 |
| proliferation_dominant | 0.7087 | 0.2644 | 0.8026 | 0.7328 |

These numbers describe the notebook's per-fold LOO models, not a direct evaluation of the shipped full-data-refit object.

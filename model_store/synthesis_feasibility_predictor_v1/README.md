---
tags:
  - peptide
  - synthesis-feasibility
  - esm2
  - scikit-learn
library_name: scikit-learn
---

# synthesis-feasibility-predictor-v1

Synthesis-feasibility probability classifier for candidate peptides, used by Stage 9 (Synthesis/CMC Feasibility).

## Architecture

Soft-voting ensemble of 3 scikit-learn classifiers over ESM2 (`facebook/esm2_t30_150M_UR50D`) mean-pooled embeddings (640 dims, CLS/EOS excluded from pooling, no normalization): Logistic Regression, SVM (RBF kernel, `probability=True`), and Random Forest. The LR and SVM artifacts are full `sklearn.pipeline.Pipeline`s with `StandardScaler` baked in; Random Forest is a bare `RandomForestClassifier` (tree models don't need scaling). Final probability is the unweighted mean of each model's `predict_proba(X)[:, 1]`.

## Files

- `model_card.json` — ESM2 config, per-model hyperparameters, CV metrics
- `logistic_regression.pkl` — StandardScaler + LogisticRegression pipeline
- `svm.pkl` — StandardScaler + SVC(probability=True) pipeline
- `random_forest.pkl` — RandomForestClassifier

## Usage

```python
from synthesis_feasibility_predictor_v1 import SynthesisFeasibilityEnsemble

predictor = SynthesisFeasibilityEnsemble()
prob = predictor.predict_proba("GIGKFLHSAKKFGKAFVGEIMNS")
```

## Provenance

Exported from `Models/Sythesis feasibility/models/combined/v2` (`pipeline_combined_35M_embedding.ipynb`; despite the filename, this run embeds with `facebook/esm2_t30_150M_UR50D`, not a 35M checkpoint), trained on the combined dataset (PepSySco.csv merged with MS2-labeled peptides, 3,560 rows total).

**Ensemble selection:** Logistic Regression, SVM, and Random Forest were selected as an explicit user choice out of the 5 model types the notebook trains (KNN and Decision Tree excluded).

**Performance (5-fold StratifiedKFold CV, mean across folds):**

| Model | ROC-AUC | F1 | Accuracy | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.7761 | 0.7340 | 0.7107 | 0.4171 |
| SVM | 0.7597 | 0.7273 | 0.6927 | 0.3788 |
| Random Forest | 0.7351 | 0.7079 | 0.6652 | 0.3221 |

SVM probability calibration was checked at export time: `svm.pkl`'s `SVC` step has `probability=True`, so `predict_proba` is valid (no `decision_function`/sigmoid fallback needed).

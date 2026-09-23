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

Soft-voting ensemble of 3 scikit-learn classifiers over ESM2 (`facebook/esm2_t33_650M_UR50D`) mean-pooled embeddings (1280 dims, CLS/EOS excluded from pooling, no normalization): Logistic Regression, SVM (RBF kernel, `probability=True`), and Random Forest. The LR and SVM artifacts are full `sklearn.pipeline.Pipeline`s with `StandardScaler` baked in; Random Forest is a bare `RandomForestClassifier` (tree models don't need scaling). Final probability is the unweighted mean of each model's `predict_proba(X)[:, 1]`.

## Files

- `model_card.json` — ESM2 config, per-model hyperparameters, CV and independent-validation metrics
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

Exported from `Models/Sythesis feasibility/models/` (base data variant, produced by `pipeline.ipynb`, not `pipeline_combined.ipynb`/`models/combined/`).

**Data variant decision:** the base variant (PepSySco.csv, 1,771 peptides) was chosen over the combined/3,560-peptide variant per explicit user instruction to defer to "whichever path is indicated in the notebook" — `pipeline.ipynb` is the notebook that includes a self-contained independent-validation step (evaluating cached models against an MS2-PSM-derived out-of-distribution set), making it the more complete, self-checking pipeline. `pipeline_combined.ipynb` has no equivalent validation step.

**Ensemble selection:** Logistic Regression, SVM, and Random Forest were selected as an explicit user choice out of the 5 model types the notebook trains (KNN and Decision Tree excluded).

**Performance — read this before trusting the score:**

| Model | 5-fold CV ROC-AUC | Independent-validation ROC-AUC |
|---|---|---|
| Logistic Regression | 0.727 | 0.59 |
| SVM | 0.737 | 0.57 |
| Random Forest | 0.737 | 0.56 |

CV performance (≈0.73 ROC-AUC for all three) looked reasonable, but on the independent validation set — peptides labeled by MS2 spectral-count success ratio from a separate synthesis-error dataset, not drawn from the same distribution as PepSySco.csv — ROC-AUC dropped to near-chance (0.56–0.59). **This ensemble's real-world generalization is weak and unverified beyond this single out-of-distribution check.** Downstream consumers should treat `predict_proba` output as a weak prior at best, not a reliable feasibility signal, and should not use it as a hard gate without additional validation.

SVM probability calibration was checked at export time: `svm.pkl`'s `SVC` step has `probability=True`, so `predict_proba` is valid (no `decision_function`/sigmoid fallback needed).

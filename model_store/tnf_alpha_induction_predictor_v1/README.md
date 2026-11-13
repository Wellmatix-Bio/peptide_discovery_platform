---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# tnf-alpha-induction-predictor-v1

**First version.** P(peptide induces TNF-alpha) from sequence, for the immunomodulation studio. Mean of 3 classifiers on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| MLP_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.233 | 0.664 | 0.660 | 0.617 | 0.589 | 0.644 |
| SVM_RBF_smote | SMOTE oversampling of the training rows | 0.237 | 0.674 | 0.681 | 0.619 | 0.616 | 0.621 |
| XGBoost_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.238 | 0.672 | 0.672 | 0.619 | 0.611 | 0.627 |
| **ensemble** | recomputed on the notebook's folds with these members (out-of-fold mean of member probabilities) | 0.259 | 0.682 | 0.681 | 0.629 | 0.616 | 0.642 |

Trained on 3208 peptides (50% positive), lengths 7-50.

## Limits

- The training notebook does not evaluate this ensemble; its cross-validated metrics here were computed on the notebook's own folds with the same members.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 7-50 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.tnf_alpha_induction_predictor_v1 import TnfAlphaInductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = TnfAlphaInductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/tnf_alpha_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# il6-induction-predictor-v1

**First version.** P(peptide induces IL-6) from sequence, for the immunomodulation studio. Mean of 3 classifiers on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| XGBoost_lenbin | length-bin x class sample weights | 0.378 | 0.739 | 0.439 | 0.617 | 0.247 | 0.987 |
| MLP_adasyn | ADASYN oversampling of the training rows | 0.348 | 0.733 | 0.407 | 0.647 | 0.337 | 0.957 |
| SVM_RBF_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.356 | 0.746 | 0.431 | 0.590 | 0.186 | 0.994 |
| **ensemble** | recomputed on the notebook's folds with these members (out-of-fold mean of member probabilities) | 0.409 | 0.752 | 0.471 | 0.636 | 0.288 | 0.984 |

Trained on 3356 peptides (11% positive), lengths 8-25.

## Limits

- The notebook's own IL-6 ensemble uses MLP_posw; this version uses MLP_adasyn as requested. Its cross-validated metrics here were computed on the notebook's folds with these members.
- Trained only on 8-25 residue peptides.
- Sensitivity is low at the 0.5 threshold; the training set is 11% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-25 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.il6_induction_predictor_v1 import Il6InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il6InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/il6_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

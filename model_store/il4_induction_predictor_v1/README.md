---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# il4-induction-predictor-v1

**First version.** P(peptide induces IL-4) from sequence, for the immunomodulation studio. A single classifier on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| XGBoost_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.252 | 0.668 | 0.686 | 0.621 | 0.755 | 0.486 |
| **model** |  | 0.252 | 0.668 | 0.686 | 0.621 | 0.755 | 0.486 |

Trained on 1276 peptides (55% positive), lengths 7-45.

## Limits

- A single XGBoost member. The notebook's own three-model ensemble scored no better (MCC 0.226, AUROC 0.672 against 0.252, 0.668).
- Sensitivity (0.75) is well above specificity (0.49): it over-calls induction.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 7-45 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.il4_induction_predictor_v1 import Il4InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il4InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/il4_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

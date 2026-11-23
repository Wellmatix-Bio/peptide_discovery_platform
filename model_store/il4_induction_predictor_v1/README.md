---
tags:
  - peptide
  - cytokine
  - immunomodulation
---

# il4-induction-predictor-v1

**First version.** P(peptide induces IL-4) from sequence, for the immunomodulation studio.

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|
| 0.252 | 0.668 | 0.686 | 0.621 | 0.755 | 0.486 |

Trained on 1276 peptides (55% positive), lengths 7-45.

## Limits

- Sensitivity (0.75) is well above specificity (0.49): it over-calls induction.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 7-45 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.

## Usage

```python
from model_store.il4_induction_predictor_v1 import Il4InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il4InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, in_length_range, applicability
```

## Files

Code and weights are kept privately; weights live in `model_store/model_weights/il4_induction_predictor_v1/`. Not wired into any pipeline stage or API yet.

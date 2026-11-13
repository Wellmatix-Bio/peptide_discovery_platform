---
tags:
  - peptide
  - cytokine
  - immunomodulation
---

# tnf-alpha-induction-predictor-v1

**First version.** P(peptide induces TNF-alpha) from sequence, for the immunomodulation studio.

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|
| 0.259 | 0.682 | 0.681 | 0.629 | 0.616 | 0.642 |

Trained on 3208 peptides (50% positive), lengths 7-50.

## Limits

- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 7-50 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.

## Usage

```python
from model_store.tnf_alpha_induction_predictor_v1 import TnfAlphaInductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = TnfAlphaInductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, in_length_range, applicability
```

## Files

Code and weights are kept privately; weights live in `model_store/model_weights/tnf_alpha_induction_predictor_v1/`. Not wired into any pipeline stage or API yet.

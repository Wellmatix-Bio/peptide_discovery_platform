---
tags:
  - peptide
  - cytokine
  - immunomodulation
---

# il10-induction-predictor-v1

**First version.** P(peptide induces IL-10) from sequence, for the immunomodulation studio.

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|
| 0.332 | 0.760 | 0.445 | 0.682 | 0.502 | 0.862 |

Trained on 4981 peptides (16% positive), lengths 8-46.

## Limits

- Sensitivity is about 0.50 against specificity 0.86; the training set is 16% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-46 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.

## Usage

```python
from model_store.il10_induction_predictor_v1 import Il10InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il10InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, in_length_range, applicability
```

## Files

Code and weights are kept privately; weights live in `model_store/model_weights/il10_induction_predictor_v1/`. Not wired into any pipeline stage or API yet.

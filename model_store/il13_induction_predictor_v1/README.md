---
tags:
  - peptide
  - cytokine
  - immunomodulation
---

# il13-induction-predictor-v1

**First version.** P(peptide induces IL-13) from sequence, for the immunomodulation studio.

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|
| 0.282 | 0.751 | 0.370 | 0.611 | 0.259 | 0.964 |

Trained on 3221 peptides (10% positive), lengths 8-35.

## Limits

- Very low sensitivity (0.26) at the 0.5 threshold: most true inducers are missed. The training set is 10% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-35 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.

## Usage

```python
from model_store.il13_induction_predictor_v1 import Il13InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il13InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, in_length_range, applicability
```

## Files

Code and weights are kept privately; weights live in `model_store/model_weights/il13_induction_predictor_v1/`. Not wired into any pipeline stage or API yet.

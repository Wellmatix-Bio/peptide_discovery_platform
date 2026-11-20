---
tags:
  - peptide
  - cytokine
  - immunomodulation
---

# il6-induction-predictor-v1

**First version.** P(peptide induces IL-6) from sequence, for the immunomodulation studio.

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|
| 0.409 | 0.752 | 0.471 | 0.636 | 0.288 | 0.984 |

Trained on 3356 peptides (11% positive), lengths 8-25.

## Limits

- Trained only on 8-25 residue peptides.
- Sensitivity is low at the 0.5 threshold; the training set is 11% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-25 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.

## Usage

```python
from model_store.il6_induction_predictor_v1 import Il6InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il6InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, in_length_range, applicability
```

## Files

Code and weights are kept privately; weights live in `model_store/model_weights/il6_induction_predictor_v1/`. Not wired into any pipeline stage or API yet.

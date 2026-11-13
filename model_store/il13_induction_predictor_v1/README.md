---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# il13-induction-predictor-v1

**First version.** P(peptide induces IL-13) from sequence, for the immunomodulation studio. Mean of 2 classifiers on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| MLP_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.261 | 0.744 | 0.355 | 0.644 | 0.383 | 0.905 |
| XGBoost_lenbin | length-bin x class sample weights | 0.306 | 0.716 | 0.333 | 0.575 | 0.156 | 0.993 |
| **ensemble** | ensemble_summary.csv of the training notebook | 0.282 | 0.751 | 0.370 | 0.611 | 0.259 | 0.964 |

Trained on 3221 peptides (10% positive), lengths 8-35.

## Limits

- Matches the notebook's own IL-13 ensemble (MLP_posw + XGBoost_lenbin).
- Very low sensitivity (0.26) at the 0.5 threshold: most true inducers are missed. The training set is 10% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-35 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.il13_induction_predictor_v1 import Il13InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il13InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/il13_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

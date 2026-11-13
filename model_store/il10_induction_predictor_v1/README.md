---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# il10-induction-predictor-v1

**First version.** P(peptide induces IL-10) from sequence, for the immunomodulation studio. A single classifier on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| MLP_posw | positive-class weight (negatives 1, positives n_neg/n_pos) | 0.332 | 0.760 | 0.445 | 0.682 | 0.502 | 0.862 |
| **model** |  | 0.332 | 0.760 | 0.445 | 0.682 | 0.502 | 0.862 |

Trained on 4981 peptides (16% positive), lengths 8-46.

## Limits

- A single MLP_posw member, the strongest single model in the notebook's IL-10 ensemble table (MCC 0.332, AUROC 0.760); the notebook's three-model ensemble scored lower (MCC 0.317, AUROC 0.754).
- Sensitivity is about 0.50 against specificity 0.86; the training set is 16% positive.
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-46 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.il10_induction_predictor_v1 import Il10InductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = Il10InductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/il10_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

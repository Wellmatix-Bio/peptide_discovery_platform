---
tags:
  - peptide
  - cytokine
  - immunomodulation
  - esm2
library_name: scikit-learn
---

# ifn-gamma-induction-predictor-v1

**First version.** P(peptide induces IFN-gamma) from sequence, for the immunomodulation studio. A single classifier on mean-pooled ESM2-150M features (640-d).

## Accuracy (5-fold cross-validation, cluster-disjoint, threshold 0.5)

| Member | Imbalance handling | MCC | AUROC | AUPRC | BalAcc | Sens | Spec |
|---|---|---|---|---|---|---|---|
| MLP_lenbin | length-bin x class sample weights | 0.264 | 0.698 | 0.513 | 0.641 | 0.607 | 0.676 |
| **model** |  | 0.264 | 0.698 | 0.513 | 0.641 | 0.607 | 0.676 |

Trained on 84926 peptides (30% positive), lengths 8-20.

## Limits

- Trained only on 8-20 residue peptides; the platform accepts 6-50, so most of that range is outside the trained domain.
- Weakest model of the six (cross-validated MCC 0.26, AUROC 0.70).
- Applicability: predict_batch returns an `applicability` assessment per peptide (in_domain, limited_evidence or out_of_domain, with reasons) from the trained length range 8-20 and the positive/negative training counts at that exact length (info.json `domain`); sequence similarity and embedding-distribution checks are not run yet. `in_length_range` is kept.
- Probabilities are not comparable across cytokines or calibrated; label is simply probability >= 0.5.
- Embeddings in the training project were computed in fp16 on GPU and here in fp32; member probabilities differ by up to about 0.02 on a 40-sequence check per model.
- Weights were pickled with an earlier scikit-learn than the one installed here; loading emits a version warning.

## Usage

```python
from model_store.ifn_gamma_induction_predictor_v1 import IfnGammaInductionPredictor
from pipeline.feature_extractor import FeatureExtractor

predictor = IfnGammaInductionPredictor()
predictor.predict_proba_batch(sequences, FeatureExtractor())   # list[float]
predictor.predict_batch(sequences, FeatureExtractor())         # probability, label, per-member probabilities, in_length_range
```

## Files

Code here; weights in `model_store/model_weights/ifn_gamma_induction_predictor_v1/` (one `<member>.joblib` per member and `info.json`). Shared inference code is `src/common/cytokine_ensemble.py`. Not wired into any pipeline stage or API yet.

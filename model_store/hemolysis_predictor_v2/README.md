---
tags:
  - peptide
  - hemolysis
  - ensemble
  - esm2
library_name: pytorch
---

# hemolysis-predictor-v2

Hemolysis from sequence, with two outputs from one ensemble:

- **`phc50`** (regression): pHC50 = 6 - log10(HC50 in uM). Higher = more hemolytic. Used for **ranking**.
- **`p_hemolytic`** (classification): P(HC50 < 100 uM). Used for **filtering**, evaluated at threshold 0.5.

Ensemble of a multi-task network (per-residue ESM2-150M embeddings), an SVR and an SVC (mean-pooled embeddings). pHC50 averages the SVR and network; P(hemolytic) averages the SVC and network.

Used only by the metabolic pipeline (`s08_metabolic_safety_developability`, `s11_metabolic_ranking`). The wound-healing stages still use `hemolysis_predictor_v1`.

## Accuracy (out-of-fold, grouped CV at 80% identity)

| | |
|---|---|
| pHC50 ensemble | Pearson 0.567, R2 0.307, MAE 0.477, RMSE 0.632 |
| Hemolytic ensemble | AUROC 0.782, accuracy 0.717 (majority class 0.531), precision 0.727, recall 0.746 |

This is lower than the roughly 0.7 Pearson of earlier hemolysis work; the training project's `report.md` attributes those scores to a leaky random split, and this evaluation is audited. See `model_card.json` for the limits, in particular that incretin-like peptides are out of domain (about 29% identity to anything in training).

## Usage

```python
from model_store.hemolysis_predictor_v2 import HemolysisPredictor

predictor = HemolysisPredictor()
predictor.predict_batch(sequences, feature_extractor)               # list of dicts: phc50, p_hemolytic, hc50_um, member outputs
predictor.predict_phc50_batch(sequences, feature_extractor)         # list[float], ranking
predictor.predict_hemolytic_proba_batch(sequences, feature_extractor)  # list[float], filtering
```

`feature_extractor` is the shared `FeatureExtractor`; its ESM2 cache is reused.

## Files

Code here; weights in `model_store/model_weights/hemolysis_predictor_v2/`: `network.pt`, `svm_svr.joblib`, `info.json`. `architecture.py` is the network, copied verbatim from the training project.

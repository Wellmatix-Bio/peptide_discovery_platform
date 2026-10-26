# collagen_ecm_predictor_v1

Scores how likely a peptide is to promote collagen / ECM production, from sequence alone.

- **Model:** positive-unlabeled bagging. A StandardScaler + PCA (62 components) feeds 500 logistic regressions, each trained on all positives against all known negatives plus a random draw of unlabeled peptides. The output is the mean bag probability.
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over residues with BOS/EOS excluded.
- **Weights:** `model_store/model_weights/collagen_ecm_predictor_v1/` (`reducer.joblib`, `bags.npz`, `metadata.json`).
- **Input:** 20 uppercase standard amino acids, length 2-50. Anything else is rejected.

```python
from model_store.collagen_ecm_predictor_v1 import CollagenECMPredictor

pred = CollagenECMPredictor()
pred.predict_proba_batch(sequences, feature_extractor)               # list[float], mean P(active)
pred.predict_proba_with_std_batch(sequences, feature_extractor)      # (means, across-bag stds)
```

The score is a ranking signal, not a calibrated probability. Use it to order candidates.

Training code and metrics: `Models/Collagen - ECM activity/pipeline.ipynb`.

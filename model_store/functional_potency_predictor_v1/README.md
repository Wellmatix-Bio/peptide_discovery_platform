# functional_potency_predictor_v1

Predicts functional agonism potency (pEC50 = -log10(EC50 [M]), higher = more potent) at GCGR, GLP1R and GIPR.

- **Model:** per receptor, the unweighted mean of Ridge, Random Forest, XGBoost and an MLP, each on the raw mean-pooled ESM2-150M embedding (no scaling).
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over residues with BOS/EOS excluded. Input is `plain_sequence(seq)`, so Aib, Sar, Nle and lipidated lysines map to their closest canonical residue.
- **Weights:** `model_store/model_weights/functional_potency_predictor_v1/<member>/<RECEPTOR>.joblib`.

```python
from model_store.functional_potency_predictor_v1 import FunctionalPotencyPredictor

pred = FunctionalPotencyPredictor()                      # or members=("ridge", "xgb", "mlp")
pred.predict_pec50_batch(sequences, feature_extractor, "GLP1R")      # list[float]
pred.predict_pec50_all_batch(sequences, feature_extractor)           # {receptor: list[float]}
```

Training code and metrics: `Models/Functional Agonism/training_regression.ipynb`.

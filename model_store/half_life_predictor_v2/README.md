# half_life_predictor_v2

Mammal-only version of [half_life_predictor_v1](../half_life_predictor_v1/README.md). Predicts peptide half-life as log10(seconds) from sequence, ESM2-150M embedding, physicochemical descriptors, modification flags and measurement context.

- **What differs from v1:** same features, same XGBoost setup, but trained only on half-lives measured in mammals (mouse, rat, human, monkey, pig, dog, rabbit, cow, sheep, cat): 1,179 rows, 795 unique sequences. Chicken, insect and unknown-species rows are dropped, and CD-HIT clusters and folds are recomputed on the mammal sequences.
- **Model:** mean of five cluster-fold XGBoost regressors, `fold_0..4.joblib` in `model_store/model_weights/half_life_predictor_v2/`.
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over all tokens including BOS/EOS.
- **Context:** the prediction is conditional on `species`, `matrix`, `vivo_vitro` and five modification flags, passed as a `HalfLifeContext`. The default is an unmodified peptide in human plasma, in vitro. `species` must be a mammal from the training set, and `vivo_vitro` is one of `ex_vivo`, `in_vitro`, `in_vivo` (no `unknown`).
- **Input:** 20 uppercase standard amino acids, length 6-50 (the trained range). Anything else is rejected.

```python
from model_store.half_life_predictor_v2 import HalfLifeContext, HalfLifePredictor

pred = HalfLifePredictor()
pred.predict_log10_half_life_batch(sequences, feature_extractor)                       # list[float]
pred.predict_half_life_seconds_batch(sequences, feature_extractor, HalfLifeContext(species="rat", matrix="serum"))
```

`context` is either one `HalfLifeContext` shared by the batch or one per sequence.

Training code and metrics: `Models/Half-life prediction/pipeline_esm2_150m_mammal.ipynb` (XGBoost variant, cached to `models/A_mammal/`).

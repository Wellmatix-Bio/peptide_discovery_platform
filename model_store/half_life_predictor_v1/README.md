# half_life_predictor_v1

Predicts peptide half-life as log10(seconds) from sequence, ESM2-150M embedding, physicochemical descriptors, modification flags and measurement context.

- **Model:** mean of five cluster-fold XGBoost regressors, `fold_0..4.joblib` in `model_store/model_weights/half_life_predictor_v1/`.
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over all tokens including BOS/EOS.
- **Context:** the prediction is conditional on `species`, `matrix`, `vivo_vitro` and five modification flags, passed as a `HalfLifeContext`. The default is an unmodified peptide in human plasma, in vitro.
- **Input:** 20 uppercase standard amino acids, length 5-50. Anything else is rejected.

```python
from model_store.half_life_predictor_v1 import HalfLifeContext, HalfLifePredictor

pred = HalfLifePredictor()
pred.predict_log10_half_life_batch(sequences, feature_extractor)                       # list[float]
pred.predict_half_life_seconds_batch(sequences, feature_extractor, HalfLifeContext(species="rat", matrix="serum"))
```

`context` is either one `HalfLifeContext` shared by the batch or one per sequence.

Training code and metrics: `Models/Half-life prediction/pipeline_esm2_150m.ipynb`.

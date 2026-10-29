# pigmentation_predictor_v1

Scores whether a peptide is a TIP (most likely tyrosinase inhibitory peptide; the training data does not define the term) from sequence alone.

- **Model:** equal-weight soft vote of Random Forest, XGBoost, calibrated RBF SVC and shrinkage LDA on frozen ESM2 embeddings. The output is P(TIP); the trained decision threshold is 0.5.
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over residues with BOS/EOS excluded.
- **Weights:** `model_store/model_weights/pigmentation_predictor_v1/` (`ensemble.joblib`, `run_metadata.json`). Trained on the 80% split only; the test set was never refit.
- **Input:** 20 uppercase standard amino acids, length 2-22 (the trained range). Anything else is rejected.
- **Environment:** the joblib was written with scikit-learn 1.9.0 and xgboost 3.2.0; load it with matching versions.

```python
from model_store.pigmentation_predictor_v1 import PigmentationPredictor

pred = PigmentationPredictor()
pred.predict_proba_batch(sequences, feature_extractor)   # list[float], P(TIP)
```

Held-out ensemble results (41 TIP / 101 non-TIP): Sn 0.78, Sp 0.92, MCC 0.71, AUC 0.94. The split is random, so these do not show generalization to unrelated sequence families, and about a quarter of TIPs are missed.

Training code and metrics: `Models/Pigmentation model/training.ipynb`.

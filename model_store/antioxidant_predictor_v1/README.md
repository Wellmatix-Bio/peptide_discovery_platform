# antioxidant_predictor_v1

Scores whether a peptide is antioxidant, from sequence alone.

- **Model:** average-logit ensemble of three base networks (CNN, two-layer BiLSTM, single-block Transformer) on one-hot residues. The output is the sigmoid of the mean base-model logit. The notebook's learned stacker is not used.
- **Weights:** `model_store/model_weights/antioxidant_predictor_v1/` (`cnn.pt`, `bilstm.pt`, `transformer.pt`).
- **Input:** 20 uppercase standard amino acids, length 1-68. Anything else is rejected. No ESM embeddings are needed.

```python
from model_store.antioxidant_predictor_v1 import AntioxidantPredictor

pred = AntioxidantPredictor()
pred.predict_proba_batch(sequences)   # list[float]
```

Held-out test sheet (444 neg / 447 pos), threshold 0.5: accuracy 0.899, MCC 0.801, AUROC 0.950, AUPRC 0.964. Test peptides are at most 29 residues, so behaviour on longer peptides rests on the training distribution only.

Training code and metrics: `Models/Antioxidant/models/B/training.ipynb`.

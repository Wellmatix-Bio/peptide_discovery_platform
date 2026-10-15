# binding_affinity_predictor_v1

Predicts how tightly a peptide binds a receptor, as pKd = 9 - log10(Kd [nM]); higher = tighter.

- **Model:** a frozen ESM2-150M encodes the target and the peptide into one pooled vector each; a small MLP on the concatenated 1280-d vector regresses pKd.
- **Receptors:** the human sequences of GLP1R, GCGR and GIPR ship with the weights (`receptors.json`, UniProt full length), so you pass only the peptide.
- **Checkpoints:** `cluster` (default) was selected on a cluster-held-out validation set; `random` on a random one. Both are in `model_store/model_weights/binding_affinity_predictor_v1/`.
- **Input:** peptides of the 20 uppercase standard amino acids.

```python
from model_store.binding_affinity_predictor_v1 import BindingAffinityPredictor

pred = BindingAffinityPredictor()                                    # or split="random"
pred.predict_pkd_batch(peptides, feature_extractor, "GLP1R")         # list[float]
pred.predict_pkd_all_batch(peptides, feature_extractor)              # {receptor: list[float]}
pred.predict_pkd_pairs(targets, binders, feature_extractor)          # any target sequence
```

Training code and metrics: `Models/Binding Affinity/training_balm.ipynb`, "Model A".

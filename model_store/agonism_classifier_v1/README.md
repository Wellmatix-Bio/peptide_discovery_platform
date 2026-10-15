# agonism_classifier_v1

P(EC50 < 1000 pM) that a peptide is a potent agonist at GLP1R, GIPR and GCGR. Note the label threshold is 1000 **pM** (1 nM), not 1000 nM.

- **Model:** mean of five fold MLPs on per-token physicochemical features; one head per receptor.
- **Features:** `FeatureExtractor.get_token_features` (shared extractor); the residue table is `nonstandard_residues.csv` next to this file.
- **Weights:** `model_store/model_weights/agonism_classifier_v1/fold{1..5}.pt`.

```python
from model_store.agonism_classifier_v1 import AgonismClassifier

clf = AgonismClassifier()
clf.predict_proba_batch(sequences, feature_extractor)   # {receptor: [p or None, ...]}
```

`None` is returned for a sequence with an unknown residue or more than 44 tokens.

Training code and metrics: `Models/Functional Agonism/training.ipynb`.

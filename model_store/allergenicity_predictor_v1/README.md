# allergenicity_predictor_v1

Scores how likely a short peptide is to bind IgE (to be an allergen epitope), from sequence alone.

- **Output:** `score` (0 is the decision boundary) and `p_allergen = sigmoid(score)`. It is not a calibrated probability: 0.5 is the model's own class boundary.
- **Weights:** `model_store/model_weights/allergenicity_predictor_v1/` (ten fold files and `ensemble.json`).
- **Input:** 20 uppercase standard amino acids, length 3-49, the trained range. Anything else is rejected.

```python
from model_store.allergenicity_predictor_v1 import AllergenicityPredictor

pred = AllergenicityPredictor()
pred.predict_proba_batch(sequences, feature_extractor)   # list[float], P(allergen)
pred.predict_batch(sequences, feature_extractor)         # list of {"score", "p_allergen"}
```

## Training data

IEDB IgE-binding peptides: 8,939 unique peptides of 3-49 residues (median 12), 2,725 IgE-binding and 6,214 not. Peptides are split by source allergen protein, so peptides of one protein never straddle train and test, and held-out peptides that share an 8-mer with the training side are excluded from evaluation. The label records that IgE binding was seen in an assay, not a clinical allergy.

## Hold-out performance (1,581 evaluable peptides, 482 IgE-binding)

| Sens | Spec | Prec | Acc | AUC | MCC |
|---|---|---|---|---|---|
| 0.452 | 0.881 | 0.625 | 0.750 | 0.729 | 0.370 |

The deployed weights were re-fitted with the notebook's code, so they differ slightly from the notebook's cached run (AUC 0.740, MCC 0.410). Sensitivity is low at the default boundary: fewer than half of the IgE-binding peptides are flagged.

Training code and metrics: `Models/Allergenicity/training.ipynb`, export: `Models/Allergenicity/export_deployment.py`.

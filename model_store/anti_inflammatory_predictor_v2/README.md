---
tags:
  - peptide
  - anti-inflammatory
  - immunomodulation
  - esm2
  - pytorch
library_name: pytorch
---

# anti-inflammatory-predictor-v2

Anti-inflammatory peptide probability classifier on frozen ESM2-150M embeddings, a window-based local branch and 80 selected sequence descriptors. Same task as v1, but it needs the shared `FeatureExtractor` for the embeddings.

## What changed from v1

- Input is ESM2-150M embeddings plus handcrafted descriptors, not a one-hot encoding.
- **No truncation.** v1 cuts input to 25 residues. v2 scores the whole sequence and warns when the length is outside 8 to 30; `predict_batch` returns `in_length_range` per sequence.
- Deterministic: dropout is off and nothing is sampled.

## Accuracy

At threshold 0.5, scored by this loader:

| Set | n | ACC | Sn | Sp | AUC | MCC |
|---|---|---|---|---|---|---|
| Independent set 1 | 1049 | 0.747 | 0.662 | 0.805 | 0.809 | 0.470 |
| Independent set 2 | 425 | 0.849 | 0.879 | 0.829 | 0.899 | 0.698 |

**Read the next table as the cautious one.** Most independent peptides have a close relative in the training set. Restricted to those with no training peptide at the stated identity or higher:

| Set | Identity below | n | AUC | MCC |
|---|---|---|---|---|
| 1 | 0.8 | 732 | 0.700 | 0.275 |
| 1 | 0.6 | 343 | 0.551 | 0.051 |
| 2 | 0.8 | 128 | 0.786 | 0.421 |
| 2 | 0.6 | 58 | 0.641 | 0.181 |

For peptides unlike the training data, expect AUC of about 0.55 to 0.8, not 0.8 to 0.9.

## Limits

- Trained on 11 to 30 residues (augmented crops from 8). Outside 8 to 30 the score is extrapolation and is returned with a warning. The platform accepts up to 50.
- Accuracy on its own (augmented) training set is 0.999 against the figures above: the model overfits, and the checkpoint was selected on the training set.
- This extended-data variant was chosen over the base variant because it scored higher on the same independent sets, which adds a small optimistic bias.
- One training seed.

## Usage

```python
from pipeline.feature_extractor import FeatureExtractor
from model_store.anti_inflammatory_predictor_v2 import AntiInflammatoryPredictor

fe = FeatureExtractor()
predictor = AntiInflammatoryPredictor()
predictor.predict_proba("GLFDIIKKIAESF", fe)                         # float
predictor.predict_proba_batch(["GLFDIIKKIAESF", "ALPMHIRKLAA"], fe)  # list[float]
predictor.predict_batch(["GLFDIIKKIAESF"], fe)                        # [{"probability", "in_length_range"}]
```

Lazy-loaded: weights load on the first call.

## Files

Code here; weights in `model_store/model_weights/anti_inflammatory_predictor_v2/weights.pt` (`state_dict`, `feature_index`, `feature_mean`, `feature_scale`; loads with `weights_only=True`). Used by Stage 6 and the metabolic Stage 6.

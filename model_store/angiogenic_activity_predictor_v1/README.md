---
tags:
  - peptide
  - angiogenesis
  - svm
  - mlp
  - esm2
library_name: scikit-learn
---

# angiogenic-activity-predictor-v1

Angiogenic-dominant probability classifier used by Stage 6 (Functional AI Models) to score the angiogenesis function axis of the five target peptide functions.

## Architecture

Two independently trained models, each a `Pipeline` wrapping a `ColumnTransformer` that passes the 16 hand-built physicochemical descriptors through untouched and PCA-reduces (95% explained variance) a 640-dim ESM2 (`facebook/esm2_t30_150M_UR50D`, mean-pooled, BOS/EOS excluded) embedding block:

- `SVC(probability=True, C=1.0, kernel="rbf", gamma="scale", class_weight={0:1, 1:5}, random_state=0)`
- `TorchMLPClassifier` — sklearn-compatible wrapper around a 3-layer `MLPNet` (32 -> 16 -> 2, softmax output), trained 300 epochs with `NLLLoss` and class weights `[1, 5]`

Both use `class_weight`/`pos_class_weight = 5` to counter heavy class imbalance (23 positive / 183 negative). Deployed prediction is the **unweighted mean** of the two models' `P(angiogenic_dominant)`.

A third architecture, `RandomForestClassifier`, was trained alongside these in the source notebook but is deliberately excluded from the deployed ensemble.

## Files

- `production_svm.joblib`, `production_mlp.joblib` — the two fitted pipelines
- `model_card.json` — feature list, hyperparameters, notebook LOO-CV metrics

## Usage

```python
from angiogenic_activity_predictor_v1 import AngiogenicActivityPredictor

predictor = AngiogenicActivityPredictor()
result = predictor.predict("GLFDIIKKIAESF")
# {"angiogenic": 0.36, "svm": 0.38, "mlp": 0.34}
```

Lazy-loaded: ESM2 model and the two pipelines load on first `predict()` call, not at import time.

## Provenance

Exported from `Models/Angiogenic activity prediction/pipeline_35M_embedder.ipynb` (despite the filename, this run embeds with `facebook/esm2_t30_150M_UR50D`, not a 35M checkpoint), trained on `datasets/ijms-2544295-supplementary.xlsx` (206 clean sequences after filtering to standard amino acids and deduplication).

**Full-data refit, not the notebook's evaluated objects.** As with `proliferation_migration_predictor_v1`, the notebook's LOOCV loop fits fresh model instances per fold purely to produce honest out-of-sample metrics; the `.joblib` artifacts here come from the notebook's separate `fit_production_model()` cell, which fits each architecture **once on the full 206-row dataset** — this is the notebook's own designated deployable object, not a deviation invented for export.

Because the deployed objects are full-data refits, they have **no held-out evaluation of their own**. The best available generalization estimate is the notebook's own Leave-One-Out CV (206 folds):

| model | AUC | ACC | MCC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| svm | 0.7330 | 0.8932 | 0.2738 | 0.2667 | 0.5714 | 0.1739 |
| rf (not deployed) | 0.7173 | 0.8884 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp | 0.5890 | 0.8592 | 0.1851 | 0.2564 | 0.3125 | 0.2174 |

These numbers describe the notebook's per-fold LOO models, not a direct evaluation of the shipped full-data-refit objects, and reflect a severely imbalanced, small (n=206) dataset. Treat `angiogenic` as a soft screening signal alongside the platform's other functional scores, not a high-confidence standalone classifier.

## Note on the MLP artifact

`production_mlp.joblib` pickles a `TorchMLPClassifier`/`MLPNet` instance under the module path recorded at pickle time (`__main__`, since the notebook defines these classes at its own top level, not by importing them from `predictor.py`). `predictor.py`'s `_load()` aliases `TorchMLPClassifier`/`MLPNet` into `sys.modules["__main__"]` immediately before unpickling so this resolves correctly regardless of how the predictor itself is invoked — same underlying constraint as `anti_inflammatory_predictor_v1`'s `DACAIPs` checkpoint, just via `joblib`/pickle instead of a raw `torch.load` state dict.

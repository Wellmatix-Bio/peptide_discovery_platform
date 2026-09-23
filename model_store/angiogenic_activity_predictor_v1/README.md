---
tags:
  - peptide
  - angiogenesis
  - svm
  - random-forest
  - mlp
  - esm2
library_name: scikit-learn
---

# angiogenic-activity-predictor-v1

Angiogenic-dominant probability classifier used by Stage 6 (Functional AI Models) to score the angiogenesis function axis of the five target peptide functions.

## Architecture

Three independently trained models, each a `Pipeline` wrapping a `ColumnTransformer` that passes the 16 hand-built physicochemical descriptors through untouched and PCA-reduces (95% explained variance) a 1280-dim ESM2 (`facebook/esm2_t33_650M_UR50D`, mean-pooled, BOS/EOS excluded) embedding block:

- `SVC(probability=True, C=1.0, kernel="rbf", gamma="scale", class_weight={0:1, 1:5}, random_state=0)`
- `RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, class_weight={0:1, 1:5}, random_state=0)`
- `TorchMLPClassifier` — sklearn-compatible wrapper around a 3-layer `MLPNet` (32 -> 16 -> 2, softmax output), trained 300 epochs with `NLLLoss` and class weights `[1, 5]`

All three use `class_weight`/`pos_class_weight = 5` to counter heavy class imbalance (23 positive / 183 negative). Deployed prediction is the **unweighted mean** of the three models' `P(angiogenic_dominant)`.

## Files

- `production_svm.joblib`, `production_rf.joblib`, `production_mlp.joblib` — the three fitted pipelines
- `model_card.json` — feature list, hyperparameters, notebook LOO-CV metrics

## Usage

```python
from angiogenic_activity_predictor_v1 import AngiogenicActivityPredictor

predictor = AngiogenicActivityPredictor()
result = predictor.predict("GLFDIIKKIAESF")
# {"angiogenic_dominant": 0.41, "svm": 0.38, "rf": 0.52, "mlp": 0.34}
```

Lazy-loaded: ESM2 model and the three pipelines load on first `predict()` call, not at import time.

## Provenance

Exported from `Models/Angiogenic activity prediction/pipeline.ipynb`, trained on `datasets/ijms-2544295-supplementary.xlsx` (206 clean sequences after filtering to standard amino acids and deduplication).

**Full-data refit, not the notebook's evaluated objects.** As with `proliferation_migration_predictor_v1`, the notebook's LOOCV loop fits fresh model instances per fold purely to produce honest out-of-sample metrics; the three `.joblib` artifacts here come from the notebook's separate `fit_production_model()` cell, which fits each architecture **once on the full 206-row dataset** — this is the notebook's own designated deployable object, not a deviation invented for export.

Because the deployed objects are full-data refits, they have **no held-out evaluation of their own**. The best available generalization estimate is the notebook's own Leave-One-Out CV (206 folds):

| model | AUC | ACC | MCC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| svm | 0.7733 | 0.8641 | 0.1616 | 0.2222 | 0.3077 | 0.1739 |
| rf | 0.7299 | 0.8835 | 0.0856 | 0.0769 | 0.3333 | 0.0435 |
| mlp | 0.5904 | 0.8447 | 0.0348 | 0.1111 | 0.1538 | 0.0870 |

These numbers describe the notebook's per-fold LOO models, not a direct evaluation of the shipped full-data-refit objects, and reflect a severely imbalanced, small (n=206) dataset — SVM has the strongest LOOCV AUC/MCC of the three. Treat `angiogenic_dominant` as a soft screening signal alongside the platform's other functional scores, not a high-confidence standalone classifier.

## Note on the MLP artifact

`production_mlp.joblib` pickles a `TorchMLPClassifier`/`MLPNet` instance defined in `predictor.py`. Those class definitions must stay importable at this module path (`angiogenic_activity_predictor_v1.predictor`) for the pickle to unpickle — same constraint as `anti_inflammatory_predictor_v1`'s `DACAIPs` checkpoint, just via `joblib`/pickle instead of a raw `torch.load` state dict.

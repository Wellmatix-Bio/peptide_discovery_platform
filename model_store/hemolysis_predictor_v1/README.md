---
tags:
  - peptide
  - hemolysis
  - random-forest
  - hemopi2-reproduction
license: mit
---

# hemolysis-predictor-v3

Default hemolysis predictor as of this version. Replaces `hemolysis_predictor_v1`
(a BiLSTM+attention model over ESM2 embeddings, unrelated to HemoPI2, with
an unbounded/uncalibrated pHC50 range) in both Stage 8 and Route A's GA
fitness function.

## What this is

An in-process, native Python/sklearn reproduction of HemoPI2's regression
model (Random Forest over hand-computed sequence composition descriptors),
ported verbatim from `Models/Hemolysis prediction/paper_replicated.ipynb`.
Same underlying model family and feature set as `hemolysis_predictor_v2`
(which subprocess-wraps the actual HemoPI2 CLI, GPL-3.0), but trained
independently on HemoPI2's own published cross-validation/test splits —
no subprocess, no vendored GPL code, no Windows encoding/`python3`-shim
workarounds.

**Validated against the paper's own reported benchmark** on the independent
test set: Pearson r = 0.740, R² = 0.544 (paper's targets: r = 0.739,
R² = 0.543).

## Architecture

`Pipeline([('imputer', SimpleImputer(strategy='median')), ('rf', RandomForestRegressor(n_estimators=800, max_depth=30, ...))])`,
trained on 1167 hand-computed descriptors per sequence: molecular weight,
length, AAC, DPC, ATC, BTC, PCP, RRI, PRI, DDR, SER, SEP, CTC, CeTD, PAAC,
APAAC, QSO, SOC — the same descriptor groups HemoPI2's own feature-extraction
script computes.

`predictor.py`'s `Extractor` class reimplements each descriptor group's exact
arithmetic (including a couple of faithfully-preserved quirks/bugs in the
original author script — see the inline comments on `_feature_atc` and
`_feature_rri` — since the released model was trained on that exact output).

## pHC50 conversion

The model predicts `y = -log10(HC50 in uM)` directly (not this codebase's
`pHC50 = -log10(HC50 in M)` convention). `predictor.py` converts:

    pHC50 = y_pred + 6.0

Same offset and direction as `hemolysis_predictor_v2`'s `HC50(uM) -> pHC50`
conversion. HIGH pHC50 = hemolytic at LOW concentration = worse.

## Files

- `model.joblib` (~26MB) — the trained `Pipeline`, copied from
  `Models/Hemolysis prediction/models/regressor/v2/model.joblib` (already
  trained and cached by the notebook; not retrained here).
- `property_tables.py` — the seven embedded CSV/TSV property tables the
  descriptor extractor needs (amino-acid attribute groups, atom/bond
  composition, physicochemical index, Schneider-Wrede and Grantham distance
  matrices), verbatim from the notebook.
- `predictor.py` — `Extractor` (descriptor computation),
  `normalize_sequences`, and `ReplicatedHemoPI2Predictor`
  (`predict_phc50(sequence)` / `predict_phc50_batch(sequences)`, matching
  every other hemolysis predictor's interface).

## Usage

```python
from model_store.hemolysis_predictor_v3 import ReplicatedHemoPI2Predictor

predictor = ReplicatedHemoPI2Predictor()
phc50_values = predictor.predict_phc50_batch(sequences)  # one extraction pass + one model.predict() call
```

Selected in Stage 8 via `hemolysis_predictor_version: "v3"` (the default —
see `s08_safety_developability/stage.py`'s `DEFAULT_HEMOLYSIS_PREDICTOR_VERSION`)
or `"v2"` for the actual HemoPI2 CLI subprocess.

## Provenance

- Source notebook: `Models/Hemolysis prediction/paper_replicated.ipynb`
- Training data: HemoPI2's own `cross_val_dataset.csv` / `independent_dataset.csv`
  (see `model_store/hemolysis_predictor_v2/vendor/`)
- Trained/cached 2026-09-18, scikit-learn 1.9.0, no unpickling version
  warnings (unlike `hemolysis_predictor_v1`'s fold checkpoints, which
  predated the environment's current library versions)

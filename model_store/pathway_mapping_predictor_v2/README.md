---
tags:
  - peptide
  - pathway-mapping
  - mechanism-of-action
  - lda
  - random-forest
library_name: scikit-learn
---

# pathway-mapping-predictor-v2

Per-pathway mechanism-of-action classifier for candidate peptides, used by Stage 7 (Structure & Mechanism) and, through it, Stage 11's immunomodulation score.

## Architecture

9 independent pathways (MAPK, ERK, PI3K_AKT_MTOR, TGFB_SMAD, NF_KB, WNT_BCATENIN, FGFR_JAK2_STAT3, VEGF_ANGIOGENESIS, CYTOKINE_MACROPHAGE), each a 2-model ensemble (LDA + RF; the notebook's SVM is not used) trained in `train_per_pathway.ipynb`:

- `LDA`: `StandardScaler` + `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`
- `RF`: `RandomForestClassifier(n_estimators=300, max_depth=5, class_weight='balanced')`

Task per pathway: **activator (+1) vs inhibitor (-1)**; label-0 rows were dropped. Unlike v1 (pathway engagement, PU-bagging), the probability here is **P(activator)**, so a low value means "inhibitor", not "not involved". Final probability is the unweighted mean of the two models' `predict_proba` P(activator).

Features: the same 11 modlAMP physicochemical descriptors as v1 (`MW, Charge, ChargeDensity, pI, InstabilityInd, Aromaticity, AliphaticInd, BomanInd, HydrophRatio, Hydrophobicity, HydrophobicMoment`).

## Files

- `model/<LABEL>/{LDA,RF}.joblib`: 2 models per label, 9 labels, 18 files (the `SVM.joblib` files in the folder are unused)
- `model/signature.txt`: hash of the train split + feature list used for training

## Usage

```python
from pathway_mapping_predictor_v2 import PathwayMappingPredictor

predictor = PathwayMappingPredictor()
result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")
# {"MAPK": {"probability": 0.61, "label_predicted": True, "n_models": 2,
#           "model_probabilities": {"LDA": ..., "RF": ...}}, ...}
```

Lazy-loaded: all 18 models load on first `predict()` call.

## Provenance

Exported from `Models/Pathway Interface Mapping/model/` (`train_per_pathway.ipynb`). Test sets there are small (a handful of curated peptides per pathway); per-pathway metrics are noisy.

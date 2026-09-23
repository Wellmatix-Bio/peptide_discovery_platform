---
tags:
  - peptide
  - pathway-mapping
  - mechanism-of-action
  - random-forest
  - pu-learning
library_name: scikit-learn
---

# pathway-mapping-predictor-v1

Per-pathway mechanism-of-action classifier for candidate peptides, used by Stage 7 (Structure & Mechanism).

## Architecture

9 independent pathways (MAPK, ERK, PI3K_AKT_MTOR, TGFB_SMAD, NF_KB, WNT_BCATENIN, FGFR_JAK2_STAT3, VEGF_ANGIOGENESIS, CYTOKINE_MACROPHAGE), each a 100-model PU (positive-unlabeled) bagging ensemble of `RandomForestClassifier`s (`n_estimators=100, max_depth=5`). Each bag was trained on all known positives for that label plus an equal-sized random sample of the unlabeled set; final probability is the mean of `predict_proba` across all 100 bags. Only the RandomForest half of the source notebook's bagging (RF + XGBoost) is vendored here.

Features: 11 modlAMP physicochemical descriptors (`MW, Charge, ChargeDensity, pI, InstabilityInd, Aromaticity, AliphaticInd, BomanInd, HydrophRatio, Hydrophobicity, HydrophobicMoment`) — `Hydrophobicity`/`HydrophobicMoment` from the Eisenberg scale, the rest from modlAMP's `GlobalDescriptor`.

## Files

- `model/<LABEL>/RandomForestClassifier_k{0-99}.joblib` — 100 bag models per label, 9 labels, 900 files total

## Usage

```python
from pathway_mapping_predictor_v1 import PathwayMappingPredictor

predictor = PathwayMappingPredictor()
result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")
# {"MAPK": {"probability": 0.34, "label_predicted": False, "n_bags": 100}, ...}
```

Lazy-loaded: all 900 bag models load on first `predict()` call.

**Version note:** bags were pickled under scikit-learn 1.9.0; loading under an older sklearn (this platform runs 1.6.1) raises `InconsistentVersionWarning` — treat as a warning, not a blocker, but re-export from a matching sklearn if results look off.

## Provenance

Exported from `Models/Pathway Interface Mapping/model/` (`pipeline.ipynb`), RandomForest bags only.

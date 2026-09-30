---
tags:
  - peptide
  - structure-prediction
  - esmfold
  - pytorch
library_name: pytorch
---

# esmfold-v1

Single-sequence structure prediction for Stage 7 (Structure & Mechanism), and the CA coordinate/distance-matrix source Stage 8's cleavage-site stability screen needs.

## Architecture

`facebook/esmfold_v1` via `transformers.EsmForProteinFolding`: an ESM2-650M language-model backbone (`model.esm`, optionally fp16 on GPU) feeding a folding trunk (fp32, geometry-sensitive) that outputs per-residue atom37 coordinates and per-residue/per-atom pLDDT confidence. No MSA required (single-sequence).

## Files

- `weights/` — full `facebook/esmfold_v1` snapshot (config, tokenizer, `pytorch_model.bin`, ~7.9GB), vendored via `huggingface_hub.snapshot_download` rather than left to the `transformers` cache. Gitignored (`model_store/` is excluded repo-wide); re-fetch with `snapshot_download("facebook/esmfold_v1", local_dir="model_store/esmfold_v1/weights")` if missing.

## Usage

```python
from esmfold_v1 import ESMFoldPredictor

predictor = ESMFoldPredictor()
result = predictor.predict("EFELDRICGYGTARCRKKCRSQEYRIGRCPNTYACCLRKWDESLLNRTKP")
# {"sequence": ..., "pdb_str": "...", "mean_plddt": 0.69,
#  "per_residue_plddt": [...], "ca_coordinates": [[x,y,z], ...],
#  "ca_distance_matrix": [[...], ...]}
```

**pLDDT scale:** this checkpoint/transformers version returns pLDDT as a 0-1 fraction, not the usual 0-100 scale — confirmed by smoke test (`mean_plddt` ~0.69 on the reference peptide). Scale downstream thresholds accordingly.

`ca_distance_matrix` is ready to pass straight into `cleavage_site_predictor_v1.CleavageSitePredictor.predict_one()`'s `distance_matrix` argument.

Lazy-loaded: the tokenizer, ESM2 backbone, and folding trunk load on first `predict()` call. GPU is used automatically when available (`model.esm` runs fp16, the trunk stays fp32); falls back to CPU otherwise.

## Provenance

Wraps `facebook/esmfold_v1` from the Hugging Face Hub directly, weights vendored unmodified; no fine-tuning.

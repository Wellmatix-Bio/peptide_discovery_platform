---
tags:
  - peptide
  - cytotoxicity
  - esm2
  - pytorch
library_name: pytorch
---

# cytotoxicity-predictor-v1

Mammalian-cell cytotoxicity probability classifier used by Stage 8 (Safety & Developability).

## Architecture

4-fold ensemble. Each fold is the same local/global/cross-attention backbone as `hemolysis_predictor_v1` (BiLSTM + self-attention over per-residue ESM2 (`facebook/esm2_t33_650M_UR50D`) embeddings, cross-attended against a global sequence embedding), extended with an `nn.Embedding(7, 16)` lookup over assay `cell_type`, concatenated with the length feature before the final head. The head outputs a single logit; `predict_cytotoxicity()` applies `sigmoid` and averages across folds, so — unlike `hemolysis_predictor_v1`'s raw `pHC50` — the returned value is already a calibrated-by-training `[0, 1]` probability requiring no external calibrator.

`cell_type` is a required model input, not just metadata. When the target cell line for a candidate is unknown, `predict_cytotoxicity()` defaults to `"DRAMP_aggregate"` (an unspecified/aggregate cytotoxicity context) rather than a specific cell line or the DBAASP negative-control bucket. Pass `cell_type` explicitly to score against a specific line — valid values are the 7 keys in each checkpoint's `cell_type_to_id`: `DBAASP_no_cytotoxicity_hit`, `DRAMP_aggregate`, `Human PBMC`, `Human Primary Epidermal Keratinocytes (HEK)`, `Human keratinocytes HaCat`, `Human microvascular endothelial cells HMEC-1`, `Human skin fibroblasts`.

## Files

- `cytotoxicity_predictor_fold{0-3}.pt` — fold checkpoints, each containing `model_state_dict`, `config` (hidden_dim, d_model, n_heads, dropout), `test_metrics`, and `cell_type_to_id`

## Usage

```python
from cytotoxicity_predictor_v1 import CytotoxicityClassifier

model = CytotoxicityClassifier()
prob = model.predict_cytotoxicity("GIGKFLHSAKKFGKAFVGEIMNS")
prob_hacat = model.predict_cytotoxicity("GIGKFLHSAKKFGKAFVGEIMNS", cell_type="Human keratinocytes HaCat")
```

## Provenance

Exported from `Models/Cytotoxicity prediction/models/v1`. Model/training scaffolding ported from `Models/Hemolysis prediction/hemolysis_pred_pipeline_A.ipynb`, adapted for binary classification (`BCEWithLogitsLoss` in place of censored regression loss) with a `cell_type` embedding added — see that project's `impl_notes.md` for details.

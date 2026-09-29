---
tags:
  - peptide
  - antimicrobial-peptide
  - mic
  - esm2
  - random-forest
  - pytorch
library_name: pytorch
---

# mic-predictor-v1

Minimum inhibitory concentration (MIC) regressor used by Stage 6 (Functional AI Models) to score antimicrobial potency against 3 ATCC reference organisms.

## Architecture

3-model ensemble, simple unweighted mean of each model's `log10(MIC, uM)` prediction:

- **BiLSTM** — bidirectional LSTM over per-residue ESM2 (`facebook/esm2_t30_150M_UR50D`) embeddings, concatenated with a small MLP branch over an 84-dim genome nucleotide-composition vector (NAC+DNC+TNC) for the target organism, then a regressor head.
- **CNN** — two Conv1D+ReLU+MaxPool stages over the same per-residue ESM2 embeddings, concatenated with the same genome-composition branch, then a regressor head.
- **Random Forest** — 400-tree sklearn `RandomForestRegressor` over 319 iFeature protein descriptors (AAC, GAAC, CTDC, CTDT, CTDD, PAAC with lambda=1) concatenated with the same 84-dim genome-composition vector. No ESM2 involved.

All 3 models were trained with a weighted MSE loss (BiLSTM/CNN) or on point estimates (RF) against `log10(MIC, uM)`, on data restricted to 3 ATCC reference organisms.

**Prediction output is `log10(MIC, uM)`.** Convert to micromolar with `MIC_uM = 10 ** prediction`, or call `predict_mic_um()`.

## Files

- `bilstm_model.pt`, `cnn_model.pt` — raw `state_dict()` checkpoints, architecture hardcoded in `predictor.py`
- `rf_model.joblib` — the fitted Random Forest, `feature_names_in_` defines the exact 403-column input order
- `genome_vectors.json` — precomputed 84-dim NAC/DNC/TNC genome-composition vector for each of the 3 supported organisms (the raw genome FASTA files, several MB each, are NOT shipped — this package is self-contained without them)
- `model_card.json` — architectures, hyperparameters, feature schemes, per-model test metrics

## Usage

```python
from mic_predictor_v1 import MICPredictorEnsemble

model = MICPredictorEnsemble()
log_mic = model.predict_log_mic("GLFDIVKKVVGALGSL", "Escherichia coli")
mic_um = model.predict_mic_um("GLFDIVKKVVGALGSL", "Escherichia coli")
```

`organism` must be exactly one of: `"Escherichia coli"`, `"Staphylococcus aureus"`, `"Pseudomonas aeruginosa"`. Any other value raises `ValueError`. Sequences must be standard amino acids, length <=50 aa (the training-data length filter).

## Provenance

Exported from `Models/MIC prediction/mic_prediction_pipelineC_35M_embedding.ipynb` (despite the filename, this run embeds with `facebook/esm2_t30_150M_UR50D`, not the 35M checkpoint), using the 3 checkpoints in `saved_models/`. This supersedes the earlier pipeline B export (`saved_model_huber_loss/`, `facebook/esm2_t33_650M_UR50D`) now that pipeline C's output cells are clean and complete. `modelA/` and `modelB_no_medium/` are separate, unrelated experiments.

**Organism restriction.** This model is trained exclusively on 3 ATCC reference organisms (E. coli ATCC 25922, S. aureus ATCC 25923, P. aeruginosa ATCC 27853). It is NOT organism-general and will not produce meaningful predictions for any other species — there is no fallback or interpolation, `predict_log_mic` raises if `organism` isn't one of the 3.

**Per-model test performance (from the source notebook, held-out test split by CD-HIT cluster):**

| Model | Test R² |
|---|---|
| BiLSTM | 0.4889 |
| CNN | 0.4728 |
| Random Forest | 0.4218 |

**The ensemble's own performance is not independently verified.** The source notebook evaluates and reports each of the 3 models separately (see its final comparison table) but never computes a combined/ensembled metric. Averaging 3 models that draw on different feature sources (BiLSTM/CNN: residue-level ESM2 embeddings; RF: iFeature global descriptors) is expected to be at least as robust as any single model — this is standard practice, not unique to this data — but that expectation is inferred from the components, not measured directly. Treat the ensemble's R² as approximately in the 0.43-0.48 range pending direct validation.

**Censored-data caveat.** BiLSTM and CNN train against a weighted MSE loss where MIC rows reported as a bound (`<value` or `>value`, e.g. "MIC > 256 uM") are down-weighted to 0.5x versus 1.0x for exact-value rows. This down-weighting is NOT bound-aware: the loss does not enforce that a censored row's prediction stays on the correct side of its bound, it simply reduces that row's influence during training. The Random Forest has no censored-loss variant at all and trains on every row's point-estimate value directly, bound or not. Treat predictions for peptides expected to be weakly active (near or beyond the assay's tested concentration range) with additional caution.

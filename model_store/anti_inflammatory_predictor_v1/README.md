---
tags:
  - peptide
  - anti-inflammatory
  - immunomodulation
  - variational-autoencoder
  - contrastive-learning
  - pytorch
library_name: pytorch
---

# anti-inflammatory-predictor-v1

Anti-inflammatory peptide (AIP) probability classifier, relevant to Stage 6 (Functional AI Models) for the immunomodulation target function.

## Architecture

DAC-AIPs (Deep variational Autoencoder + Contrastive learning for AIPs identification) — a reimplementation of the architecture in Xu, Y., Zhang, S., Zhu, F. & Liang, Y., "A deep learning model for anti-inflammatory peptides identification based on deep variational autoencoder and contrastive learning," *Scientific Reports* 14, 18451 (2024).

Sequences are encoded with a fused one-hot (k=1) + multi-hot (k=2) + multi-hot (k=3) sliding-window scheme (60 channels x `seq_len`, no ESM2 embeddings) and truncated/padded to `seq_len=25`. A Conv1d/MaxPool1d encoder produces a 208-dim latent (`mu`, `logvar`); the latent is decoded (reconstruction), used in a triplet contrastive loss, and fed to a 2-layer classification head. Inference uses only the encoder + classification head: `softmax(logits)[:, 1]` is the anti-inflammatory-class probability.

## Files

- `dac_aips_final.pt` — final trained weights (`model_state`, `config`, `history`, `independent_metrics`). This is the notebook's own designated inference artifact (loaded explicitly in its "Reload the cached model for inference" cell), holding the best-validation-MCC epoch's weights. The sibling `dac_aips.pt` in the source project is a resumable training checkpoint (adds `optimizer_state`, `epoch`, `best_state`) and was not exported here.

## Usage

```python
from anti_inflammatory_predictor_v1 import AntiInflammatoryPredictor

predictor = AntiInflammatoryPredictor()
proba = predictor.predict_proba("GLFDIIKKIAESF")
```

Lazy-loaded: weights load on first `predict_proba()` call, not at import time.

## Provenance

Exported from `Models/Anti-inflammatory prediction`. Independent-test metrics recorded in the checkpoint: ACC 0.697, Sn 0.652, Sp 0.727, AUC 0.734, MCC 0.376. See that project's `report.md` for the training/debugging history (KL warm-up tuning, reconstruction-loss and class-imbalance fixes).

---
tags:
  - peptide
  - antimicrobial-peptide
  - protgpt2
  - lora
  - text-generation
library_name: peft
base_model: nferruz/ProtGPT2
---

# routeb_protgpt2_lora_v1

Conditional peptide generator used by Stage 4 Route B (de novo generation, per
CLAUDE.md's multi-route generation convention). A rank-16 LoRA adapter fine-tuned
on top of the frozen ProtGPT2 (`nferruz/ProtGPT2`) base, conditioned on
`<AMP>` / `<ANTIBACTERIAL>` / `<ANTIBIOFILM>` tags.

## Architecture

ProtGPT2 (738M params) loaded in 4-bit NF4 (frozen, quantized) with LoRA adapters
(r=16, alpha=32, dropout=0.05) on the `c_attn` attention projections. The 3 tag
tokens' embedding rows and the LM head (`modules_to_save`) were also trained, so
this checkpoint includes their resized weights, not just the LoRA deltas —
that's why `adapter_model.safetensors` is ~1.5GB rather than a few MB.

Trained on a combined AMP/antibacterial/antibiofilm corpus (DRAMP, ESCAPE, BAAMP,
DBAASP; ~24K sequences, 6-50 aa) with leakage-safe identity-cluster train/val
splitting. Checkpoint is epoch 12 (best val_loss=1.2222).

## Files

- `adapter_config.json` / `adapter_model.safetensors` — the LoRA adapter + trained embedding/LM-head weights
- `tokenizer.json` / `tokenizer_config.json` — ProtGPT2 tokenizer with the 3 tag tokens added
- `checkpoint_meta.json` — epoch, train_loss, val_loss at save time

## Usage

Loaded lazily by `ProtGPT2Generator` in `predictor.py`; see `generate(n_peptides, tags=["<AMP>"])`.
Generation only — this package does not implement training. The base ProtGPT2
weights are downloaded from Hugging Face Hub on first load, not stored here.

## Provenance

Exported from `Generation Stage/data/route_B/checkpoints/best` (see `route_B.ipynb`
for the full training pipeline, not reproduced in this platform).

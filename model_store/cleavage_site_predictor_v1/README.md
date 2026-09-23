---
tags:
  - peptide
  - cleavage-site
  - protease
  - stability
  - structure
  - transformer
  - esm2
library_name: pytorch
---

# cleavage-site-predictor-v1

Per-residue protease cleavage-site probability for a candidate peptide (as substrate), relevant to Stage 5's `protease_cleavage_susceptibility` screen (currently a `not_implemented` stub) and to Stage 9 (Synthesis/CMC Feasibility) / stability assessment more broadly.

## Architecture

The UniZyme joint model (structure-only ablation, `UniZyme\SE`): a shared `EnzymeEncoder` + `ActiveSiteHead` produce a single pooled enzyme representation `h^e` (attention-pooled over predicted active-site residues); a `SubstrateEncoder` produces per-residue hidden states `H^s` for the candidate peptide; `CleavageSiteHead` gathers a ±15-residue window of `H^s` around each candidate position, concatenates it with `h^e`, and scores a per-residue cleavage logit via MLP. Both encoders are local-attention (±128 residues), distance-biased transformers (5 layers, 4 heads) over PCA-reduced ESM2 embeddings. Full architecture detail: `architecture.py`'s docstrings and the source project's `MODEL_ARCHITECTURE.md`.

**This model needs a 3D structure, not just a sequence.** Both the enzyme and substrate encoders take a Cα pairwise-distance matrix as a required input (`DistanceBias` biases attention toward spatially close residues). For the bundled enzyme panel, distance matrices are precomputed and cached (`enzymes/distance_matrices/`). For the candidate peptide (substrate), **the caller must supply `distance_matrix`** — this predictor does not run structure prediction itself. Options for producing one: AlphaFold/ColabFold, OmegaFold (used for structure-less substrates in the source project), or any other Cα-coordinate source, then `sqrt(((coords[:,None]-coords[None,:])**2).sum(-1))`.

## Enzyme panel

118 wound-healing-relevant proteases, in `enzymes/enzyme_catalog.csv` alongside their cached PCA-reduced ESM2 embeddings (`enzymes/embeddings/`) and Cα distance matrices (`enzymes/distance_matrices/`):

- **Matrix metalloproteinases** — collagenases, gelatinases, stromelysins (MMP-1/-2/-3/-8/-9/-11/-12/-13/-14/-19/-20/-24/-25/-26 family members, mostly non-human orthologs — see coverage gap below)
- **Cathepsins** — B, D, E, F, H, K, L, S, Z
- **Elastases** — including mouse neutrophil elastase (Q3UP87), pancreatic/macrophage elastases
- **Plasminogen activators** — t-PA, u-PA, plasminogen itself

Call `predictor.available_enzymes()` for the full catalog (accession, protein name, family, EC, active-site positions, sequence).

**Known coverage gap:** human MMP-1, -2, -3, -8, -9, -12, and human neutrophil elastase (the proteases most cited in chronic-wound-exudate literature) are **not** in this panel. Their structures failed the source pipeline's alignment check (resolved-structure length didn't match UniProt sequence length) and were dropped upstream of this migration — this is a gap in the source project's data, not something introduced here. Non-human orthologs of most of these (rat/mouse/rabbit MMPs, mouse neutrophil elastase) are included and can serve as directional evidence of protease-family susceptibility.

## Files

- `architecture.py` — `JointModel` and its submodules (`EnzymeEncoder`, `SubstrateEncoder`, `ActiveSiteHead`, `CleavageSiteHead`, `DistanceBias`), copied verbatim from the source project's `model.py`
- `predictor.py` — `CleavageSitePredictor`: loads the checkpoint + enzyme panel, computes substrate ESM2/PCA features, runs `JointModel.forward_cleavage`
- `joint_model_best.pt` — best checkpoint by combined validation PR-AUC/AUPR (optimizer state stripped: 22MB → 7.6MB)
- `enzymes/enzyme_catalog.csv` — 118-row enzyme metadata (accession, name, family, EC, active-site positions, sequence)
- `enzymes/embeddings/`, `enzymes/distance_matrices/` — precomputed per-enzyme PCA-reduced ESM2 embeddings and Cα distance matrices (`.npy`, one file per accession)
- `enzymes/pca_model.joblib` — the fitted `IncrementalPCA` (480→128 dims), fit on the source project's enzyme train split; reused unmodified for both enzyme and substrate embeddings, exactly as the source project does

## Usage

```python
from cleavage_site_predictor_v1 import CleavageSitePredictor

predictor = CleavageSitePredictor()

# distance_matrix: (len(sequence), len(sequence)) Ca-Ca distances in Angstroms,
# from a resolved or predicted structure of the candidate peptide.
result = predictor.predict_one(sequence, distance_matrix, enzyme_accession="A1E295")
# {"enzyme_accession": "A1E295", "enzyme_name": "cathepsin B precursor",
#  "per_residue_probability": [...], "cleavage_site_positions": [4, 11, ...],
#  "max_probability": 0.91, "n_predicted_sites": 3}

# score against the full 118-enzyme panel (or pass enzyme_accessions=[...] for a subset)
summary = predictor.predict(sequence, distance_matrix)
# {"per_enzyme": {...}, "susceptible_to": ["A1E295", ...], "any_predicted_cleavage": True}
```

Lazy-loaded: ESM2, the PCA model, the enzyme catalog, and the checkpoint load on first `predict()`/`predict_one()` call (or explicit `available_enzymes()` for just the catalog).

## Provenance

Exported from `Cleavage site prediction/unizyme.ipynb` (main notebook) and its upstream, read-only dependencies `unizyme_active_site_pred.ipynb` (enzyme-side data) and `merops_prep.ipynb` (MEROPS enzyme/substrate/cleavage joint table). See that project's `MODEL_ARCHITECTURE.md` for the full pipeline, explicit departures from the UniZyme paper (no energetic-frustration term, PCA-compressed embeddings, local rather than global attention), training/loss/batching details, and open items.

Checkpoint: `unizyme model/joint_model/best_model.pt`, epoch 13 (retrained; **updated 2026-09-16**, supersedes the epoch-6 checkpoint migrated 2026-09-15 — architecture unchanged, weights only), selected by best combined validation PR-AUC/AUPR (`val_pr_auc_cleavage + 10 * val_aupr_active_site`) — training used early stopping on combined validation loss (patience 3) as a separate, deliberately different criterion (see source doc §7). Validation-split metrics at epoch 13: `val_pr_auc_cleavage` 0.8298 (up from 0.14 at the previously migrated epoch 6), `val_aupr_active_site` 0.9916, `val_combined_pr_auc` 5.7876 — see `model_card.json`'s `checkpoint_val_metrics_epoch13` for the full set. No independent held-out evaluation was re-run for this update; these are the source project's own training-time validation-split metrics, not a fresh test-set score.

**Small cleavage training set.** Only ~52 enzymes have usable (enzyme, substrate, cleavage-label) training data, versus ~7,844 for the active-site task — see source doc §3.4 for why (only 52 MEROPS-aligned enzymes pass the manifest + structure-alignment checks). The joint loss weights the active-site task 10x (`LAMBDA_ACTIVE_SITE = 10`, matching the paper's own hyperparameter sweep) partly to compensate. Treat cleavage-site probabilities as a screening signal, not a validated high-confidence prediction.

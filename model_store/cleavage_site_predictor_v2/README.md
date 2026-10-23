---
tags:
  - peptide
  - cleavage-site
  - protease
  - stability
  - transformer
  - esm2
library_name: pytorch
---

# cleavage-site-predictor-v2

Per-residue protease cleavage-site probability for a candidate peptide (as substrate), using **UniZyme** (`Transformer_DE`): a transformer over the substrate and over the enzyme, where the enzyme side also sees its structure (inverse Ca distances) and residue frustration energies, and an active-site head guides the cleavage head.

This replaces the earlier v2 (a retrained joint model with a 121-enzyme panel). The public API is unchanged: `available_enzymes()`, `predict_one()`, `predict()`.

## Inputs

- `sequence`: the candidate peptide, standard 20 amino acids, at most 1500 residues.
- `enzyme_accession`: a UniProt accession from the bundled panel (or a list of them, for `predict()`).
- `distance_matrix` (optional): `(len(sequence), len(sequence))` Ca-Ca distances in Angstroms. When omitted, the peptide is folded with ESMFold (`esmfold_v1`) and the matrix is taken from that structure. Pass it to reuse a structure you already have, such as Stage 7's.

## Enzyme panel (6,460)

Every enzyme in UniZyme's data that has both a structure and a precomputed frustration matrix: its 542 MEROPS-keyed training enzymes, 5,886 more from its pretraining set, and 32 added for the wound-healing and metabolic pipelines (including the 10 wound-allowlist enzymes whose frustration matrices were computed here with `frustratometer`). `enzymes/enzyme_catalog.csv` lists each with its UniProt accession, name, MEROPS id (where it has one), length, sequence and the source of its energy matrix.

The pipelines screen a handful of them:
- **Metabolic (3):** DPP4 (P27487), NEP (P08473) and the Pro-X carboxypeptidase PRCP (P42785, the PPCE in the old panel was P48147).
- **Wound healing (33):** the accessions in `WOUND_PROTEASE_ACCESSIONS` (MMP-1/2/3/8/9/12, neutrophil elastase, cathepsin G, plasmin and others), all present.

Because the panel is large, `predict()` needs the proteases of interest named explicitly.

## Files

Code is in this folder; weights are in `model_store/model_weights/cleavage_site_predictor_v2/` (about 2.2 GB):

- `UniZyme.pt`: checkpoint (saved from `nn.DataParallel`; the `module.` prefix is stripped on load)
- `enzymes/enzyme_catalog.csv`: the panel
- `enzymes/structures/<accession>.npy`: Ca coordinates, `(L, 3)` float32; the distance matrix is computed from them at load
- `enzymes/energy_matrices/<accession>.npy`: configurational frustration matrix, `(L, L)` float16

The panel was built from UniZyme's `data/` folder: Ca coordinates from its structures, frustration matrices copied from `Enzyme_Energy` and `Pretrain_Enzyme/Energy`. For the 10 wound-allowlist enzymes with none, they were computed with `frustratometer` (`AWSEM(structure, k_electrostatics=0, min_sequence_separation_rho=12, min_sequence_separation_contact=12).frustration(kind="configurational")`, the settings in UniZyme's `predictor.py`). Storing energies as float16 changes predictions by at most about 1e-4.

The substrate and enzyme embeddings come from **ESM2-35M** (`facebook/esm2_t12_35M_UR50D`, 480-d), expected at `model_store/model_weights/esm2_t12_35M/` and synced like the 150M model.

## Usage

```python
from model_store.cleavage_site_predictor_v2 import CleavageSitePredictor

predictor = CleavageSitePredictor()
result = predictor.predict_one(sequence, "P27487")
summary = predictor.predict(sequence, ["P27487", "P08473", "P42785"])  # the enzymes to screen are required
# Reusing a structure you already have:
summary = predictor.predict(sequence, ["P27487"], distance_matrix=ca_distances)
```

## Limits

- **The probability threshold is not calibrated.** `CLEAVAGE_PROB_THRESHOLD` in `predictor.py` is 0.5, and no precision or recall has been measured for this checkpoint on peptides here. Treat the probabilities as a screening signal.
- **Ten panel energy matrices were computed here**, with `frustratometer`, because UniZyme shipped none for them. The same procedure reproduced UniZyme's own stored matrix for KLK7 to within 0.033 (values span about -3.5 to 3).
- **Inputs are padded to 1500 residues** as in the original code, so every call is slower than the peptide length suggests.
- Only standard residues are supported.

## Provenance

Architecture, checkpoint and original predictor come from the UniZyme project (`Models/UniZyme`, MIT licence; see its `LICENSE`). Output matches the original `UniZymePredictor` to within 2e-6 for the same enzyme, structure and substrate.

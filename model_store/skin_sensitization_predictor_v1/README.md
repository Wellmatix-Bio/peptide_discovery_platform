# skin_sensitization_predictor_v1

Scores how likely a peptide is a skin sensitizer, from its structure.

- **Model:** equal-weight mean of four probabilities (random forest, LDA, RBF SVM, MLP) on 14 scalar features. Each member's score is its positive-class probability.
- **Features:** the peptide is written as a SMILES string (linear, free termini unless modified, L-residues) and RDKit gives molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds, formal charge, heavy atoms, aromatic rings, Fsp3, heteroatoms, sulfur atoms, carbonyl groups and the electrophilic-alert class count (0-9, heuristic SMARTS, ordinary amides do not count). Column order is `features.FEATURES`.
- **Modifications** (cosmetic pipeline keys): `n_acetyl`, `c_amide`, `d_amino_acid` (0-based), `lipid_tail` (palmitoyl; N-terminus when the position is `"n_terminus"` or 0, otherwise on that Lys). Training peptides included acetylated, amidated, D- and palmitoylated structures.
- **Decision:** a sensitizer at ensemble probability `>= DECISION_THRESHOLD = 0.55`. The constant is part of the model and is not a run parameter.
- **Weights:** `model_store/model_weights/skin_sensitization_predictor_v1/` (`RF.joblib`, `LDA.joblib`, `SVM.joblib`, `MLP.pt`), converted from the notebook's cached bundle (`all_models_d06b2715a8fc680d.joblib`, sklearn 1.9.0, torch 2.14.0). The MLP is stored as a state dict plus its fitted scaler, so the notebook's `scripts/mlp.py` is not needed to load it.
- **Dependency:** `rdkit`, imported on first use.

```python
from model_store.skin_sensitization_predictor_v1 import SkinSensitizationPredictor

pred = SkinSensitizationPredictor()
pred.predict_batch(["KTTKS"], [{"c_amide": True}])   # [{"p_sensitizer", "members": {rf, lda, svm, mlp}}]
```

## Evidence is thin

Training: 1,084 molecules (138 small molecules from tx0c00186, 921 HPPT human patch-test molecules, 25 peptides), 169 sensitizers. Only 25 are peptides; the rest are small molecules with a much wider property range.

Held-out peptides (24, 8 positive; CD-HIT cluster split, already inspected before this ensemble was chosen):

| Threshold | TP | FP | TN | FN | Balanced acc | Precision | Recall | AUC |
|---|---|---|---|---|---|---|---|---|
| 0.55 | 5 | 3 | 13 | 3 | 0.719 | 0.625 | 0.625 | 0.734 |

The notebook states 0.55 "was explored on held-out peptides for RF; not validated for this ensemble". Cross-validated AUC on the training data: RF 0.863, MLP 0.836, SVM 0.835, LDA 0.802. With 24 test peptides these numbers carry wide uncertainty. Known cosmetic peptides can score above the threshold (KTTKS scores 0.60), so expect many rejections when rejection is on.

Training code and metrics: `Models/SkinSense/training.ipynb`.

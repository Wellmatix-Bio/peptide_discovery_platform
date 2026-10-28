# allergenicity_predictor_v1

Scores how likely a protein sequence is to be an allergen, from sequence alone.

- **Model:** equal-weight mean of three probabilities from an RBF SVM, an LDA and an MLP, all on one mean-pooled ESM2-150M embedding. Each member's 0.5 is its own class boundary (SVM margin 0, MLP logit 0, LDA posterior 0.5), so the ensemble reads as an average vote with confidence. It is not a calibrated probability.
- **Embedding:** `facebook/esm2_t30_150M_UR50D` from the shared `FeatureExtractor`, mean over residues with BOS/EOS excluded.
- **Weights:** `model_store/model_weights/allergenicity_predictor_v1/` (`SVM.joblib`, `LDA.joblib`, `MLP.pt`).
- **Input:** 20 uppercase standard amino acids, length 1-1022. Longer sequences are rejected, not truncated.

```python
from model_store.allergenicity_predictor_v1 import AllergenicityPredictor

pred = AllergenicityPredictor()
pred.predict_proba_batch(sequences, feature_extractor)   # list[float], P(allergen)
pred.predict_batch(sequences, feature_extractor)         # list of {"p_allergen", "members": {svm, lda, mlp}}
```

## Trained on full-length proteins

The training data (AlgPred 2.0) holds proteins of 50-3,408 residues, median 170-345. **Nothing shorter than 50 residues was seen.** On random 3-12-residue peptides the ensemble spreads over 0.09-0.91 with 40% at or above 0.5, and the cosmetic peptide KTTKS scores 0.81, so values on peptides are not meaningful. The wrapper still scores any length (`TRAINED_MIN_LENGTH = 50` is exposed), and callers decide: cosmetic Stage 8 does not score sequences under 50 residues by default.

## Validation (4,030 proteins)

| Model | Sens | Spec | Acc | AUC | MCC |
|---|---|---|---|---|---|
| SVM | 0.810 | 0.979 | 0.895 | 0.971 | 0.801 |
| LDA | 0.807 | 0.965 | 0.886 | 0.963 | 0.782 |
| MLP | 0.738 | 0.986 | 0.862 | 0.945 | 0.747 |
| Ensemble (mean of probabilities) | 0.774 | 0.981 | 0.878 | 0.973 | 0.772 |

The notebook does not define the ensemble rule. The mean of probabilities has the best AUC of the rules tried (majority vote 0.925, mean of logits 0.971, the latter dominated by LDA's much larger scale) but does not beat SVM alone on MCC. The notebook's random forest (validation MCC 0.688) is left out.

Training code and metrics: `Models/Allergenicity/training.ipynb`.

# Screening models: validation and applicability report

What was checked: the six models behind `s05_food_screening` (`src/food/c_discovery/screening/adapters.py`), on real weights. For each: does the score point the right way (higher = safer or more developable), is the 0.5 label the model's own boundary, and does the applicability band match the training data? For allergenicity, which of the two score definitions to use. The spec asks to "verify allergenicity and bitterness threshold calibration and score direction".

Reproduce: `DEV_MODE=true python scripts/food_validation/models.py` (needs the real weights; writes `model_validation.json` here).

## Verdict

| Model | Score direction | 0.5 label threshold | Applicability labels | Overall |
|---|---|---|---|---|
| Allergenicity | Correct | Equals the model's boundary | Correct (7-20 matches the weights) | Direction and bands verified. **The hold-out used here is probably not independent of the training data**, so the card's 0.729 AUC stands, not the 0.852 measured here. Not calibrated. |
| Cytotoxicity | Supported by the card and two examples only | Not verified | Plausible, not verified | **Not verified on labelled data.** No clean labelled set exists in the repo. |
| Solubility | Correct (AUC 0.958 on training data) | Plausible | **Understated: 42% of the pool is shorter than any training peptide** | Direction verified; applicability needs a fix. |
| Aggregation | Correct (AUC 0.714 on held-out data) | Misses over half of aggregators | **Understated: the training data are mostly 20-mers** | Direction verified. The model is weak and misses known amyloid peptides. |
| Synthesis feasibility | Correct (AUC 0.977 on training data) | Plausible | **Understated: 52% of the pool is shorter than the PepSySco training range** | Direction verified; applicability needs a fix. |
| Bitterness | Correct (AUC 0.943 on training data) | Equals the model's own boundary | Correct (2-14 matches the training percentiles) | Direction and bands verified. |

"Direction" means: peptides in the safe or developable class get higher scores than peptides in the unsafe class. Where the labelled data is the model's own training data, only the direction can be read from it, not the performance.

Sensitivity below is the share of the unsafe or undevelopable class that the 0.5 label flags (IgE-binding, bitter, insoluble, aggregating, difficult to synthesise). Specificity is the share of the safe or developable class that it passes.

## Results by model

### Allergenicity
- **Hold-out:** 1,591 labelled peptides from `iedb_ige_peptides_splits.csv` (490 IgE-binding). The adapter's score ranks the non-binding above the binding with **AUC 0.852**; sensitivity 0.635, specificity 0.896, MCC 0.553 at the 0.5 label.
- **This is probably not a clean hold-out.** The model card reports AUC 0.729, sensitivity 0.452, MCC 0.370 on 1,581 hold-out peptides. The splits file is newer than the deployed weights (20 November against 19 November) and has 1,750 hold-out peptides where the card says 1,779. Some of these peptides may have been in the training data, which would explain the higher figures. **The card's numbers are the ones to quote.** Direction is confirmed either way.
- **Threshold:** the weights record a decision threshold of 0.0 on the raw score, which is p = 0.5. The label threshold is the model's own boundary.
- **Threshold search:** the MCC-optimal threshold on cross-validation did not beat the default: on the MLP, SVM and GAT models and their ensembles it raised cross-validated MCC by at most 0.001 and never raised hold-out MCC (default 0.321 to 0.377, searched 0.311 to 0.353), so the 0.5 label threshold is calibrated.
- **Calibration of the probability:** `p_allergen` is not a probability. On this hold-out, predicted 0.1 to 0.4 corresponds to observed rates of 0.06 to 0.16 (over-predicts), and predicted 0.6 to 0.8 to observed 0.78 to 0.94 (under-predicts). The card says the same ("uncalibrated"). If the hold-out is partly in the training data, the real gaps are larger.
- **Applicability:** the 7-20 residue "in domain" band equals the 5th and 95th percentile of the training lengths stored in the weights (3 to 49, p5 7, p95 20). AUC is 0.822 inside the band and 0.890 outside (131 peptides), so the model does not do worse outside it on this set.

### Bitterness
- **Data:** `BTP-720-test.fasta`, 72 bitter and 72 not. These peptides are part of the deployed model's training data, so this reads the direction only.
- **Direction:** AUC 0.943; sensitivity 0.75, specificity 0.944, MCC 0.708 at the 0.5 label. The card reports cross-validated AUCs of 0.82 and 0.83.
- **Threshold:** the label is predicted bitter at 0.5, the model's own boundary.
- **Applicability:** the 2-14 residue band equals the 5th and 95th percentile of the training lengths on the card.

### Solubility
- **Data:** ultrapure-water rows of `long_peptide_solvent.csv` (827 soluble, 510 insoluble), training data. Direction only.
- **Direction:** AUC 0.958; sensitivity 0.845, specificity 0.925 at the 0.5 label.
- **Applicability:** the adapter marks every peptide "limited". The training peptides are 7 to 43 residues (5th percentile 9, median 25). In the 400-peptide pool, 217 (54%) are under 9 residues and 170 (42%) are under 7, shorter than any training peptide. "Limited" understates this; those are outside the data.
- The card says the training length range is not documented, so the adapter's choice was a placeholder.

### Aggregation
- **Data:** the model's own held-out split (`train_test = test`, 10,956 rows; 1,500 sampled, 21% aggregating). This is a true held-out check.
- **Direction:** AUC 0.714, in line with the card's 0.70. Sensitivity 0.455, specificity 0.846, MCC 0.298 at the 0.5 label: it misses more than half of the aggregating peptides.
- **Known amyloid peptides:** KLVFFAE (Abeta 16-22) scored 0.755, GNNQQNY 0.877 and VQIVYK 0.606, all above 0.5, meaning "not aggregation-prone". The model misses all three. The check is an illustration with three peptides, not a validation, but together with the 0.45 sensitivity it means a good aggregation score is weak evidence of safety.
- **Applicability:** the adapter marks all scored peptides "limited". The training peptides are mostly 20 residues (median 20, 95th percentile 20, 5th percentile 4). In the pool, 249 (62%) are 10 residues or shorter. The model does not run on peptides under 5 residues; that is 115 of 400 (29%), reported "not run".

### Synthesis feasibility
- **Data:** `PepSySco.csv` (1,540 feasible, 231 not), part of the training data. Direction only. In-sample AUC 0.977 is far above the card's cross-validated 0.74 to 0.78, so it says nothing about performance.
- **Direction:** correct; sensitivity 0.749, specificity 0.978 at the 0.5 label.
- **Applicability:** PepSySco peptides are 8 to 25 residues (median 15). The deployed model also used MS2-labelled peptides whose lengths are not in the repo. In the pool, 206 (52%) are under 8 residues. The adapter marks all "limited".

### Cytotoxicity
- **No labelled set in the repo.** The card names DRAMP aggregate and HaCaT assay data (161 cytotoxic peptides, balanced to 322), and the label is derived in the training notebook.
- **What exists:** the card's fold AUROC of 0.73 to 0.83 on 322 balanced peptides (cell type DRAMP aggregate, mostly antimicrobial peptides). Two cytotoxic peptides, melittin (score 0.371) and LL-37 (0.242), score low, so flagged. Three benign peptides score 0.59 to 0.79, and KDEL scores 0.394, which is flagged although it is a benign tetrapeptide. That is an illustration, not a validation.
- **Applicability:** "in domain" for 5 to 50 residues (the model's maximum length). 115 of 400 pool peptides (all 4 residues) are "limited". The training population is antimicrobial peptides, so it is far from food peptides in kind, which the band does not show.

## Allergenicity score definition
The pipeline scores allergenicity as `1 - p_allergen`. Phase 1 used 1 minus the percentile rank of the peptide's score among the IEDB hold-out peptides.

- **They rank peptides almost identically.** Spearman 0.9998 on the hold-out and 0.9996 on the 400-peptide pool; AUC 0.8519 against 0.8520. Both are monotone transforms of the same score.
- **The scales differ.** On the pool, the 5th, 50th and 95th percentile of the score are 0.001, 0.073 and 0.830 for Phase 1, and 0.069, 0.276 and 0.800 for `1 - p`. The percentile version is lower and spreads low scores more.
- **Effect on the ranking:** with each definition, the 400-peptide pool gives **the same 25 final peptides**. The order differs a little (rank correlation 0.978).
- **Effect on safety:** the safety objective is the lower of the allergenicity and cytotoxicity scores. Allergenicity is the lower one for 368 of 400 peptides with Phase 1's score and 358 with `1 - p`. Under either definition the safety objective is in practice the allergenicity score.
- **Which to keep:** `1 - p` is already in place and gives the same shortlist. The percentile version needs a reference file (`holdout_reference_z.npy`) that is not in the platform's weights folder. I recommend keeping `1 - p`, with the "not a probability" caveat on every report. Choose the percentile only if the biology lead wants scores read relative to the IEDB hold-out.

## Applicability against the real pool
The pool is Phase 1's 400-peptide screening set (4 to 40 residues, median 7). What the adapters label, and what the training data say:

| Model | Adapter label (of 400) | Training lengths | Pool outside the training range |
|---|---|---|---|
| Allergenicity | 190 in domain, 210 limited | 3-49 (p5 7, p95 20) | 210 outside 7-20 (52%) |
| Cytotoxicity | 285 in domain, 115 limited | up to 50 | 115 under 5 residues |
| Solubility | 400 limited | 7-43 (p5 9) | 170 under 7 (42%) |
| Aggregation | 285 limited, 115 not run | mostly 20 (p5 4) | 249 at 10 residues or fewer (62%) |
| Synthesis feasibility | 400 limited | PepSySco 8-25 | 206 under 8 (52%) |
| Bitterness | 324 in domain, 76 limited | 2-14 (card p5-p95) | 76 over 14 (19%) |

Bands for solubility, aggregation and synthesis feasibility are placeholders ("limited" for everything). The data above support a stricter band for each, such as "out of domain" below the shortest training peptide. That changes the coverage score and which models count in the safety objective, so I have not changed it.

## Limits of this report
- Only allergenicity (partly) and aggregation have a held-out check. Solubility, synthesis and bitterness numbers come from training data and show direction only.
- Cytotoxicity has no labelled check. Aggregation's amyloid examples are three peptides.
- The percentile definition was evaluated on a hold-out that may overlap the training data; the ranking result does not depend on that, since it uses Phase 1's 400-peptide pool.
- The models are trained on other peptide populations (IgE epitopes, antimicrobial peptides, random 20-mers). None was trained on food-derived peptides.

## Decisions for you
1. Replace the placeholder "limited" bands with stricter ones from the training lengths (solubility, aggregation, synthesis feasibility)?
2. Keep `1 - p_allergen` (my recommendation)?
3. Keep the aggregation model in the safety and developability objectives, given its sensitivity of 0.45 and the three missed amyloid peptides, or report it for information only?

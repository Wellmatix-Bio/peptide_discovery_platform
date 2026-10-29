# Cosmetic benchmark

Does the cosmetic pipeline correctly eliminate peptides without cosmetic use (label 0) while keeping those with cosmetic use (label 1)? This report runs one labelled set through cosmetic Stages 5 to 11 and records which peptides survive each stage, split by label. The notebook is `notebooks/cosmetic_benchmark.ipynb`, local and not tracked. The methods and tables below come from its saved outputs.

## Data

Source: `cosmetic_peptides_use_evidence_benchmark 1.csv`, 218 rows.

| Group | Rows | Used |
|---|---|---|
| Label 1 (cosmetic-use positive) | 47 | 44 |
| Label 0 (negative) | 149 | 149 |
| Unknown | 22 | 22 |
| Non-standard residues, excluded | 3 | 0 |
| **Scored** | | **215** |

The three excluded rows are `KDI[Cit]`, `AP[Dab]` and `[pE]K`. All three are label 1, so the positive group shrinks from 47 to 44. The unknown rows are run but not counted in the label-split rates.

By family, the scored rows are 88 soybean, 59 silk, 45 rice and 23 original cosmetic peptides.

## Method

- **Stages run:** 5, 6, 7, 8, 9 and 11, in that order. Stage 4 is skipped because the peptides are given. Stage 1 is skipped, and the brief is built directly with primary function `collagen_ecm`, product pH 5.5, standard safety level and dermis depth.
- **Sequences:** scored unmodified. The source has no acetyl, amide or lipid fields, so palmitoyl and acetyl peptides are scored as bare sequences.
- **Seed reference:** the 23 original-table rows (21 unique sequences) are the seed set for Stage 5 percentile rules and Stage 8 irritation cut-offs. These rows are also in the benchmark set, so the reference is not independent of the labels.
- **Survival:** a candidate that a stage removes does not reach the next. Each stage's survivors are counted by label.

## Results

### Funnel

| Stage | Remaining | Removed | Label 1 | Label 0 | Unknown |
|---|---|---|---|---|---|
| Input | 215 | | 44 | 149 | 22 |
| 5 physchem | 117 | 98 | 28 (64%) | 81 (54%) | 8 (36%) |
| 6 primary function | 79 | 38 | 21 (48%) | 53 (36%) | 5 (23%) |
| 7 structure | 79 | 0 | 21 (48%) | 53 (36%) | 5 (23%) |
| 8 safety | 11 | 68 | 8 (18%) | 3 (2%) | 0 (0%) |
| 9 synthesis | 11 | 0 | 8 (18%) | 3 (2%) | 0 |
| 11 ranking | 11 | 0 | 8 (18%) | 3 (2%) | 0 |

Percentages are of each group's input count.

### Rejections by stage

- **Stage 5** (98 removed): net charge out of range 48, repeated residues 24, insoluble 23, hydrophobicity outside the seed range 18, molecular weight above 3000 Da 6, membrane-active profile 1. Several peptides fail more than one rule, so the reason counts add up to more than 98.
- **Stage 6** (38 removed): primary-function score below 0.5, 26; antimicrobial filter, 12.
- **Stage 8** (68 removed): half-life, 56; skin sensitization, 6; cytotoxicity, 3; hemolysis, 3. The reason counts add up to 68.
- **Stages 7, 9 and 11:** nothing removed.

### Final survivors (Stage 11 order)

| Rank | Peptide | Sequence | Label | Final score | Pareto front |
|---|---|---|---|---|---|
| 1 | Soybean peptide (LPYP) | LPYP | 0 | 88.6 | 1 |
| 2 | Acetyl Tetrapeptide-9 | QDVH | 1 | 83.9 | 1 |
| 3 | Acetyl Tetrapeptide-11 | PPYL | 1 | 82.8 | 2 |
| 4 | Copper Tripeptide-1 | GHK | 1 | 75.7 | 1 |
| 5 | Manganese Tripeptide-1 | GHK | 1 | 75.7 | 1 |
| 6 | Palmitoyl Tripeptide-1 | GHK | 1 | 75.7 | 1 |
| 7 | Leu-Pro-Lys | LPK | 1 | 75.4 | 1 |
| 8 | Palmitoyl Tetrapeptide-7 | GEPR | 1 | 73.7 | 3 |
| 9 | Soybean peptide (VHVV) | VHVV | 0 | 73.5 | 3 |
| 10 | Red rice bran peptide YFP | YFP | 1 | 71.4 | 2 |
| 11 | Rice peptide TNPWHSPRQ | TNPWHSPRQ | 0 | 54.2 | 4 |

None of the six skin-sensitizer predictions survived Stage 8, so every final survivor has the sensitizer flag off.

## Reading the results

- **Negatives are eliminated well; cosmetic-use peptides are eliminated too.** Of the 149 label-0 peptides, 146 are removed (98%). Of the 44 label-1 peptides, 36 are removed (82%), so only 8 of 44 (18%) are kept. The 11 survivors are 8 positive and 3 negative (73% positive). The target is high negative elimination with high positive retention, and the pipeline reaches the first but not the second.
- **Where each label is lost:** Stage 5 removes 46% of negatives and 36% of positives, so it does not separate them. Stage 6 removes 35% of remaining negatives and 25% of remaining positives. Stage 8 removes 94% of the remaining negatives and 62% of the remaining positives. Stage 8 is the only stage that separates the labels, and it also removes most positives.
- **Within each class.** Cosmetic-use peptides: 8 of 44 kept (18.2%, 95% CI 9.5 to 32.0). No-use peptides: 3 of 149 kept (2.0%, 95% CI 0.7 to 5.8). A Fisher exact test on the two kept rates gives odds ratio 10.8 and p = 0.0004, so the classes are kept at different rates. Given the seed overlap and the small survivor count, the size of the effect is uncertain.
- **Across classes.** Labelled set, cosmetic-use as the positive class: sensitivity 18.2%, specificity 98.0%, positive predictive value 72.7%, negative predictive value 80.2%. Accuracy is 79.8%, against 77.2% for removing everything, because only 22.8% of labelled peptides are cosmetic-use. Balanced accuracy is 58.1%. Likelihood ratios are 9.0 for a kept peptide and 0.83 for a removed one. A filter that kept 5.1% of peptides at random would keep about 2.3 cosmetic-use and 7.6 no-use peptides, so the pipeline's error profile is far from random, but it misses most cosmetic-use peptides.
- **Most of the separation comes from Stage 8, and mainly from half-life.** Half-life removes 56 of the 79 peptides that reach it. The half-life model was trained on longer peptides (5 to 50 residues), and this set is mostly 2 to 12 residues. Its cut therefore reflects length as much as stability. The unknown group is no help here: none of its 5 Stage 8 inputs survive.
- **The ranking carries little signal.** The top-ranked peptide is a label-0 soybean peptide. Stage 11 scores only 11 survivors, so the primary-function percentile is coarse, and the Pareto fronts are shallow.
- **Duplicate sequences:** `GHK` appears three times (copper, manganese and palmitoyl tripeptides) and receives identical scores in all three rows. Each is counted as a separate candidate, so the survival and ranking figures count `GHK` three times.
- **The reference is not independent:** the seed set is part of the scored set, so the Stage 5 and Stage 8 percentile cut-offs partly reflect the labels.

## Limitations

- No AUROC, bootstrap interval or top-k enrichment was computed. The notebook reports survival counts only. The wound-healing report uses those measures, so the two are not directly comparable yet.
- Stage 6 tested the collagen primary function only. Antioxidant and pigmentation were not run, so the survival figures apply only to collagen-ECM targeting.
- **Stage 6 AMP filter: resolved.** The notebook ran under the `peptide_platform` conda kernel (scikit-learn 1.9.0, the repo pin). The earlier failure came from the default Python 3.13 install, which has scikit-learn 1.6.1. In that install `amp_classifier_v1` raises `'LogisticRegression' object has no attribute 'multi_class'`. To confirm, the 12 antimicrobial rejections were recomputed from the saved Stage 5 survivors. The same 12 peptides exceed 0.6 (`DPR`, `FDPAL`, `HFR`, `HHL`, `LH`, `MGKVVNPTQK`, `PGTAVFK`, `SWLRL`, `WGWNPR`, `WH`, `WLRL`, `WMHVR`), leaving 79 survivors. Run the platform in the `peptide_platform` environment.
- Peptides are scored without modifications, so the palmitoyl, acetyl and amide forms are represented by their bare sequences.
- The allergenicity model does not run on sequences under 50 residues, so it does not affect these results.
- The skin-sensitization model has thin evidence (24 held-out peptides); see `model_store/skin_sensitization_predictor_v1/README.md`.
- The penetration filter is disabled, so target depth does not remove anything in this run.

## Next steps

1. Re-run with a seed set that is not in the scored set.
2. Check the half-life model on cosmetic-length peptides, or mark them not screened as the other models do for short sequences.
3. Report positive retention and negative elimination with confidence intervals, and add AUROC, bootstrap interval and top-k enrichment, matching `docs/BENCHMARK.md`.
4. Check whether Stage 8's half-life cut removes positives in particular: the reason for each removal is not broken down by label in the current outputs.
5. Repeat for the antioxidant and pigmentation primary functions.

## Reproduce

Open `notebooks/cosmetic_benchmark.ipynb` and run all cells. Outputs go to `artifacts/benchmark/cosmetic/`, including `final_survivors.csv` and the per-stage audit log. The source CSV path is set in the first parameters cell.

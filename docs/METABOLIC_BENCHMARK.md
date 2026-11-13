# Metabolic benchmark

Can the metabolic pipeline tell metabolic peptides from non-metabolic ones, keeping the first and filtering out the second? This report runs one labelled set through metabolic Stages 5 to 11 and counts which peptides survive each stage. Label 1 peptides (demonstrated metabolic benefit) should be kept, and label 0 peptides (tested without benefit) should be removed.

**Result in one line:** no. The pipeline removed 52 of 54 peptides, and it removed 94% of the label-1 peptides as well as every label-0 peptide. The gap between the two (6.5% kept against 0%) is within chance.

## Data

Source: `metabolic_peptides_verified_partial.csv`, 54 rows, all with standard residues (6 to 44 residues).

| Group | Rows |
|---|---|
| Label 1: demonstrated benefit (glucose, weight or insulin sensitivity) | 31 |
| Label 0: tested null result, or a tested antagonist or scrambled control | 23 |
| **Scored** | **54** |

Of the 54 rows, 47 are animal experiments and 7 are human interventions. The file states that each label holds for the specified experiment only, that nulls are not proof of inactivity, and that the negative count is below its target. It has no untested or in-vitro-only negatives.

By family, the largest are GIP (8, all label 0), Esculentin (6), GLP-1 (4), Humanin (4) and Tigerinin (4, all label 1).

## Method

- **Stages run:** 5, 6, 7, 8, 9 and 11, in that order. Stage 4 is skipped because the peptides are given. Stage 1 is skipped, and the brief is built directly.
- **Brief:** target receptors GLP1R and GIPR, length 5 to 50. This is the repo's example metabolic brief, fixed before the run. Stage 6 requires a predicted EC50 at or below 1000 nM at every selected receptor.
- **Sequences:** scored as given. The file records terminal chemistry only as free text, so amidation and similar modifications are not scored.
- **Survival:** a peptide a stage removes does not reach the next stage.
- **Measure:** the question is a filter, so the measures are the share of each label that survives, and how those shares differ. Stage 11 only orders survivors and removes nothing, so ranking is not part of this question.
- **Without training peptides:** a second analysis sets aside the 8 peptides in the GLP-1, exendin, amylin and glucagon-derived families, which the potency and binding models were probably trained on. The family list is an assumption and has not been checked against the training sets.
- **Statistics:** Wilson 95% intervals and a Fisher exact test on the two kept rates.

## Results

### Funnel

| Stage | Remaining | Removed | Label 1 | Label 0 |
|---|---|---|---|---|
| Input | 54 | | 31 | 23 |
| 5 physchem | 48 | 6 | 29 (93.5%) | 19 (82.6%) |
| 6 functional | 26 | 22 | 16 (51.6%) | 10 (43.5%) |
| 7 structure | 26 | 0 | 16 (51.6%) | 10 (43.5%) |
| 8 safety | 3 | 23 | 3 (9.7%) | 0 |
| 9 synthesis | 2 | 1 | 2 (6.5%) | 0 |
| 11 ranking | 2 | 0 | 2 (6.5%) | 0 |

Percentages are of each group's input count.

### Removal rate at each stage

The share of the peptides that reached a stage which that stage removed. A stage that separates the labels removes more label 0 than label 1.

| Stage | Label 1 removed | Label 0 removed |
|---|---|---|
| 5 physchem | 2 of 31 (6.5%) | 4 of 23 (17.4%) |
| 6 functional | 13 of 29 (44.8%) | 9 of 19 (47.4%) |
| 7 structure | 0 of 16 | 0 of 10 |
| 8 safety | 13 of 16 (81.3%) | 10 of 10 (100%) |
| 9 synthesis | 1 of 3 (33.3%) | none reached it |

### Rejections by stage

A peptide can fail for more than one reason.

| Stage | Reason | Label 1 | Label 0 |
|---|---|---|---|
| 5 | Hydrophobicity above the limit | 2 | 4 |
| 6 | Predicted EC50 above the limit at GIPR | 11 | 7 |
| 6 | Predicted EC50 above the limit at GLP1R | 2 | 2 |
| 6 | Anti-inflammatory probability below 0.5 | 0 | 2 |
| 8 | Cytotoxicity | 6 | 9 |
| 8 | Hemolysis | 5 | 8 |
| 8 | Plasma half-life | 5 | 3 |
| 8 | Aggregation tendency | 1 | 0 |
| 9 | Synthesis difficulty class `high` | 1 | 0 |

### Kept rates and classification measures

| Measure | Result |
|---|---|
| Label 1 kept | 2 of 31 = 6.5% (95% CI 1.8 to 20.7) |
| Label 0 kept | 0 of 23 = 0.0% (95% CI 0.0 to 14.3) |
| Fisher exact test | p = 0.50 |
| Sensitivity / specificity | 6.5% / 100% |
| PPV / NPV | 100% / 44.2% |
| Accuracy / balanced accuracy | 46.3% / 53.2% |
| Chance baseline at the same kept fraction (3.7%) | 1.1 label 1 and 0.9 label 0 |

### Without the likely training peptides

Setting aside the 8 peptides in the GLP-1, exendin, amylin and glucagon-derived families leaves 46 peptides: label 1 kept 2 of 24 (8.3%, 95% CI 2.3 to 25.8), label 0 kept 0 of 22 (0.0%, 95% CI 0.0 to 14.9).

### Survivors

| Peptide | Label |
|---|---|
| Esculentin-2CHa(1-30) | 1 |
| [L28K]Esculentin-2CHa | 1 |

## Reading the results

- **The filter does not separate the labels.** 6.5% of label-1 and 0% of label-0 peptides survive. A random filter of the same size would keep about 1 of each, and the Fisher test gives p = 0.50.
- **The 100% specificity and 100% PPV are not evidence.** They follow from the pipeline removing almost everything. NPV of 44.2% is about the label-0 share of the data (42.6%), and removing every peptide would score 42.6% accuracy, against 46.3% here.
- **Stage 5 and Stage 8 point the right way, Stage 6 does not.** Stage 5 removes 17% of label 0 against 6.5% of label 1, and Stage 8 removes 100% against 81%. Stage 6 removes about the same share of each (47% and 45%).
- **Stage 6 is largely a brief effect.** 18 of its 22 removals fail the GIPR potency gate, including GLP-1(7-36)-amide, GLP-1(7-37) and liraglutide, which act on GLP1R. A dual-receptor brief rejects single-receptor peptides by design. This result says little about the peptides' benefit, and the single-receptor briefs have not been run.
- **Stage 8 removes almost everything that reaches it.** It removed 23 of 26 peptides: 13 of the 16 label-1 and all 10 label-0. That points the right way, but the difference is not significant (Fisher p = 0.26), and 8 of the 10 label-0 removals are the GIP-family peptides, which fail on hemolysis or cytotoxicity as a group. Cytotoxicity (15 peptides), hemolysis (13) and half-life (8) also removed most of the label-1 peptides. It rejected approved drugs too: exenatide (hemolysis and cytotoxicity), lixisenatide (cytotoxicity) and pramlintide (hemolysis and half-life). Those calls are probably false alarms. The models' training data is a likely reason, but it has not been checked.
- **The survivors are not incretin-like.** Both are variants of esculentin-2CHa, a frog-skin antimicrobial peptide. They passed the predicted potency gates at both receptors. This benchmark does not show whether they act through those receptors.
- **Family structure.** All 8 GIP-family peptides (label 0) were removed, and all 4 Tigerinin peptides (label 1) were removed, three at Stage 8 for half-life and one at Stage 6. Peptides within a family get similar predictions, so the effective number of independent results is smaller than 54.
- **Setting aside likely training peptides changes nothing.** The kept rates are the same without GLP-1, exendin, amylin and glucagon-derived entries, so there is no sign that memory of known drugs inflated the result.

## Limitations

- **Label 0 is not "non-metabolic".** Most label-0 rows are fragments, antagonists or scrambled controls of metabolic peptides (the GIP set, scrambled CAP-GDF15), or other peptides tested without benefit (temporins, obestatin). None is a random or unrelated peptide, so this is a harder test than separating metabolic peptides from unrelated ones.
- **Labels are experiment-specific.** A label 1 is benefit in one assay, and a label 0 is a null in one assay. Neither is a general statement about the peptide.
- **Mechanism mismatch.** Many label-1 peptides (frog antimicrobial peptides, humanin, MOTS-c, apelin) lower glucose by routes other than GLP1R, GIPR or GCGR agonism, and the pipeline screens for receptor agonism. Label 0 includes GIP fragments and antagonists, which bind the receptor.
- **Small and partial sample.** 54 rows, few survivors, wide intervals, and related peptides that are not independent.
- **Uncalibrated thresholds.** The Stage 8 limits (hemolysis, cytotoxicity, half-life, aggregation) are working values. The Stage 6 anti-inflammatory gate (0.5) is new. The agonism classifier has no probability threshold, so it does not gate.
- **Stage 5 thresholds** are fixed values. The data they came from has not been checked for overlap with this set.
- **Training overlap is assumed from family names**, not checked against the training data.

## Next steps

1. Rerun with the single-receptor briefs (`["GLP1R"]` and `["GIPR"]`). Stages 5, 7, 8 and 9 do not depend on the brief, so only the receptor gate changes.
2. Check the Stage 8 hemolysis and cytotoxicity calls on exenatide, lixisenatide and pramlintide, and on the training data of those two models. Mark peptides outside a model's domain as not screened.
3. Replace the assumed training overlap with the models' actual training lists.
4. Add negatives that are untested, in-vitro-only or unrelated to metabolism (for example random peptides with matched lengths), and use family-aware grouping when counting results.

## Reproduce

Open `notebooks/metabolic_benchmark.ipynb` with the `peptide_platform` kernel and run all cells. Set `RECEPTORS` in the parameters cell first, and do not change it after seeing results. Stage 7 takes about 5 minutes on a GPU. Outputs go to `artifacts/benchmark/metabolic_verified/`, including `final_survivors_<receptors>.csv` and `rejections_<receptors>.csv`. The notebook is generated by `notebooks/build_metabolic_benchmark_notebook.py`.

# Limitations

## Both pipelines

- **Standard residues only.** Most models read the 20 standard amino acids. Peptides with D-residues, lipidation, Aib, Nle or other modifications cannot be scored by the potency, binding or hemolysis models and are left unscored or flagged.
- **Length limits.** Models were trained on limited ranges (hemolysis up to 100 residues, half-life 5-50, aggregation at least 5). Outside them a score is extrapolation.
- **Thresholds are not calibrated.** The cleavage-stability reject level (0.3) and cleavage probability cut-off (0.5) were set without calibrating them. The hydrophobicity cut-offs (0.3 metabolic, 0.5 wound) and the other Stage 5 flag lines are working values. The wound hemolysis cut-off (pHC50 4.0) has not been re-checked since the switch to `hemolysis_predictor_v2`.
- **Structures are low confidence.** ESMFold predicts peptides from one sequence, with no alignment, and short peptides are usually poorly resolved. Exposure and distance matrices inherit this.
- **Aggregation depends on `propy3`.** The model (ROC-AUC 0.70, PR-AUC 0.45) needs a GPL-licensed package that is installed only in the worker images. Where it is missing, the screen should report `not_screened`, but Stage 8 currently crashes instead.
- **Small-sample models.** `cytotoxicity_predictor_v1` was trained on 161 positives, and its non-default cell types rest on few examples.
- **No wet-lab, formulation or dosing model.** Delivery, manufacturing yield and clinical dosing are not predicted.
- **The Stage 8 cytotoxicity and hemolysis checks rejected known winners at least as often as random peptides** in all three sets run. See [Filters that rejected winners](#filters-that-rejected-winners).

## Wound healing pipeline

- **Tag-conditioned Route B** only knows the tags `<AMP>`, `<ANTIBIOFILM>` and `<ANTIBACTERIAL>`. Would require more tags in the future for more diverse conditional generation.

## Metabolic pipeline

- **Not modeled:** GI tolerability, immunogenicity, selectivity, signaling bias, weight-loss efficacy and dosing.
- **Generation:** Route B is untagged, so it is not conditioned on the target. Nevertheless, a PEFT modified model was used, so the results can be influenced by fine-tuned AMP, ANTIBIOFILM and ANTIFUNGAL masked data.

## Cosmetic pipeline

- **No penetration model, and no penetration filter.** Stage 8 computes a rule-based penetration check (epidermis: MW up to 1000 Da or a lipid tail; dermis: lipid tail required) and records it as `predictions["penetration"]`, but it no longer rejects. The rule was too blunt: nothing in Stage 4 adds a lipid tail, so a dermis brief rejected every candidate (all 193 of 209 that reached Stage 8 in `cosmetic-001` failed it). Until a model exists, candidates are not screened for skin penetration at all, so a dermis or epidermis target says nothing about whether the peptide reaches it. Stage 11's `delivery_fit` scores molecular weight only.

## Benchmark

See [BENCHMARK.md](BENCHMARK.md). What limits what it can show:

- **Only three of six sets have been run** (`TC-01`, `TC-02`, `GLP1R`, with 5 to 18 winners each). Counts this small give wide intervals, so a result is not conclusive either way.
- **The data was extracted from the literature by an AI model and has not been verified by hand.** Sequences, measured values, group labels, evidence tiers and citations may contain errors, including misread tables or invented entries. Evidence strength also differs between peptides. A wrong sequence or label can move a result as much as the platform does.
- **Few toxic examples.** Outside `TC-30`, each set has one to two group B peptides, so "toxic ones rank low" is barely testable.
- **Most metabolic winners cannot be scored as written.** They contain Aib, D-residues or lipidation. The main metabolic result covers only the few that are standard, and the second table uses approximated sequences, which may not behave like the real drugs.
- **Sequence-only scoring.** A lipidated or chemically different form of the same sequence gets the same score. Sequences labelled differently for that reason (for example acyl-chain variants) were left out.
- **Briefs are trimmed to the pipeline's vocabulary.** Terms it does not know (for example `hemostatic`, `scar_reduction`) were removed and a missing dosing interval was set to 24 h, so some briefs are not tested as written.
- **Rejections decide much of the order.** A candidate a filter rejects ranks by how late it was rejected. Decoys and winners are often rejected for the same reasons, so a low AUROC may reflect the filter thresholds and not the ranking model.
- **Decoys are not hard negatives.** Random and scrambled peptides are easy to tell apart from real ones by composition alone.
- **Training-data overlap is not flagged.** A well-known peptide ranking high may be memory, not recognition.

### Data audit
What was checked in the mined data, and what was not:

- **Sequences hold up.** All 20 named fragments of exendin-4, GLP-1 and GIP match their reference sequences exactly. Lengths agree with the sequences, and groups agree with evidence tiers except in `GCGR`.
- **Citations exist.** All 65 distinct DOIs resolve in Crossref and their years match. This does not show that a paper supports the value or label attached to it, and **no measured value has been checked against its source.**
- **One extraction error found.** `NN1359` has a different sequence in the `GLP1R` file (`[ImP][Aib]…`) and the `GCGR` file (`[Aib]…`). Other repeated peptides could be wrong in the same way.
- **`GCGR` tiers are inconsistent.** 28 inactive (group C) peptides have tier 1 or 2, where the other files use tier 0.
- **Some winners rest on thin evidence.** In `TC-02`, 9 of 18 winners have no numeric value. In `TC-01`, two winners (Ac2-26, ACT1) have no in-vivo or wound-type readout; ACT1 may be recorded under a differently named clinical readout, which was not checked.
- **"Winner" means "reported to help", not "should pass our filters".** A peptide whose benefit is of a different kind than the brief's functions can fail a function gate fairly.

### Filters that rejected winners
Share of the candidates that reached each stage and were rejected by that check. A candidate can fail several checks, and sample sizes are small, so treat these as indications.

| Check | Set | Winners rejected | Decoys rejected |
|---|---|---|---|
| Stage 8 cytotoxicity | GLP1R | 13 of 15 (87%) | 63 of 126 (50%) |
| | TC-02 | 7 of 12 (58%) | 21 of 46 (46%) |
| | TC-01 | 2 of 5 (40%) | 21 of 68 (31%) |
| Stage 8 hemolysis | TC-01 | 3 of 5 (60%) | 31 of 68 (46%) |
| | TC-02 | 4 of 12 (33%) | 9 of 46 (20%) |
| | GLP1R | 10 of 15 (67%) | 107 of 126 (85%) |
| Stage 6 anti-inflammatory | TC-01 | 4 of 9 (44%) | 49 of 117 (42%) |
| Stage 6 migration | TC-02 | 6 of 18 (33%) | 87 of 133 (65%) |
| Stage 8 cleavage stability | TC-01 | 0 of 5 | 24 of 68 (35%) |
| | TC-02 | 2 of 12 (17%) | 16 of 46 (35%) |
| Stage 8 half-life | GLP1R | 0 of 15 | 91 of 126 (72%) |

- **Cytotoxicity** rejected winners more often than decoys in every set, and **hemolysis** did in both wound sets. A likely, untested reason is that these models learned that membrane-active, cationic peptides are harmful, which also describes many antimicrobial winners.
- **The Stage 6 anti-inflammatory gate** removed winners at the same rate as decoys on `TC-01`, so it did not help separate them.
- **Cleavage stability, half-life and the migration gate** removed decoys more often than winners, as intended.
- In `GLP1R`, Stage 8 rejected 152 of 162 candidates (hemolysis 131, cytotoxicity 93, half-life 93), so no known peptide reached the ranking.
- `TC-30`, `GIPR` and `GCGR` are not included because they had not finished.

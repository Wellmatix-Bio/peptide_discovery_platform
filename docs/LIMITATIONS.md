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

---

## The metabolic models: one question answered, two still open

All of these are about `functional_potency_predictor_v1` and `agonism_classifier_v1`, which share
one training file: `all_sequences_aligned.csv`. That file is **not in this repository and must not
be** — it is the models' training labels, and `.gitignore` names it explicitly. A copy was
supplied on 2026-11-03 and read; what follows is what it settled and what it did not.

### 1. GIPR potency is reported but cannot rank

From `model_store/functional_potency_predictor_v1/model_card.json`, leave-one-cluster-out
(CD-HIT 90% identity):

| Receptor | n | R² | Pearson | Spearman |
|---|---|---|---|---|
| GLP1R | 271 | 0.511 | 0.722 | 0.584 |
| GCGR | 199 | 0.470 | 0.688 | 0.678 |
| **GIPR** | **55** | **−0.108** | **0.094** | **0.144** |

A negative R² means the model does worse than always predicting the training mean, so **its
ordering of two candidates carries no information**. 55 labels against 271.

What the platform does with that, today: the value is still returned, and it is **flagged** at
every layer — `UNRELIABLE_POTENCY_RECEPTORS` on the predictor, `potency_unreliable_receptors` plus
a note on each candidate in the API response, a warning in the submit form *before* a GPU job is
paid for, and a pill on the results page. The flag reads the way `not_screened` does elsewhere: a
thing that did not work reports itself rather than passing quietly.

**What is still a product decision, and is deliberately left open:** the GIPR value *still counts
toward the stage-6 gates and the stage-11 ranking score*. Nothing excludes it. Whether a run that
targets GIPR should rank on a signal with R² = −0.108 is not a question the code should answer by
itself, and the note says so outright rather than letting the flag read as "this was excluded".

The honest options are: exclude GIPR from the ranking weights; keep it and accept that the order
is partly noise; or get more GIPR labels. The third is the only one that fixes it.

### 2. The receptor column order — ANSWERED, and it was right

Both cards recorded `EC50_T1 = GCGR, EC50_T2 = GLP1R, EC50_T3 = GIPR` as an assumption nobody had
checked. It has now been checked against the file itself, and it holds.

The CSV contains the natural ligands, whose receptor selectivity is not in question. Each is most
potent in exactly the expected column, by seven to eight orders of magnitude — this is not a
judgement call:

| ligand | T1 (pM) | T2 (pM) | T3 (pM) | most potent |
|---|---|---|---|---|
| glucagon | **0.5** | 43.9 | 1e8 | T1 → **GCGR** |
| GLP-1 | 1e8 | **0.9** | 1e8 | T2 → **GLP1R** |
| exendin-4 | 1e8 | **0.4** | 2e5 | T2 → **GLP1R** |
| liraglutide | 1e8 | **6.4** | 1e8 | T2 → **GLP1R** |
| semaglutide | 1e8 | **6.2** | 1e8 | T2 → **GLP1R** |
| GIP | 1e8 | 2e5 | **0.4** | T3 → **GIPR** |

Six ligands, unanimous. Both model cards now say `CONFIRMED` with the date and the evidence, and
`tests/unit/test_receptor_column_assumption.py` keeps them from drifting apart or losing the
confirmation. **Nothing needs re-training on this account**, which was the risk worth checking:
had the order been wrong, every pEC50 and every agonism probability would have been attached to
the wrong receptor with no symptom at all.

### 3. The source column is PICOMOLAR, and neither card said so

`EC50_LOG_T*` is log10 of the EC50 **in molar**; the bare `EC50_T*` column is in **picomolar**.
Checked across all 496 paired values: assuming picomolar matches `EC50_LOG_T*` to a mean absolute
error of **0.0013**, and assuming nanomolar is out by exactly **3.0000** — three log units, a
1000× error in potency.

Nothing in this repository reads the CSV, and `pec50_to_ec50_nm()` in
`src/pipeline/s06_functional_models/metabolic_stage.py` is correct (`10 ** (9 - pEC50)`), so no
shipped code is affected. **The risk is to the next person who retrains.** Train from
`EC50_LOG_T*`, which is already log10 molar, or scale the raw column by `1e-12`. Both cards now
record this.

**For the model owner:** it would be worth confirming that the published metrics were produced
from `EC50_LOG_T*` and not from the raw column, because a uniform 3-log shift in the labels would
not change R² or Pearson at all — those are shift-invariant — but it *would* make every absolute
pEC50 the pipeline reports, and therefore every EC50 gate in stages 6 and 11, wrong by 1000×.
The metrics cannot tell you which happened.

### 4. Censored values are mixed in with measurements

The columns carry sentinel "no measurable activity" values — `1e8`, `1e7` and `1e6` pM, which land
on suspiciously round pEC50 values of exactly 4.0, 5.0 and 6.0:

| receptor | labels | censored (pEC50 ≤ 6) | at |
|---|---|---|---|
| GCGR | 206 | 10 (5%) | 4.0, 5.0, 5.24 |
| GLP1R | 234 | 10 (4%) | 6.0 |
| **GIPR** | **56** | **4 (7%)** | **4.0** |

These are **ceilings, not measurements**: a plate that showed nothing, recorded as a number. A
regression trained on them treats "we could not measure this" as "the value is exactly 4.0".

This is worth putting to the model owner alongside question 1, because the two may be the same
problem. GIPR's labels have mean pEC50 9.13 and standard deviation 1.86, and **4 of its 56 points
sit at 4.0 — nearly three standard deviations below the mean**. Under leave-one-cluster-out, a
fold holding some of those four would be asked to predict a value no feature can explain, and a
handful of such points is enough to drive R² negative on n=56. That would make GIPR's −0.108 partly
an artefact of how non-binders were encoded, not only a shortage of data — and censored regression,
or dropping the sentinels, would be a cheaper experiment than collecting more GIPR labels.

**It is a hypothesis, not a finding.** Confirming it means re-running the leave-one-cluster-out CV
with those points excluded and comparing, which needs the training notebook.

### Label counts, for reference

The CSV holds 234 rows. Per-column label counts are GCGR 206, GLP1R 234, GIPR 56, against the
cards' recorded `n_train` of 199, 271 and 55. GLP1R being *higher* in the cards is expected — they
name a ChEMBL GLP-1R extract as a second source. The small shortfalls for GCGR and GIPR are
consistent with the stated 5–50 length filter and with replicate merging, but they have not been
reconciled row by row.

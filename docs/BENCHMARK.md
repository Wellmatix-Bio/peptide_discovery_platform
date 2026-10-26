# Benchmark

Does the platform rank peptides with published evidence of working above toxic or inactive peptides and random ones? The data, scripts and results live outside version control (`data/raw/benchmark/`, `scripts/benchmark/`, `artifacts/benchmark/`).

## Data
Peptides were **extracted from the literature by an AI model** (prompts in `data/raw/benchmark/brief_prompts.json`) and not verified by hand, one file per brief: wound briefs `TC-01`, `TC-02`, `TC-30` and metabolic receptors `GLP1R`, `GIPR`, `GCGR`. Each peptide has a group: **A** works, **B** toxic or harmful, **C** inactive or failed (including truncated and scrambled variants). Peptides with modifications the models cannot read, or two conflicting labels for the same sequence, are left out of the main analysis.

Each set also gets 100 random peptides (natural amino-acid frequencies, lengths matched to the set) and 2 scrambled copies of every winner.

## Method
Every candidate runs through the real screening stages (5, 6, 7, 8, 9, 11) for that set's brief. Stage 4 is skipped because the peptides are given. A candidate a stage rejects ranks below every survivor, and a later rejection ranks above an earlier one. Survivors rank by Stage 11 `final_score`.

Reported per set: winners in the top 5% and 10% against chance, AUROC with a bootstrap interval, where the toxic peptides fall, and the stage that rejected each known peptide. For metabolic sets, a second table adds approximated sequences (modified residues replaced by close standard ones).

Run it in `scripts/benchmark/benchmark.ipynb`, one set at a time; each stage saves its output and resumes if stopped.

## Results: TC-01 (infected diabetic foot ulcer)
This report covers **TC-01 only**: 9 winners among 132 candidates. It says nothing about the other briefs. Results for the other sets will be added here as each is run.

| Measure | Result |
|---|---|
| Winners in top 5% / top 10% | 1 / 1 (about 0.5 / 0.9 expected by chance) |
| AUROC | 0.51 (95% interval 0.32 to 0.71) |
| Candidates reaching Stage 11 | 21 of 132 |

Eight of the nine winners were rejected by a filter: 4 in Stage 6, 3 in Stage 8 and 1 in Stage 9. All 18 scrambled copies of winners were rejected too. The ranking therefore has little to work with, and on this brief it does not separate winners from decoys. The Stage 6 and Stage 8 cut-offs are the first thing to examine.

Not yet run: `TC-02`, `TC-30`, `GLP1R`, `GIPR` and `GCGR`.

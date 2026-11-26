# Protein-family clustering: validation report

What was checked: `src/pipeline/food/protein_library/clusters.py`, which gives each protein a cluster (UniProt family, else MMseqs2 through `pymmseqs` 1.2.0 at 70% identity, else one cluster per accession; it was 90% and CD-HIT was an option when this was checked, see "Decision") so the ranking can cap peptides per family.

## Verdict
1. **MMseqs2 has not been executed by the platform code yet.** The platform now clusters with `pymmseqs` 1.2.0 (MMseqs2 inside). pymmseqs has Linux and macOS builds only, so it cannot run on this Windows machine. A test that runs it for real is in the suite and runs in Linux CI. See "Not done".
2. **A 90% identity setting does not reproduce protein families.** It behaves as a near-duplicate filter. Against the data foundation's 21 curated families it scores an adjusted Rand index of 0.167. At 70% it scores 0.855. **Decided: the platform now uses 70%.**

Reproduce: `python scripts/food_validation/clustering.py` (writes `clustering_validation.json` here).

## Data
The data foundation (`datasets/Food-Derived Metabolic Wellness Peptide/evaluation/`) clustered its 524 digestion substrates with real CD-HIT at 100, 95, 90, 80, 70, 60 and 50% identity, and classified each substrate into one of 21 curated protein families by rule (`family_rules.py`; for example "Wheat: LMW Glutenins", "Casein: αS1-Casein"). Both are used here as the reference. The 524 substrates are Whey 70, Casein 23, Soy 57, Pea 60, Wheat 314 (the Phase 1 fixture).

## Result 1: the reference tables are read correctly
The seven real CD-HIT `.clstr` files of the data foundation were parsed by the reader in `scripts/food_validation/clustering.py`. At every level the partition equals the table's cluster column (same clusters, same members; 486, 222, 160, 92, 63, 46, 41 clusters). This validates how the reference is read. The platform itself no longer uses CD-HIT, so there is no platform reader for CD-HIT output to test.

## Result 2: how well each identity level reproduces the curated families

| Identity | Clusters | Homogeneity | Completeness | Adjusted Rand index | Families kept in one cluster | Most clusters one family spans | Families that could fill all 25 slots |
|---|---|---|---|---|---|---|---|
| 100% | 486 | 0.998 | 0.380 | 0.003 | 0 of 21 | 165 | 12 |
| 95% | 222 | 0.993 | 0.469 | 0.106 | 4 | 46 | 8 |
| 90% (was) | 160 | **0.989** | **0.510** | **0.167** | **5** | **33** | **6** |
| 80% | 92 | 0.986 | 0.599 | 0.342 | 6 | 18 | 2 |
| **70% (now)** | **63** | **0.984** | **0.742** | **0.855** | **10** | **12** | **1** |
| 60% | 46 | 0.951 | 0.774 | 0.872 | 10 | 7 | 0 |
| 50% | 41 | 0.928 | 0.783 | 0.870 | 10 | 8 | 0 |

- Homogeneity says clusters rarely mix families at any level. Completeness says whether a family stays together; it is low at 90%, meaning one family is spread over many clusters.
- "Could fill all 25 slots" counts families that span at least 9 clusters, since the ranking allows 3 peptides per cluster (3 × 9 = 27 ≥ 25).

Concrete cases, clusters per family:

| Family (substrates) | 90% | 70% | 60% |
|---|---|---|---|
| Wheat: LMW glutenins (177) | 24 | 3 | 2 |
| Whey: immunoglobulins (56) | 33 | 12 | 7 |
| Wheat: HMW glutenins (46) | 23 | 7 | 3 |
| Soy: glycinin 11S (30) | 15 | 6 | 7 |

**Consequence:** with the 90% setting, wheat LMW glutenins span 24 clusters, so the cap allows up to 72 peptides from that one family, more than the 25 final slots. The cap the ranking applies stops near-identical sequences filling the list; it does not stop one family filling it. At 70% the same family can contribute at most 9.

## Decision
The intent stated for the cap is "at most 3 peptides per parent **protein family**". That points to a family-level identity, not 90%. The data say:
- 70% to 50% all reproduce the curated families well (adjusted Rand 0.855 to 0.872), and 60% and 50% leave no family able to fill the slots on its own.
- Lower identity gives a lower homogeneity (0.951 at 60%, 0.928 at 50%): some unrelated proteins start to share a cluster.
- 70% is the highest identity that clearly recovers families, so it keeps the most separation between unrelated proteins.

**`IDENTITY` is now 0.7** in `clusters.py` (it was 0.9, copied from the Phase 1 `c90` column). With 70%, a family such as wheat LMW glutenins spans 3 clusters, so the cap lets it contribute at most 9 of the 25 final peptides, not all of them.
To revisit: 60% gives a slightly higher adjusted Rand index (0.872) and no family that can fill the list, at the price of more mixing of unrelated proteins (homogeneity 0.951).

## Not done: running MMseqs2 for real
- `cluster_by_sequence` calls `pymmseqs.commands.easy_cluster(fasta, prefix, tmp, min_seq_id=0.7)` and reads `<prefix>_cluster.tsv`. It has been run only against stand-ins that write that file. pymmseqs has no Windows build, so it cannot run on this machine.
- `tests/unit/pipeline/test_food_clusters.py::test_real_mmseqs2_groups_a_close_variant_with_its_parent_and_leaves_an_unrelated_protein_alone` runs it for real (a 98% identical variant must join its parent, an unrelated protein must not). It is skipped without pymmseqs and runs in CI on Linux, where `requirements.txt` installs pymmseqs 1.2.0. **Its first CI result is the first real check.**
- After that, run `cluster_by_sequence` on the 524 substrates at 70% identity on Linux and compare with the CD-HIT tables above.
- Only the identity is set; coverage, sensitivity and the rest stay at MMseqs2's defaults (`-c 0.8`, coverage computed on both sequences). Because of the 80% coverage rule, a short protein and a much longer relative may not cluster even at high identity. That has not been measured.
- Until then, with pymmseqs missing (any Windows machine) every protein falls back to its own cluster, which is what the dashboard and the 15-protein runs used.

## Other limits
- The curated families come from rules written by the data foundation, not an independent classification; agreement with them shows the clusters are family-like, not that the families are right.
- The 524 substrates cover five food sources; user-supplied proteins can sit outside them.
- The UniProt-family step has no data in a dashboard run: it reads the source records' `protein_family` column, which the dashboard does not send, so it is untested end to end.

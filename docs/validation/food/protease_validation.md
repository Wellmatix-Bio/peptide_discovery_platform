# Protease rules: validation report

What was validated: the Expasy PeptideCutter rules in `src/pipeline/food/digestion/rules.py`, which s03 (digestion) and s06 (gut stability) use. Digestion uses `pepsin_ph1.3`, `trypsin` and `chymotrypsin_high`. Gut survival (s06) uses `pepsin_ph1.3`, `trypsin` and, since this report, `chymotrypsin_low`. The report gives sensitivity, specificity and a balanced accuracy for each rule against experimentally observed cleavage sites, a verdict against the bar Phase 1 set, and the limits that apply when the rules are used on food peptides.

Reproduce: `python scripts/food_validation/protease_rules.py` (writes `protease_validation.json` here). The data are Phase 1's `Model Validation` inputs, read in place.

## Verdict

| Protease (rule used) | Verdict | Why |
|---|---|---|
| Trypsin (`trypsin`) | **Validated** | Balanced accuracy 0.936, lower 95% bound 0.933, on 7,525 sites from 1,441 proteins. |
| Chymotrypsin, gut survival (`chymotrypsin_low`) | **Validated** | Balanced accuracy 0.897, lower bound 0.884, on 150 sites from 82 proteins. It finds 97% of observed cuts. |
| Chymotrypsin, digestion (`chymotrypsin_high`) | **Not validated** | Balanced accuracy 0.780, lower bound 0.737 (< 0.80). It misses 38% of observed sites. |
| Pepsin (`pepsin_ph1.3`) | **Not validated** | Balanced accuracy 0.711, lower bound 0.700 on 663 yeast proteins. MEROPS has too little pepsin data to test (4 proteins). `pepsin_ph2.0` is 0.739, also below the bar. |

The bar is Phase 1's (agreed 2026-11-23): lower 95% bound of the AUROC at or above 0.80 on unseen proteins, with at least 100 sites from at least 50 proteins. A yes/no rule has a single operating point, and the AUROC of a yes/no call equals its balanced accuracy, so balanced accuracy is what is compared with 0.80. One further consequence for the pipeline is described under "Limits": short peptides get an optimistic pepsin result.

## Results

MEROPS observed cleavage sites (mass spectrometry and N-terminal sequencing). "Strict" is the set Phase 1 used for UniZyme (proteins UniZyme never trained on); "all" is every site for the protease, which is also valid here because the rules are not fitted to anything. Every site was checked against its P4 to P4′ window in the parent sequence: all matched, none dropped.

| Protease | Rule | Set | Proteins | Sites | Sensitivity | Specificity | Precision | Balanced accuracy (95% CI) | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Trypsin 1 | `trypsin` | strict | 1,441 | 7,525 | 0.979 | 0.892 | 0.058 | **0.936** (0.933-0.938) | 0.225 |
| Trypsin 1 | `trypsin` | all | 1,673 | 8,153 | 0.978 | 0.891 | 0.058 | 0.934 (0.932-0.937) | 0.224 |
| Chymotrypsin A | `chymotrypsin_high` | strict | 82 | 150 | 0.620 | 0.940 | 0.032 | **0.780** (0.737-0.819) | 0.131 |
| Chymotrypsin A | `chymotrypsin_low` | strict | 82 | 150 | 0.973 | 0.820 | 0.017 | 0.897 (0.884-0.910) | 0.115 |
| Chymotrypsin A | `chymotrypsin_high` | all | 249 | 571 | 0.606 | 0.936 | 0.048 | 0.771 (0.750-0.795) | 0.158 |
| Chymotrypsin A | `chymotrypsin_low` | all | 249 | 571 | 0.948 | 0.820 | 0.027 | 0.884 (0.874-0.893) | 0.144 |
| Pepsin A | `pepsin_ph1.3` | strict | 4 | 74 | 0.365 | 0.881 | 0.037 | 0.623 (0.596-0.692) | 0.083 |
| Pepsin A | `pepsin_ph2.0` | strict | 4 | 74 | 0.513 | 0.861 | 0.044 | 0.687 (0.657-0.845) | 0.118 |

Pepsin on porcine-pepsin peptidomics of yeast proteins (Baghalabadi et al., Proteomes 2021), the data Phase 1 built. "Primary" scores a site seen in at least two experiments against the interiors of recovered peptides; "secondary" scores any observed site against every other residue.

| Rule | Labels | Proteins | Sites | Sensitivity | Specificity | Precision | Balanced accuracy (95% CI) | MCC |
|---|---|---|---|---|---|---|---|---|
| `pepsin_ph1.3` | primary | 663 | 2,741 | 0.565 | 0.857 | 0.274 | **0.711** (0.700-0.723) | 0.310 |
| `pepsin_ph2.0` | primary | 663 | 2,741 | 0.658 | 0.820 | 0.259 | 0.739 (0.728-0.751) | 0.325 |
| `pepsin_ph1.3` | secondary | 667 | 4,840 | 0.522 | 0.815 | 0.041 | 0.668 (0.659-0.678) | 0.103 |
| `pepsin_ph2.0` | secondary | 667 | 4,840 | 0.610 | 0.763 | 0.037 | 0.687 (0.677-0.696) | 0.105 |

### What the numbers say
- **Trypsin:** the rule finds 98% of observed sites and calls 11% of all bonds, about 17 times the 0.7% of bonds that are observed sites. The low precision (0.058) is mostly the labels: only observed cuts are listed, so many of the rule's "false positives" are untested real sites.
- **Chymotrypsin:** the choice of rule matters more than anything else. `chymotrypsin_high` (F, Y, W) calls few bonds and misses 38% of sites; `chymotrypsin_low` (adds L, M, H) finds 97% at the cost of calling 18% of bonds instead of 6%. In the 150 strict chymotrypsin sites, 62% follow F, Y or W and 35% follow L or M (51 follow L); only the broader rule covers the L and M cuts, which is exactly the 38% the narrow rule misses. Gut survival uses the broad rule; digestion keeps the narrow one.
- **Pepsin:** the rule is only moderately informative. On yeast it finds 57% of primary sites; on MEROPS it finds 37% but that set is four proteins (Phase 1 notes that 67 of the 78 held-out pepsin sites sit on one protein) and is not evidence either way.
- **Against UniZyme (Phase 1's numbers, not re-run):** UniZyme scored AUROC 0.886 (trypsin), 0.560 (chymotrypsin) and 0.576 (pepsin, primary). The rules score higher on all three, but the comparison is loose: UniZyme was scored on the subset with a usable structure and under 1,500 residues, and with a graded score rather than a yes/no call.

## Limits

1. **Yes/no, not a probability.** A bond is cut or not. No rule says how readily, so `uncut_bond_fraction` (the share of bonds no protease cuts; formerly `gut_survival`, renamed because it does not measure survival in the gut) counts a fast cut and a slow cut equally and ignores enzyme amount, time and competition between cuts.
2. **Labels are observed cuts only.** An unlisted residue may simply be untested, so precision and MCC are understated and should not be quoted as the rule's accuracy. Sensitivity and specificity are the more reliable figures.
3. **Peptide-sized substrates are untested.** No parent of 40 residues or fewer exists in the held-out sets, and the food peptides are 4 to 40 residues. The benchmark proteins are large and folded; peptides are not, so a cut that is blocked in a folded protein may happen in a peptide. Precision could differ in either direction.
4. **Short peptides get an optimistic pepsin result.** The pepsin rules look at two residues before and two after the bond, so a bond near an end cannot be cut. The most bonds pepsin can cut, in a sequence chosen to favour it, are:

   | Peptide length | 4 | 6 | 8 | 10 | 15 | 20 | 40 |
   |---|---|---|---|---|---|---|---|
   | Pepsin can cut (of internal bonds) | 0 of 3 | 2 of 5 | 4 of 7 | 6 of 9 | 11 of 14 | 16 of 19 | 36 of 39 |

   Trypsin and chymotrypsin can cut every internal bond at any length. So a short peptide's `uncut_bond_fraction` is higher than the gastric cut would justify, which favours short peptides in the ranking.
5. **pH.** The yeast data's digestion pH is not identified, so neither pepsin rule can be matched to its conditions; the pH 2.0 rule fits it slightly better (0.739 against 0.711). The pipeline uses the pH 1.3 rule for the gastric stage.
6. **Chymotrypsin not validated.** The rule in use is the narrow one; see the table.
7. **Only three enzymes.** Brush-border peptidases, which act on peptides in the small intestine, are not modelled, so survival through the full gut is not assessed.
8. **Independence.** MEROPS records and paralogous proteins are not independent; the intervals resample proteins but cannot fully account for it. The Expasy rules and MEROPS may draw on the same literature, so the benchmark is not shown to be independent of the rules' sources.

## Implications for the pipeline
These are findings, not changes made:
- The applicability of the stability dimension stays "limited" (already so); the data support nothing stronger.
- **Decided:** gut survival uses `chymotrypsin_low`, which is above the bar; calling more cuts is the cautious direction for a survival score. The rule marks 18% of bonds as cut against 6% for the narrow rule, so survival values are lower than before and the ranking can change.
- Digestion (s03) still uses `chymotrypsin_high`, which is below the bar. It sets which fragments exist, so the narrow rule can miss fragments that a cut after L or M would release. Moving s03 to the broad rule is the open choice; it would give more fragments.
- Short peptides are favoured by the pepsin blind spot. A fix needs a decision (for example scoring only bonds a rule can reach, or flagging peptides under about 8 residues).
- Pepsin remains unvalidated and should keep being reported that way.

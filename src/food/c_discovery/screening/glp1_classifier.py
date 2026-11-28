"""GLP-1 secretion classifier status: reported next to every ranked peptide, never used in the ranking.

Not validated. Under homology-aware cross-validation the ESM2 + logistic regression model reaches AUROC about 0.68, which is not clearly above length alone (0.65),
far below source alone (0.90), and falls to 0.45 when whole studies are held out (Phase 1 `4. Evidence Database/Evidence_Database_Report.md`, section 5).
The model itself is not ported: it fails its gate, so only the status is carried.
"""

STATUS = "Not validated"
REASON = ("fails the baseline gate: not clearly above length-only, far below source-only, "
          "chance level when whole studies are held out (Evidence Database report, section 5)")

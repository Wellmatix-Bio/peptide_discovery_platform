# Food GLP-1 evidence database: audit

Question: does the 81-peptide evidence set carry assay conditions, controls, negative results and study provenance well enough to be an auditable evidence database?

**Answer: no, not yet.** The peptide table is internally consistent and every peptide traces to a study, but assay conditions are recorded for 41% of observations, controls for 38% (vehicle controls only, no positive controls), and every negative is a single-study, single-dose null. It supports evidence lookup and prioritisation, as Phase 1 already stated. It does not support a claim that a peptide is inactive, or a comparison of potency across studies.

Reproduce: `python scripts/food_validation/evidence_audit.py` (reads `labeled_peptides_v1.csv` and the source workbook `GLP-1 Secretion_merged&labeled.xlsx`; writes `evidence_audit.json`, `evidence_audit_peptides.csv`, `evidence_observations.csv` here). Nothing was corrected or filled in.

## What was audited
- Shipped table: `src/pipeline/food/evidence/data/labeled_peptides_v1.csv`, 81 peptides (51 positive, 30 negative), peptide-level.
- Source: the workbook's Merged sheet, 120 labelled assay observations (68 positive, 52 negative) covering the same 81 peptides, plus the ChatGPT-mining sheet (45 rows) that holds the fuller assay record.

## Findings

### 1. Consistency of the shipped table with its source: good
- All 81 peptides are in both; none missing either way.
- The direction (positive / negative / mixed) recomputed from the source rows matches the shipped table for every peptide.

### 2. Assay conditions: partial
Of the 120 labelled observations:

| Field | Present | Share |
|---|---|---|
| Cell model | 105 | 87.5% |
| Concentration | 51 | 42.5% |
| Exposure time | 49 | 40.8% |
| Cell model, concentration and exposure all present | 49 | 40.8% |

By source sheet: all 45 ChatGPT-mining rows are complete; the 15 BIOPEP rows have none of the three; the 27 Paper-Agent rows have a cell model only; of the 33 "Purity verified" rows, 6 have a concentration and 4 an exposure time.
The shipped peptide table keeps none of dose, time or endpoint, so a user of the table cannot see this at all. Only the study's cell model names are carried.

### 3. Controls: partial, and no positive controls
- A control is recorded for 45 of 120 observations (37.5%): 20 buffer, 12 untreated cells in matched medium, 7 assay buffer, 6 HEPES buffer. All are vehicle controls.
- **The workbook has no positive-control field.** Whether an assay responded to a known stimulus is not recorded for any observation.
- Of those 45: replicate count 45, p-value 17, an uncertainty value 2, cell viability 0. The endpoint is typed (active GLP-1) for these 45 only; for the other 75 observations the active-versus-total endpoint is not recorded in the workbook.
- 75 observations (63%) carry no control information at all.

### 4. Negative results: present, but weak
- 30 peptides are negative only. All 30 rest on a single study. None was tested at two or more doses: 26 at exactly one recorded dose, 4 with no recorded dose.
- 19 of the 30 come from one study (PMID33396793). Only 7 have cell model, concentration and exposure recorded in every observation; 24 have a control recorded.
- The wording is mostly "no significant change" (46 of 52 negative observations). One null at one dose does not show a peptide is inactive.
- Consequence for the pipeline: a peptide matched to a negative here is "no effect observed in one assay", never "inactive". The screening already words it as "no match means no data, not inactive"; the same care applies to negative matches.

### 5. Positive results
- 51 positive peptides: 14 are database-curated only (BIOPEP, no assay record), 20 come from patents only. Only 5 have a control recorded and only 4 have complete conditions in every observation.

### 6. Study provenance: identifiers complete, fuller citation partial
- A study identifier is present for all 120 observations (84 paper PMIDs, 21 patent numbers, 15 BIOPEP database records). 30 independent studies.
- DOI is present for 72 of 120; 47 of the 81 peptides have no DOI in any observation. A source URL is present for 93.
- 80 of 81 peptides are supported by exactly one study identifier. The exception is LKPT (a BIOPEP record plus one paper), and the BIOPEP record carries no assay detail, so no peptide is independently replicated across two assay studies.
- One study supplies 39 of the 120 observations.

## What would make it auditable
These need the papers and patents, not a rule. Nothing below can be inferred.
1. One row per assay observation with every field above, and an explicit "not recorded" instead of an empty cell. `evidence_observations.csv` is that table for what exists today (120 rows, values copied as they are); it is not wired into the pipeline.
2. A positive-control field and the control's result, for every observation.
3. The GLP-1 endpoint (active or total) for the 75 observations that lack it.
4. A second dose or a second study for each negative before it is called a negative.
5. DOI or patent number plus the figure or table the value came from, for every row.
6. Cell viability, since a reduction in secretion can be toxicity.

## Limits of this audit
- It reads the workbook as given and cannot tell whether a value is correct, only whether it is present and consistent.
- Peptide sequences were not re-checked against the source papers.
- The 120 observations are the labelled subset; the workbook's other rows are unlabelled and were not audited.

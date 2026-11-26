"""Audit of the food GLP-1 evidence dataset: does it carry assay conditions, controls, negative results and study provenance?

Reads the shipped peptide table (src/pipeline/food/evidence/data/labeled_peptides_v1.csv) and the source workbook it was built from
(`GLP-1 Secretion_merged&labeled.xlsx`, sheets Merged and ChatGPT mining), recomputes every figure, and writes
`evidence_audit.json` and `evidence_audit_peptides.csv` to the output folder. Nothing is corrected or filled in: a missing field stays missing.

    python scripts/food_validation/evidence_audit.py [--workbook PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "src" / "pipeline" / "food" / "evidence" / "data" / "labeled_peptides_v1.csv"
WORKBOOK = Path(r"D:\AI Peptide Discovery Platform\Datasets\GLP-1 Secretion_merged&labeled.xlsx")

CONDITION_FIELDS = ["concentration", "exposure", "cell_model"]
PROVENANCE_FIELDS = ["study_id", "doi", "source_url"]


def study_kind(study_id: str) -> str:
    m = re.match(r"[A-Za-z]+", str(study_id))
    return {"PMID": "paper", "BIOPEP": "database"}.get(m.group(0) if m else "", "patent")


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def audit(workbook: Path) -> tuple[dict, pd.DataFrame]:
    shipped = pd.read_csv(SHIPPED)
    merged = pd.read_excel(workbook, sheet_name="Merged")
    rows = merged[merged.label.notna()].copy()
    rich = pd.read_excel(workbook, sheet_name="ChatGPT mining")
    rows["study_kind"] = rows.study_id.map(study_kind)
    rows["has_conditions"] = rows[CONDITION_FIELDS].notna().all(axis=1)
    out: dict = {"shipped_table": {"peptides": len(shipped), "columns": list(shipped.columns), "positive": int((shipped.label == 1).sum()), "negative": int((shipped.label == 0).sum())},
                 "source_rows": {"labeled_rows": len(rows), "unique_peptides": int(rows.sequence.nunique()), "positive_rows": int((rows.role == "P").sum()), "negative_rows": int((rows.role == "N").sum())}}

    # 1. Assay conditions per labeled row, by the sheet the row came from.
    cond = {}
    for sheet, g in rows.groupby("source_sheet"):
        cond[sheet] = {"rows": len(g), **{f: int(g[f].notna().sum()) for f in CONDITION_FIELDS + PROVENANCE_FIELDS[1:]}, "all_three_conditions": int(g.has_conditions.sum())}
    out["conditions_by_sheet"] = cond
    out["conditions_all_rows"] = {f: {"present": int(rows[f].notna().sum()), "pct": pct(int(rows[f].notna().sum()), len(rows))} for f in CONDITION_FIELDS}
    out["rows_with_all_three_conditions"] = {"n": int(rows.has_conditions.sum()), "pct": pct(int(rows.has_conditions.sum()), len(rows))}

    # 2. Controls, replicates and the rest of the assay record exist only for the rows of the ChatGPT mining sheet.
    mined = rows[rows.source_sheet == "ChatGPT mining"]
    out["controls"] = {
        "rows_with_a_control_recorded": int(len(rich)), "of_labeled_rows": len(rows), "pct_of_labeled_rows": pct(len(rich), len(rows)),
        "control_types": rich.control_type.value_counts().to_dict(),
        "replicate_count_present": int(rich.replicate_count.notna().sum()), "p_value_present": int(rich.p_value.notna().sum()),
        "uncertainty_present": int(rich.uncertainty_value.notna().sum()) if "uncertainty_value" in rich else None,
        "cell_viability_present": int(rich.cell_viability.notna().sum()), "glp1_type": rich.GLP1_type.value_counts().to_dict(),
        "workbook_has_a_positive_control_field": any(re.search(r"posit|forskolin", c, re.I) for c in rich.columns),  # co_treatments is the assay buffer, not a control
        "rows_without_any_control_information": len(rows) - len(mined),
    }

    # 3. Negative results: how strong is each negative peptide?
    peptide_rows = rows.groupby("sequence")
    per = pd.DataFrame({
        "sequence": list(peptide_rows.groups),
        "rows": peptide_rows.size().values,
        "pos_rows": peptide_rows.role.apply(lambda s: int((s == "P").sum())).values,
        "neg_rows": peptide_rows.role.apply(lambda s: int((s == "N").sum())).values,
        "studies": peptide_rows.study_id.apply(lambda s: ";".join(sorted(set(s)))).values,
        "n_studies": peptide_rows.study_id.nunique().values,
        "study_kinds": peptide_rows.study_kind.apply(lambda s: ";".join(sorted(set(s)))).values,
        "distinct_doses": peptide_rows.concentration.apply(lambda s: s.dropna().astype(str).nunique()).values,
        "rows_with_all_conditions": peptide_rows.has_conditions.sum().astype(int).values,
        "has_control_record": peptide_rows.source_sheet.apply(lambda s: bool((s == "ChatGPT mining").any())).values,
        "has_doi": peptide_rows.doi.apply(lambda s: bool(s.notna().any())).values,
    })
    per["label_in_shipped_table"] = per.sequence.map(shipped.set_index("sequence").label)
    per["direction_in_shipped_table"] = per.sequence.map(shipped.set_index("sequence").direction)
    neg = per[per.pos_rows.eq(0)]
    pos = per[per.pos_rows.gt(0)]
    out["negatives"] = {
        "negative_only_peptides": len(neg),
        "with_a_single_study": int((neg.n_studies == 1).sum()),
        "dose_not_recorded": int((neg.distinct_doses == 0).sum()),
        "tested_at_one_recorded_dose": int((neg.distinct_doses == 1).sum()),
        "tested_at_two_or_more_doses": int((neg.distinct_doses >= 2).sum()),
        "with_a_control_record": int(neg.has_control_record.sum()),
        "with_all_three_assay_conditions_in_every_row": int((neg.rows_with_all_conditions == neg.rows).sum()),
        "from_one_study": neg.studies.value_counts().head(3).to_dict(),
        "result_wording": rows[rows.role == "N"].result.value_counts().to_dict(),
    }
    out["positives"] = {
        "positive_peptides": len(pos), "with_a_control_record": int(pos.has_control_record.sum()),
        "database_curated_only": int((pos.study_kinds == "database").sum()), "patent_only": int((pos.study_kinds == "patent").sum()),
        "with_all_three_assay_conditions_in_every_row": int((pos.rows_with_all_conditions == pos.rows).sum()),
    }

    # 4. Provenance.
    out["provenance"] = {
        "study_id_present": int(rows.study_id.notna().sum()), "doi_present": int(rows.doi.notna().sum()), "source_url_present": int(rows.source_url.notna().sum()),
        "rows_by_study_kind": rows.study_kind.value_counts().to_dict(),
        "peptides_with_no_doi_in_any_row": int((~per.has_doi).sum()),
        "independent_studies": int(rows.study_id.nunique()),
        "largest_study": rows.study_id.value_counts().head(2).to_dict(),
        "peptides_supported_by_one_study_only": int((per.n_studies == 1).sum()),
    }

    # 5. Consistency of the shipped table with the source rows.
    merged_dir = per.assign(direction_from_rows=lambda d: d.apply(lambda r: "mixed" if r.pos_rows and r.neg_rows else "positive" if r.pos_rows else "negative", axis=1))
    mismatch = merged_dir[merged_dir.direction_from_rows != merged_dir.direction_in_shipped_table.fillna("missing")]
    out["consistency"] = {
        "peptides_in_source_not_in_shipped_table": int((~per.sequence.isin(shipped.sequence)).sum()),
        "peptides_in_shipped_table_not_in_source": int((~shipped.sequence.isin(per.sequence)).sum()),
        "direction_mismatches": mismatch[["sequence", "direction_from_rows", "direction_in_shipped_table"]].to_dict("records"),
        "shipped_table_has_fields_for": [c for c in shipped.columns if c in ("studies", "cell_models", "n_observations", "n_pos_obs", "n_neg_obs", "evidence_tier", "food_derivation", "direction")],
        "shipped_table_lacks": [f for f in ("dose", "exposure_time", "endpoint (active/total GLP-1)", "control type", "positive control", "replicate count", "effect size", "viability", "DOI or PMID per observation")
                                if not any(f.split()[0].lower() in c.lower() for c in shipped.columns)],
    }
    return out, per


OBSERVATION_COLUMNS = ["sequence", "label", "source_sheet", "study_id", "doi", "source_url", "cell_model", "concentration", "exposure", "glp1_type",
                       "assay_method", "control_type", "replicate_count", "p_value", "fold_change", "result"]


def observations(workbook: Path) -> pd.DataFrame:
    """One row per labeled assay observation, values copied from the workbook as they are; a field the source does not hold stays empty.

    The 45 rows of the ChatGPT mining sheet come from that sheet (it holds the control, replicate and assay fields); the other 75 come from the Merged sheet."""
    merged = pd.read_excel(workbook, sheet_name="Merged")
    rich = pd.read_excel(workbook, sheet_name="ChatGPT mining")
    rest = merged[merged.label.notna() & (merged.source_sheet != "ChatGPT mining")]
    a = pd.DataFrame({"sequence": rest.sequence, "label": rest.label.astype(int), "source_sheet": rest.source_sheet, "study_id": rest.study_id, "doi": rest.doi,
                      "source_url": rest.source_url, "cell_model": rest.cell_model, "concentration": rest.concentration, "exposure": rest.exposure, "result": rest.result})
    unit = lambda v, u: v.astype(str).str.replace(r"\.0$", "", regex=True) + " " + u.astype(str)  # noqa: E731
    b = pd.DataFrame({"sequence": rich.sequence, "label": rich.classification_label.astype(int), "source_sheet": "ChatGPT mining", "study_id": rich.study_id, "doi": rich.DOI,
                      "source_url": rich.source_URL, "cell_model": rich.cell_model, "concentration": unit(rich.peptide_concentration, rich.concentration_unit),
                      "exposure": unit(rich.exposure_time, rich.exposure_time_unit), "glp1_type": rich.GLP1_type, "assay_method": rich.assay_method,
                      "control_type": rich.control_type, "replicate_count": rich.replicate_count, "p_value": rich.p_value, "fold_change": rich.fold_change, "result": rich.outcome})
    return pd.concat([b, a], ignore_index=True).reindex(columns=OBSERVATION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "validation" / "food")
    args = parser.parse_args()
    result, per = audit(args.workbook)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence_audit.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    per.to_csv(args.out / "evidence_audit_peptides.csv", index=False)
    observations(args.workbook).to_csv(args.out / "evidence_observations.csv", index=False)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()

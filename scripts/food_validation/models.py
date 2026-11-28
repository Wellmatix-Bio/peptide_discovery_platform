"""Checks of the six screening models as the food pipeline uses them (src/food/c_discovery/screening/adapters.py), on real weights.

For each model: does the score point the right way (higher = safer / more developable), is the 0.5 label threshold the model's own boundary, and does the
applicability band match the training data? Plus, for allergenicity, the two score definitions (1 - p_allergen, Phase 1's hold-out percentile) compared on the
IEDB hold-out and on the ranking of Phase 1's 400-peptide pool.

Labelled data (Phase 1 / Models folders, read in place):
- allergenicity: the IEDB hold-out of `Models/Allergenicity/data/iedb_ige_peptides_splits.csv` (fold -1, eval_ok); NOTE the file post-dates the deployed weights, see the report;
- bitterness: `BTP-720-test.fasta` (in the deployed model's training data: a direction check, not a performance estimate);
- solubility: ultrapure-water rows of `long_peptide_solvent.csv` (training data: direction only);
- synthesis feasibility: `PepSySco.csv` (training data: direction only).
Cytotoxicity and aggregation have no clean labelled set here; they get named well-known peptides as an illustration, not a validation.

    DEV_MODE=true python scripts/food_validation/models.py        (needs the real model weights under model_store/model_weights)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pipeline.feature_extractor import FeatureExtractor  # noqa: E402
from food.c_discovery import settings  # noqa: E402
from food.c_discovery.screening import adapters  # noqa: E402

MODELS = ROOT.parent / "Models"
PHASE1_FIXTURE = ROOT / "tests" / "fixtures" / "food" / "phase1" / "screening_scored.csv"
AGGREGATION_DATA = Path("D:/AI Peptide Discovery Platform/Models/Aggregation/data/adt5111_data_files_s1_to_s7.xlsx")
SAMPLE_CAP = 1500  # rows per labelled set sent to the models


def separation(good: np.ndarray, bad: np.ndarray, threshold: float = 0.5) -> dict:
    """`good` and `bad` are adapter scores of peptides known to be in the safe/developable and the unsafe class. AUC says whether the score points the right way."""
    y = np.r_[np.ones(len(good)), np.zeros(len(bad))]
    s = np.r_[good, bad]
    pred_good = s > threshold  # the adapter's label: bad when P(bad) >= 0.5, i.e. score <= 0.5
    return {"n_good": len(good), "n_bad": len(bad), "auc_score_ranks_good_above_bad": round(float(roc_auc_score(y, s)), 4),
            "mean_score_good": round(float(good.mean()), 3), "mean_score_bad": round(float(bad.mean()), 3),
            "specificity_at_label_threshold": round(float((pred_good[y == 1]).mean()), 4), "sensitivity_at_label_threshold": round(float((~pred_good[y == 0]).mean()), 4),
            "mcc_at_label_threshold": round(float(matthews_corrcoef(y, pred_good)), 4)}


def scores(model, seqs: list[str]) -> dict[str, dict]:
    out: dict = {}
    for i in range(0, len(seqs), 256):
        out.update(model.predict(seqs[i:i + 256]))
    return out


def allergenicity(models: dict, fe) -> dict:
    splits = pd.read_csv(MODELS / "Allergenicity" / "data" / "iedb_ige_peptides_splits.csv")
    ho = splits[(splits.fold == -1) & splits.eval_ok].drop_duplicates("sequence")
    got = scores(models["allergenicity"], ho.sequence.tolist())
    ho = ho[ho.sequence.isin(got)]
    sc = np.array([got[s]["score"] for s in ho.sequence])
    y = ho.label.to_numpy()
    meta = json.loads((ROOT / "model_store" / "allergenicity_predictor_v1" / "model_card.json").read_text(encoding="utf-8"))
    weights = ROOT / "model_store" / "model_weights" / "allergenicity_predictor_v1"
    ref = np.sort(np.load(MODELS / "Allergenicity" / "models" / "deployment" / "holdout_reference_z.npy"))
    info = json.loads((weights / "ensemble.json").read_text(encoding="utf-8"))
    p = np.array([float(d.split("p_allergen=")[1].split(" ")[0]) for d in (got[s]["detail"] for s in ho.sequence)])  # P(allergen), as the adapter prints it
    z = np.log(p / (1 - p))  # the model's raw score: p = sigmoid(z)
    percentile_score = 1 - np.searchsorted(ref, z) / len(ref)  # Phase 1's definition
    bins = pd.cut(p, [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0], include_lowest=True)
    calib = pd.DataFrame({"bin": bins.astype(str), "y": y, "p": p}).groupby("bin", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), observed_rate=("y", "mean")).round(3)
    inband = ho.sequence.str.len().between(info["train_length"]["p5"], info["train_length"]["p95"]).to_numpy()
    return {
        "holdout_n": int(len(ho)), "holdout_positives": int(y.sum()), "card_holdout": {"evaluable": 1581, "AUC": meta["holdout_metrics"]["AUC"], "MCC": meta["holdout_metrics"]["MCC"]},
        "adapter_score_vs_label": separation(sc[y == 0], sc[y == 1]),
        "percentile_score_vs_label_auc": round(float(roc_auc_score(1 - y, percentile_score)), 4),
        "one_minus_p_vs_label_auc": round(float(roc_auc_score(1 - y, sc)), 4),
        "spearman_one_minus_p_vs_percentile": round(float(spearmanr(sc, percentile_score).statistic), 4),
        "ks_of_scores_vs_stored_reference": round(float(max(abs(np.searchsorted(np.sort(z), ref, side="right") / len(z) - np.arange(1, len(ref) + 1) / len(ref)))), 3),
        "weights_train_length": info["train_length"], "decision_threshold_in_weights": info["decision_threshold"],
        "auc_inside_7_20": round(float(roc_auc_score(1 - y[inband], sc[inband])), 4), "auc_outside_7_20": round(float(roc_auc_score(1 - y[~inband], sc[~inband])), 4),
        "n_inside_7_20": int(inband.sum()),
        "calibration_of_p_allergen": calib.reset_index().to_dict("records"),
        "score_summary": {"one_minus_p_quantiles_5_50_95": [round(float(v), 3) for v in np.percentile(sc, [5, 50, 95])],
                          "percentile_score_quantiles_5_50_95": [round(float(v), 3) for v in np.percentile(percentile_score, [5, 50, 95])]},
    }


def bitterness(models: dict) -> dict:
    seqs, labels, cur = [], [], None
    for line in (MODELS / "Bitterness Prediction" / "data" / "BTP-720-test.fasta").read_text().splitlines():
        if line.startswith(">"):
            cur = 1 if "Positive" in line else 0
        elif line.strip():
            seqs.append(line.strip())
            labels.append(cur)
    got = scores(models["bitterness"], list(dict.fromkeys(seqs)))
    y = np.array([l for s, l in zip(seqs, labels) if s in got])
    sc = np.array([got[s]["score"] for s, _ in zip(seqs, labels) if s in got])
    return {"note": "in the deployed model's training data: direction only", **separation(sc[y == 0], sc[y == 1])}


def solubility(models: dict) -> dict:
    d = pd.read_csv(MODELS / "Solubility" / "data" / "long_peptide_solvent.csv")
    d = d[d.solvent_name == "Ultrapure water"].drop_duplicates("sequence_clean")
    d = d[d.sequence_clean.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")]
    got = scores(models["solubility"], d.sequence_clean.tolist()[:SAMPLE_CAP])
    d = d[d.sequence_clean.isin(got)]
    sc = np.array([got[s]["score"] for s in d.sequence_clean])
    return {"note": "in the model's training data: direction only; label y: 1 = soluble", **separation(sc[d.y == 1], sc[d.y == 0])}


def synthesis(models: dict) -> dict:
    d = pd.read_csv(MODELS / "Sythesis feasibility" / "data" / "PepSySco.csv").drop_duplicates("peptide")
    d = d[d.peptide.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")]
    got = scores(models["synthesis_feasibility"], d.peptide.tolist()[:SAMPLE_CAP * 2])
    d = d[d.peptide.isin(got)]
    sc = np.array([got[s]["score"] for s in d.peptide])
    return {"note": "in the model's training data: direction only; label 1 = feasible", **separation(sc[d.label == 1], sc[d.label == 0])}


def aggregation(models: dict) -> dict:
    """The model's own held-out split (`train_test == test`, 10,306 peptides): seed_bh = 1 means aggregation-prone, the bad class."""
    d = pd.read_excel(AGGREGATION_DATA, sheet_name="SD_2")
    test = d[d.train_test == "test"].drop_duplicates("aa_seq")
    test = test[test.aa_seq.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+") & (test.aa_seq.str.len() >= 5)].sample(n=min(SAMPLE_CAP, len(test)), random_state=0)
    got = scores(models["aggregation"], test.aa_seq.tolist())
    test = test[test.aa_seq.isin(got)]
    sc = np.array([got[q]["score"] for q in test.aa_seq])
    return {"note": "held-out split of the model's own data (sample)", "split_sizes": d.train_test.value_counts().to_dict(), "positive_rate_test": round(float(test.seed_bh.mean()), 3),
            **separation(sc[test.seed_bh == 0], sc[test.seed_bh == 1])}


def training_lengths() -> dict:
    """5th, 50th, 95th percentile and range of the peptide lengths each model's training data holds (where that data is available here)."""
    q = lambda lengths: {"min": int(np.min(lengths)), "p5": float(np.percentile(lengths, 5)), "median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)), "max": int(np.max(lengths)), "n": int(len(lengths))}  # noqa: E731
    out = {"allergenicity": json.loads((ROOT / "model_store" / "model_weights" / "allergenicity_predictor_v1" / "ensemble.json").read_text())["train_length"]}
    d = pd.read_excel(AGGREGATION_DATA, sheet_name="SD_2")
    out["aggregation"] = q(d[d.train_test == "train"].aa_seq.dropna().str.len())
    sol = pd.read_csv(MODELS / "Solubility" / "data" / "long_peptide_solvent.csv").drop_duplicates("sequence_clean")
    out["solubility"] = q(sol.sequence_clean.str.len())
    out["synthesis_feasibility_PepSySco_only"] = q(pd.read_csv(MODELS / "Sythesis feasibility" / "data" / "PepSySco.csv").peptide.str.len())
    bit = pd.read_csv(MODELS / "Bitterness Prediction" / "data" / "BTP-720-merged.csv")
    col = next((c for c in bit.columns if c.lower() in ("sequence", "peptide", "seq")), None)
    out["bitterness_BTP720_only"] = q(bit[col].str.len()) if col else "no sequence column"
    return out


def illustrations(models: dict) -> dict:
    cyto = {"melittin (cytotoxic, haemolytic)": "GIGAVLKVLTTGLPALISWIKRKRQQ", "LL-37 (cytotoxic at high dose)": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
            "carnosine-like GHK": "GHK", "tuftsin TKPR": "TKPR", "KDEL": "KDEL", "SAGPQGPIGPR (collagen fragment)": "SAGPQGPIGPR"}
    agg = {"Abeta16-22 KLVFFAE (forms amyloid)": "KLVFFAE", "GNNQQNY (Sup35 amyloid core)": "GNNQQNY", "VQIVYK (tau amyloid core)": "VQIVYK",
           "KKKKKKK": "KKKKKKK", "EEEEEEE": "EEEEEEE", "GSGSGSGS": "GSGSGSGS"}
    out = {}
    for name, panel in (("cytotoxicity", cyto), ("aggregation", agg)):
        got = models[name].predict(list(panel.values()))
        out[name] = {label: (round(got[s]["score"], 3) if s in got else None) for label, s in panel.items()}
    return out


def applicability_coverage() -> dict:
    d = pd.read_csv(PHASE1_FIXTURE)
    cols = [c for c in d.columns if c.startswith("applicability_") and c.split("applicability_")[1] in {n for n, *_ in adapters.SLOTS}]
    return {"pool": len(d), "length_quantiles_min_median_max": [int(d.length.min()), int(d.length.median()), int(d.length.max())],
            **{c.removeprefix("applicability_"): d[c].value_counts().to_dict() for c in cols}}


def ranking_with_each_allergenicity_score(models: dict) -> dict:
    """The ranking of Phase 1's 400-peptide pool with Phase 1's allergenicity score (percentile) and with the platform's (1 - p_allergen)."""
    from types import SimpleNamespace

    from food.pipeline.stages.ranking import FoodRankingDecision
    from schemas.run_config import StageConfig

    spec = importlib.util.spec_from_file_location("ranking_fixture", ROOT / "tests" / "unit" / "pipeline" / "test_food_ranking_stage.py")
    sys.path.insert(0, str(ROOT / "tests" / "unit" / "pipeline"))
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    seqs = [r["sequence"] for r in fx.POOL]
    ours = scores(models["allergenicity"], seqs)

    def run(use_platform: bool):
        cands = [fx.to_candidate(r) for r in fx.POOL]
        if use_platform:
            for c in cands:
                if "allergenicity" in c.predictions["models"]:
                    c.predictions["models"]["allergenicity"]["score"] = ours[c.sequence]["score"]
        stage = FoodRankingDecision()
        return stage.run(cands, StageConfig(), SimpleNamespace()), cands

    phase1, _ = run(False)
    platform, cands = run(True)
    a, b = [c.sequence for c in phase1], [c.sequence for c in platform]
    old = np.array([float(r["model_allergenicity_score"]) for r in fx.POOL])
    new = np.array([ours[r["sequence"]]["score"] for r in fx.POOL])
    cyto = np.array([float(r["model_cytotoxicity_score"]) if r["model_cytotoxicity_score"] != "" else np.nan for r in fx.POOL])
    common = [s for s in a if s in b]
    return {
        "pool": len(seqs), "final_with_phase1_score": len(a), "final_with_platform_score": len(b), "same_peptides_in_final": len(set(a) & set(b)),
        "same_order": a == b, "only_in_phase1_final": sorted(set(a) - set(b)), "only_in_platform_final": sorted(set(b) - set(a)),
        "rank_spearman_of_common": round(float(spearmanr([a.index(s) for s in common], [b.index(s) for s in common]).statistic), 3) if len(common) > 2 else None,
        "pool_spearman_phase1_vs_platform_score": round(float(spearmanr(old, new).statistic), 4),
        "pool_score_quantiles_5_50_95": {"phase1_percentile": [round(float(v), 3) for v in np.percentile(old, [5, 50, 95])], "platform_one_minus_p": [round(float(v), 3) for v in np.percentile(new, [5, 50, 95])]},
        "allergenicity_is_the_lower_safety_score": {"phase1_percentile": int((old < cyto).sum()), "platform_one_minus_p": int((new < cyto).sum()), "of": int((~np.isnan(cyto)).sum())},
        "tiebreak_order": list(settings.TIEBREAK),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "validation" / "food")
    args = parser.parse_args()
    fe = FeatureExtractor()
    models = {m.name: m for m in adapters.build_model_slots(fe)}
    status = {n: m.status + ("" if m.status == "ready" else f": {m.detail}") for n, m in models.items()}
    print("slots:", status, flush=True)
    result = {"slots": status, "applicability_on_phase1_pool": applicability_coverage()}
    for name, fn in (("allergenicity", lambda: allergenicity(models, fe)), ("bitterness", lambda: bitterness(models)), ("solubility", lambda: solubility(models)),
                     ("synthesis_feasibility", lambda: synthesis(models)), ("aggregation", lambda: aggregation(models)), ("training_lengths", training_lengths), ("illustrations", lambda: illustrations(models)),
                     ("ranking_allergenicity_definition", lambda: ranking_with_each_allergenicity_score(models))):
        result[name] = fn()
        print(name, "done", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_validation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "allergenicity"}, indent=1)[:6000])
    a = result["allergenicity"]
    print(json.dumps({k: v for k, v in a.items() if k != "calibration_of_p_allergen"}, indent=1))


if __name__ == "__main__":
    main()

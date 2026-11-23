"""Export the cytokine-induction models from the training project into model_store/model_weights/.

    python scripts/export_cytokine_models.py "<...>/Models/Cytokine Activation/Weighted_5models"

The training project pickles its own `benchlib.FinalModel`; this re-saves each model as a plain dict so the
platform never has to import benchlib. Reads only; writes model_weights/<package>/<member>.joblib and info.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from common.applicability import domain_profile  # noqa: E402
WEIGHTS = REPO / "model_store" / "model_weights"

# package -> (results directory, [(member name, source file)])
PACKAGES = {
    "ifn_gamma_induction_predictor_v1": ("IFN-gamma_FASTA", [("MLP_lenbin", "ESM__MLP.joblib")]),
    "tnf_alpha_induction_predictor_v1": (
        "TNF-alpha",
        [("MLP_posw", "ESM__MLP__posw.joblib"), ("SVM_RBF_smote", "ESM__SVM_RBF__smote.joblib"), ("XGBoost_posw", "ESM__XGBoost__posw.joblib")],
    ),
    "il4_induction_predictor_v1": ("IL-4", [("XGBoost_posw", "ESM__XGBoost__posw.joblib")]),
    "il6_induction_predictor_v1": (
        "IL-6",
        [("XGBoost_lenbin", "ESM__XGBoost.joblib"), ("MLP_adasyn", "ESM__MLP__adasyn.joblib"), ("SVM_RBF_posw", "ESM__SVM_RBF__posw.joblib")],
    ),
    "il10_induction_predictor_v1": ("IL-10", [("MLP_posw", "ESM__MLP__posw.joblib")]),
    "il13_induction_predictor_v1": ("IL-13", [("MLP_posw", "ESM__MLP__posw.joblib"), ("XGBoost_lenbin", "ESM__XGBoost.joblib")]),
}


def main(project: Path) -> None:
    sys.path.insert(0, str(project))  # unpickling needs benchlib.FinalModel
    for package, (results, members) in PACKAGES.items():
        out = WEIGHTS / package
        out.mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(project / "results" / results / "dataset_with_folds.csv")
        info = {
            "cytokine": results.split("_")[0],
            "features": "mean-pooled last-layer ESM2-150M (facebook/esm2_t30_150M_UR50D), BOS/EOS excluded, 640-d",
            "threshold": 0.5,
            "aggregation": "mean of member P(y=1)",
            "training": {
                "n": int(len(data)),
                "positive_rate": round(float(data.label.mean()), 4),
                "length_min": int(data.length.min()),
                "length_max": int(data.length.max()),
            },
            "members": [],
            "domain": {**domain_profile(data.length.values, data.label.values), "source": f"{results}/dataset_with_folds.csv"},
        }
        for name, source in members:
            path = project / "results" / results / "models" / source
            model = joblib.load(path)
            payload = {"kind": model.kind, "scaler": model.scaler, "est": model.est, "state": model.state, "dim": model.dim}
            joblib.dump(payload, out / f"{name}.joblib")
            info["members"].append({"name": name, "kind": model.kind, "source": f"{results}/models/{source}", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"{package}: {name} <- {source}")
        (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

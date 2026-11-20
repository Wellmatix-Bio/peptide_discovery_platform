"""Give each cytokine model a sequence-similarity profile: its reference peptides and the identity cut-offs for the bands.

    python scripts/add_similarity_profiles.py "<...>/Models/Cytokine Activation/Weighted_5models"

For each package in scripts/export_cytokine_models.PACKAGES this
  1. writes the model's training peptides to model_store/model_weights/<package>/reference.txt.gz (read by the worker only;
     the API never returns them),
  2. scores up to SAMPLE training peptides against the reference peptides of the OTHER cross-validation folds (CD-HIT
     clusters never span folds, so near-duplicates of the query are excluded), and sets "distant" at the 10th and "similar"
     at the 50th percentile of those best identities,
  3. records in info.json a `similarity` block: the cut-offs, the procedure, and the out-of-fold AUROC by band as evidence
     of whether distance from the reference actually costs accuracy.
It refuses to run if the dataset disagrees with the weights' info.json, like add_domain_profiles.py. Re-upload the weight
folders (info.json and reference.txt.gz) to VERTEX_MODEL_STORE afterwards.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.similarity import REFERENCE_FILE, SHORTLIST, best_identities, composition  # noqa: E402
from export_cytokine_models import PACKAGES, WEIGHTS  # noqa: E402

SAMPLE = 3000
SEED = 0
DISTANT_PERCENTILE, SIMILAR_PERCENTILE = 10, 50

#: results folder -> (OOF score file, one out-of-fold score column), used only to report accuracy by band.
EVIDENCE = {
    "IFN-gamma_FASTA": ("ensemble_oof_scores.csv", "MLP"),
    "TNF-alpha": ("oof_scores.csv", "ESM__MLP"),
    "IL-4": ("ensemble_oof_scores.csv", "XGBoost_posw"),
    "IL-6": ("ensemble_oof_scores.csv", "Ensemble"),
    "IL-10": ("oof_scores.csv", "ESM__MLP"),
    "IL-13": ("ensemble_oof_scores.csv", "Ensemble"),
}


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return round(float(roc_auc_score(labels, scores)), 3) if len(set(labels)) == 2 and len(labels) >= 30 else None


def main(project: Path) -> None:
    for package, (results, _members) in PACKAGES.items():
        info_path = WEIGHTS / package / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        folder = project / "results" / results
        data = pd.read_csv(folder / "dataset_with_folds.csv")
        t = info["training"]
        found = (len(data), round(float(data.label.mean()), 4), int(data.length.min()), int(data.length.max()))
        if found != (t["n"], t["positive_rate"], t["length_min"], t["length_max"]):
            raise SystemExit(f"{package}: dataset {found} disagrees with info.json {t}")

        sequences = data.sequence.tolist()
        matrix = composition(sequences)
        folds = data.cv_fold.to_numpy()
        rng = np.random.default_rng(SEED)
        picked = np.sort(rng.choice(len(data), size=min(SAMPLE, len(data)), replace=False))
        identity = best_identities([sequences[i] for i in picked], sequences, matrix, exclude=(folds == folds[i] for i in picked))
        keep = ~np.isnan(identity)
        distant, similar = (float(round(np.percentile(identity[keep], p), 3)) for p in (DISTANT_PERCENTILE, SIMILAR_PERCENTILE))

        score_file, column = EVIDENCE[results]
        oof = pd.read_csv(folder / score_file)
        if not (oof.sequence.values == data.sequence.values).all():
            oof = oof.set_index("sequence").reindex(data.sequence).reset_index()
        labels = data.label.to_numpy()[picked][keep]
        scores = oof[column].to_numpy()[picked][keep]
        ident = identity[keep]
        bands = {
            "distant": ident < distant,
            "moderate": (ident >= distant) & (ident < similar),
            "similar": ident >= similar,
        }
        evidence = {name: {"n": int(mask.sum()), "oof_auroc": auroc(labels[mask], scores[mask])} for name, mask in bands.items()}

        reference_path = WEIGHTS / package / REFERENCE_FILE
        with gzip.open(reference_path, "wt", encoding="utf-8") as handle:
            handle.write("\n".join(sequences) + "\n")
        info["similarity"] = {
            "method": f"best BLOSUM62 alignment identity over the query length, among the {SHORTLIST} reference peptides closest in 2-mer composition",
            "n_reference": len(sequences),
            "similar_at": similar,
            "distant_below": distant,
            "calibration": {
                "queries": int(keep.sum()),
                "against": "reference peptides from the other cross-validation folds",
                "percentiles": {"distant_below": DISTANT_PERCENTILE, "similar_at": SIMILAR_PERCENTILE},
                "identity_quantiles": {str(q): round(float(np.percentile(ident, q)), 3) for q in (5, 10, 25, 50, 75, 90)},
                "oof_auroc_by_band": evidence,
            },
            "source": f"{results}/dataset_with_folds.csv",
            "source_sha256": hashlib.sha256((folder / "dataset_with_folds.csv").read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"{package}: distant < {distant}, similar >= {similar}; OOF AUROC by band {evidence}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

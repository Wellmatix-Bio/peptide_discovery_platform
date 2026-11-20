"""Write each cytokine model's training-data `domain` block (per-length positive/negative counts) into its info.json.

    python scripts/add_domain_profiles.py "<...>/Models/Cytokine Activation/Weighted_5models"

Reads results/<name>/dataset_with_folds.csv for each package in scripts/export_cytokine_models.PACKAGES and refuses to
write if the dataset's size, positive rate or length range disagrees with what the weights' info.json recorded, so a
profile can never describe a different training set than the model's. Re-upload the info.json files to
VERTEX_MODEL_STORE afterwards: the production worker reads them from there.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.applicability import domain_profile  # noqa: E402
from export_cytokine_models import PACKAGES, WEIGHTS  # noqa: E402


def main(project: Path) -> None:
    for package, (results, _members) in PACKAGES.items():
        info_path = WEIGHTS / package / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        dataset = project / "results" / results / "dataset_with_folds.csv"
        data = pd.read_csv(dataset)
        t = info["training"]
        found = (len(data), round(float(data.label.mean()), 4), int(data.length.min()), int(data.length.max()))
        recorded = (t["n"], t["positive_rate"], t["length_min"], t["length_max"])
        if found != recorded:
            raise SystemExit(f"{package}: {dataset} has (n, positive rate, min, max) = {found} but info.json recorded {recorded}")
        info["domain"] = {
            **domain_profile(data.length.values, data.label.values),
            "source": f"{results}/dataset_with_folds.csv",
            "source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        }
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"{package}: {len(info['domain']['length_counts'])} lengths from {results}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

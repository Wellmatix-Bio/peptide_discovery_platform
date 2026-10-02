# Licensing

This project is **Apache-2.0** (see `LICENSE` and `NOTICE`). That choice has consequences for two
dependencies and for the model weights, and they are spelled out here rather than left for someone
to discover after they have built on it.

**Nothing here is legal advice.** It is a description of what the code does and what the
dependencies declare. If you are redistributing this, or combining it with your own work, take it
to a lawyer.

---

## Two optional copyleft dependencies

Neither is required, vendored, or installed by default. The project does not depend on them to
build, to import, or to pass its tests. Install them yourself if you want what they enable, and
the resulting combined work is subject to their terms.

| Dependency | Licence | Enables | Without it |
|---|---|---|---|
| `propy3` | **GPL-2.0-only** | The aggregation predictor's feature set | The aggregation screen does not run, in stage 5 and stage 8 |
| `s4pred` | **GPL-3.0** | The secondary-structure screen in stage 5 | That one screen does not run |

### Why they are optional rather than required

GPL-2.0-only and GPL-3.0 are each copyleft, and they are **mutually incompatible** — a single
distributed work cannot satisfy both. Requiring both, as this project previously did, is a problem
independent of which licence the project itself chooses.

Making them optional removes that from the distributed work. It also, unexpectedly, fixed the
repository's longest-standing defect: `s4pred` was imported at module scope in stage 5, and stage 5
is imported eagerly by `pipeline/__init__.py`, so a missing `s4pred` made the **entire pipeline
package unimportable** and 47 tests uncollectable. They now collect (see `docs/BASELINE.md`).

### What happens when they are absent

The rule throughout is that **a screen that did not run is never reported as a pass.**

- `feature_extractor.PROPY_AVAILABLE` is `False`, and `_propy_features()` raises
  `PropyUnavailable` rather than returning defaults. Handing the aggregation model zeros would
  produce a confident-looking score computed from nothing.
- Stage 5's `compute_aggregation_tendency` returns `{"score": None, "status": "unavailable",
  "reason": ...}`.
- Stage 8 records the property as **`not_screened`**, not `pass`, and the overall verdict becomes
  `flag` — the candidate is not silently cleared, and not rejected either, because nothing was
  measured against it. The verdict carries a `not_screened` list and a note saying so.
- Stage 5's `compute_secondary_structure_consistency` returns `{"available": False, "reason": ...}`.

This also corrected a pre-existing bug. Stage 8 read `else "pass"` for a null aggregation score,
so a peptide **too short for the model's features** was already being recorded as having passed a
screen that never ran. That path now reports `not_screened` too.

### Installing them anyway

```bash
pip install propy3
```

`s4pred` is not on PyPI. It is vendored into `src/pipeline/s05_physchem_screening/s4pred/` as a
git submodule pointer with no `.gitmodules` entry, which does not resolve — see `docs/BASELINE.md`.
To use it, obtain it from its upstream project and place it there yourself.

Doing either means your installation combines Apache-2.0 code with GPL code. That is your call to
make, and the terms of the GPL dependency govern what you may then distribute.

---

## Model weights are not covered by this licence

**No weights are in this repository.** `model_store/<predictor>/` holds code — `predictor.py`, a
`README.md`, and a `model_card.json` for 8 of the 15. The weights live in
`model_store/model_weights/`, which is gitignored and populated at run time from whatever
`VERTEX_MODEL_STORE` points at.

Each predictor's weights carry their own terms, including those of any upstream model they derive
from. At minimum:

- Several predictors build on **ESM-2** (`facebook/esm2_t30_150M_UR50D`) and **ESMFold**
  (`facebook/esmfold_v1`) from Meta AI.
- Route B builds on **ProtGPT2** (`nferruz/ProtGPT2`) with a LoRA adapter.

**Open question for this repository.** Seven of the fifteen predictors ship no model card, so their
training data, applicability domain and licence are not documented anywhere. Before weights are
distributed to anyone, each one needs its terms stated. The Models & health page in the web app
already reports which predictors lack a card, and says plainly that *not checked is not the same as
passed*.

---

## Dependency licences

Audited from installed package metadata. Everything required is permissive or weak-copyleft in a
way that is satisfied by depending on it:

| Licence | Packages |
|---|---|
| MIT / BSD / Apache-2.0 | most of the tree, including torch, transformers, fastapi, pydantic, scikit-learn, xgboost, fair-esm, bitsandbytes, freesasa, modlamp |
| **LGPL** | `deap` (the genetic algorithm in route A) |
| **MPL-2.0** | `gemmi`, `tqdm` |
| **GPL** | `propy3` (optional), `s4pred` (optional) |

LGPL and MPL are file-level or library-level copyleft: using them as dependencies, unmodified, does
not impose their terms on this project. Modifying and redistributing them does.

Re-run the audit after any dependency change:

```bash
.venv/bin/python -c "import importlib.metadata as m; [print(f'{d.metadata[\"Name\"]:24} {d.metadata.get(\"License-Expression\") or d.metadata.get(\"License\") or \"-\"}') for d in m.distributions()]" | sort -u
```

---

## Before publishing

- [x] `LICENSE` (Apache-2.0) and `NOTICE` present
- [x] No secrets, project ids or bucket names in the repository **or its git history** — verified
      with `git log --all -S`, not just the working tree
- [x] Copyleft dependencies optional and documented
- [ ] Per-predictor weight licences stated — **7 of 15 have no model card**
- [ ] A lawyer has reviewed the above

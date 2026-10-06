# Wellmatix Peptide Discovery Platform

Generates, screens and ranks candidate peptide sequences for wound healing.

You describe a wound and what the peptide should do — context, desired biological functions,
target pathogens, length and dosing constraints. The pipeline generates candidates, screens them
through physicochemical, functional, structural, safety and synthesis models, and returns a ranked
shortlist with the evidence behind each one.

**Every number it produces is a model's estimate on a computationally generated sequence. None of
it substitutes for wet-lab validation.** The interface says so too; see [What this platform will
not answer](#what-this-platform-will-not-answer).

[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

**[wellmatix-bio.github.io/peptide_discovery_platform](https://wellmatix-bio.github.io/peptide_discovery_platform/)** — project page, once GitHub Pages is enabled for `main` → `/docs`.

---

## Try it without any cloud account

The pipeline runs locally. No Google Cloud, no Docker, no GPU required.

```bash
git clone https://github.com/Wellmatix-Bio/peptide_discovery_platform.git
cd peptide_discovery_platform
```

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
python main.py configs/runs/test_run.yaml
```

That executes the whole pipeline against local paths. It is slower on CPU, and the predictors need
model weights (see [Model weights](#model-weights)), but nothing about it is a reduced version of
the cloud path — it is the same code.

Running the web app and the full service stack is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## What it is made of

| | |
|---|---|
| **Pipeline** | 14 stage directories under `src/pipeline/`, 8 implemented |
| **Models** | 16 predictors in `model_store/` — ESM-2 embeddings with XGBoost/sklearn heads, a BiLSTM+CNN MIC ensemble, ESMFold for structure, ProtGPT2+LoRA for de novo generation |
| **Job API** | `src/backend/api_e2e/` — submits Vertex AI Custom Jobs, reads their artifacts |
| **Accounts** | `services/accounts/` — email/password accounts and the authenticating proxy |
| **Web app** | `web/` — React, TypeScript, one origin, no CORS |
| **Deployment** | `deploy/` — three containers, only the web one publishes a port |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how they fit together and what you can leave
out.

---

## What is optional

The core has **no hard dependency** on Google Cloud, on a GPU, or on any copyleft library. Each is
an addition, and each announces itself when missing rather than failing obscurely.

| Leave out | Install with | Without it |
|---|---|---|
| Google Cloud | `pip install -e '.[gcp]'` | Local paths work normally; a `gs://` path raises an error naming the fix |
| `propy3` (GPL-2.0-only) | `pip install -e '.[aggregation]'` | The aggregation screen reports `not_screened` — never a silent pass |
| `s4pred` (GPL-3.0) | [set it up yourself](CONTRIBUTING.md#setting-up-s4pred-optional-gpl-30) | The secondary-structure screen reports itself unavailable and the verdict flags `not_screened` |
| A GPU | — | Runs on CPU, slowly. ESMFold in stage 7 dominates |

Both copyleft packages are optional on purpose. See [docs/LICENSING.md](docs/LICENSING.md).

---

## Model weights

**Not in this repository.** `model_store/<predictor>/` holds code and a model card;
`model_store/model_weights/` is gitignored and populated at run time from wherever
`VERTEX_MODEL_STORE` points.

How outside users obtain them is **not yet settled**. Weights carry their own terms, including
those of the upstream models they derive from (ESM-2, ESMFold, ProtGPT2), and **8 of the 16
predictors have only a placeholder model card**, so their training data and applicability
domain are undocumented; the API's `model_card_status` says which.
Tracked in [docs/LICENSING.md](docs/LICENSING.md).

---

## What this platform will not answer

Stated plainly because the interface states it too:

- **Pathogens other than *E. coli*, *S. aureus* and *P. aeruginosa*.** A pathogen outside that list
  is skipped, not scored — it is not an error and produces no warning in the results.
- **Peptides shorter than 6 or longer than 50 residues.** Refused rather than scored out of domain.
- **Wound contexts and desired functions outside the API's vocabularies.** Note that the 32 example
  briefs in `data/briefs/` use many terms it rejects — see
  [docs/BRIEF_VALIDITY.md](docs/BRIEF_VALIDITY.md).
- **Which models produced a given result.** No run carries a model identity.
- **Whether a candidate works.** That is what a laboratory is for.

---

## Known issues

Open and documented rather than discovered later:

- The biofilm model scores 13 pathogens while a brief may request 3, and `antibiofilm` is a valid
  generation tag but not a valid desired function — [docs/BRIEF_VALIDITY.md](docs/BRIEF_VALIDITY.md)
- `status` never reflects the Vertex job state; a dead worker reports `"pending"` forever —
  [docs/BASELINE.md](docs/BASELINE.md)
- `s4pred` is optional and not vendored (GPL-3.0). The screen it powers reports itself
  unavailable; [CONTRIBUTING.md](CONTRIBUTING.md#setting-up-s4pred-optional-gpl-30) has the setup
  if you want it
- Stage 7's pathway model reports **P(activator)**, not P(involved), and at the shipped threshold
  of 0.5 the activated and inhibited bands meet — so every pathway gets a direction and a 0.501
  coin flip reads as a definite call — [docs/BASELINE.md](docs/BASELINE.md)
- The unimplemented stage directories (s10, s12–s14) have no tests, and nor do s01–s03. Every
  implemented scoring stage — s04 through s09 and s11 — now does

---

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pieces fit, what is optional |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Running the stack, locally and on a VM |
| [LICENSING.md](docs/LICENSING.md) | Apache-2.0, the copyleft optionals, model weights |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, tests, how changes are reviewed |
| [API_SPECIFICATION.md](API_SPECIFICATION.md) | The job API's endpoints |
| [BASELINE.md](docs/BASELINE.md) | Measured test baseline and the defects behind it |
| [WEB_WALKTHROUGH.md](docs/WEB_WALKTHROUGH.md) | Every page, with screenshots |
| [REVIEW_LOG.md](docs/REVIEW_LOG.md) | Queries raised in review, what was found, why it was missed |
| [PATHWAY_THRESHOLD.md](docs/PATHWAY_THRESHOLD.md) | For the model owner: the one open modelling decision |

---

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are **not** covered by it.

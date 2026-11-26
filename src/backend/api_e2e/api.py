"""Start and inspect complete-pipeline Vertex AI Custom Jobs."""

import hashlib
import dataclasses
import json
import os
import re
import statistics
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.auth.exceptions import GoogleAuthError
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)
from common import storage
from common.food_digestion.engine import run as run_digestion
from common.food_digestion.rules import build_registry as digestion_registry
from common.food_digestion.settings import DEFAULT_SETTINGS as DIGESTION_DEFAULT_SETTINGS
from common.food_digestion.validation import LIMITATIONS as DIGESTION_LIMITATIONS, RULE_VALIDATION
from common.food_library import Match as LibraryMatchResult, UnknownLibraryProteins, library as food_library
from common.protein_fasta import read_fasta as read_protein_fasta

# Imported for its side effect: common/env.py loads .env into os.environ; do not remove as unused.
from common import env  # noqa: F401
from common.env import DEV_MODE
from common.model_registry import ModelRegistry
from common.model_sync import weights_dir_for
from common.applicability import NOT_CHECKED, ReferenceProfile
from common.sequence_utils import parse_sequence_items
from schemas.brief import Brief, CosmeticBrief, MetabolicBrief
from schemas.e2e_config import validate_job_config
from schemas.food_run_config import FOOD_REQUIRED_WEIGHT_FILES, FoodRunConfig
from schemas.food_dossier import (FoodDossier, ReviewRecord, ReviewRequest, build_dossier, dossier_html, experiment_sheet_csv, gate_summary, latest, new_record, parse_log, ranking_summary)
from schemas.food_dossier_pdf import dossier_pdf
from schemas.immuno_config import CYTOKINE_MODELS, DEFAULT_BATCH_SIZE, IMMUNO_MAX_PEPTIDES, REQUIRED_WEIGHT_FILES, ImmunoRunConfig
from schemas.stage_configs import CosmeticE2ERequest, E2ERequest, MetabolicE2ERequest, resolve_params

app = FastAPI(title="End-to-end Pipeline Job API")

if DEV_MODE:
    # Local development only: a page opened from a file or another port (the standalone studio) is cross-origin to this API.
    # No credentials are involved, so any origin is allowed; production (DEV_MODE off) adds no CORS and stays same-origin.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def required(name):
    value = os.environ.get(name, "")
    if not value:
        raise HTTPException(503, f"Server setting {name} is required")
    return value


@lru_cache
def job_client():
    from google.cloud import aiplatform_v1

    return aiplatform_v1.JobServiceClient(
        client_options={
            "api_endpoint": f"{required('VERTEX_LOCATION')}-aiplatform.googleapis.com"
        }
    )


Pipeline = Literal["wound_healing", "metabolic", "cosmetic"]


def _stages_pipeline(stages) -> str:
    """Which request schema `stages` belongs to, by the brief stage it carries; `pipeline` is checked against it below."""
    keys = (
        stages
        if isinstance(stages, dict)
        else stages.model_fields_set | set(type(stages).model_fields)
    )
    if "s01_cosmetic_therapeutic_product_brief" in keys:
        return "cosmetic"
    if "s01_metabolic_therapeutic_product_brief" in keys:
        return "metabolic"
    return "wound_healing"


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    #: Routes the job: which stages run and which worker image executes them. Same endpoint for both.
    pipeline: Pipeline = "wound_healing"
    stages: Annotated[
        Union[
            Annotated[E2ERequest, Tag("wound_healing")],
            Annotated[MetabolicE2ERequest, Tag("metabolic")],
            Annotated[CosmeticE2ERequest, Tag("cosmetic")],
        ],
        Discriminator(_stages_pipeline),
    ]

    @model_validator(mode="after")
    def _stages_match_pipeline(self):
        actual = (
            "metabolic"
            if isinstance(self.stages, MetabolicE2ERequest)
            else "cosmetic"
            if isinstance(self.stages, CosmeticE2ERequest)
            else "wound_healing"
        )
        if actual != self.pipeline:
            raise ValueError(
                f"pipeline is {self.pipeline!r} but stages is a {actual} request; set pipeline to {actual!r}"
                " or send that pipeline's stages"
            )
        return self


class CreateJobResponse(BaseModel):
    request_id: str
    job_id: str
    result_path: str


#: Vertex states meaning the job will not progress any further. Kept in step with
#: services/accounts/accounts/history.py and web/src/api/peptide.ts, which list the same four.
TERMINAL_VERTEX_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


class JobStatusResponse(BaseModel):
    job_id: str
    #: THE WORKER'S OWN LAST WORD, unchanged and deliberately so: "pending" until it writes
    #: anything, then "running" / "success" / "failed". It can be stale, because a worker that
    #: dies writes nothing further. Read `run_state` for the reconciled answer.
    status: str
    stage: str | None = "pending"
    vertex_state: str
    #: `status` and `vertex_state` reconciled, so a client does not have to. See run_state().
    run_state: str = "submitted"
    error: str | None = None


#: Receptors whose potency the ensemble reports but CANNOT RANK. The owning definition is
#: UNRELIABLE_POTENCY_RECEPTORS in model_store/functional_potency_predictor_v1, and the note is
#: UNRELIABLE_POTENCY_NOTE in src/pipeline/s06_functional_models/metabolic_stage.py. Both are
#: repeated here rather than imported because importing either pulls torch into the API process,
#: which only submits jobs and reads JSON. tests/unit/test_metabolic_gipr.py holds the copies to
#: the originals, so a receptor added or removed there fails there rather than drifting here.
#:
#: The API needs them because stage 6 only started recording the flag per candidate recently.
#: Runs from before that have GIPR pEC50 and EC50 values and no flag at all, so every client --
#: the web app included -- showed the numbers with nothing saying they cannot be ranked.
UNRELIABLE_POTENCY_RECEPTORS = ("GIPR",)

UNRELIABLE_POTENCY_NOTE = (
    "The potency model cannot rank these receptors: its leave-one-cluster-out R2 is below zero"
    " for them, meaning it does worse than predicting the training mean, so the pEC50 and EC50"
    " returned for them order candidates no better than chance. They are reported rather than"
    " withheld, and they still count toward the gates and the ranking score. Treat them as"
    " unscreened, not as evidence. See the model card for functional_potency_predictor_v1."
)


def _label_value(value: str) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-_")
    return (label or "request")[:63]


def job_name(job_id):
    if not re.fullmatch(r"projects/[^/]+/locations/[^/]+/customJobs/[0-9]+", job_id):
        raise HTTPException(
            422, "job_id must be the full Vertex Custom Job resource name"
        )
    return job_id


@app.post("/api/v1/jobs/create", response_model=CreateJobResponse, status_code=202)
def create_job(request: CreateJobRequest):
    parent = f"projects/{required('VERTEX_CLOUD_PROJECT')}/locations/{required('VERTEX_LOCATION')}"
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    models = required("VERTEX_MODEL_STORE")
    # Each pipeline has its own seed file; the request cannot choose it.
    seeds = required(
        {
            "metabolic": "SEED_CANDIDATES_FILE_METABOLIC",
            "cosmetic": "SEED_CANDIDATES_FILE_COSMETIC",
        }.get(request.pipeline, "SEED_CANDIDATES_FILE")
    )
    metabolic = request.pipeline == "metabolic"
    cosmetic = request.pipeline == "cosmetic"
    image = required(
        {
            "metabolic": "WORKER_METABOLIC_IMAGE_URI",
            "cosmetic": "WORKER_COSMETIC_IMAGE_URI",
        }.get(request.pipeline, "WORKER_IMAGE_URI")
    )
    service_account = required("WORKER_SERVICE_ACCOUNT")
    machine_type = required("MACHINE_TYPE")
    accelerator_type = required("ACCELERATOR_TYPE")
    accelerator_count = int(required("ACCELERATOR_COUNT"))
    # Environment variables passed into the worker; everything else it needs travels in its config.json.
    worker_env_keys = ("VERTEX_MODEL_STORE",)

    if not all(storage.is_gcs_path(p) for p in (artifacts, models, seeds)):
        raise HTTPException(
            503, "Artifacts, model store, and seed FASTA must use gs:// paths"
        )
    # request_key = hashlib.sha256(request.request_id.encode()).hexdigest()
    staging_dir = storage.join(artifacts, "requests", request.request_id)
    staging_config_path = storage.join(staging_dir, "config.json")

    # A RETRY RETURNS THE JOB THE FIRST ATTEMPT STARTED. It must not start a second one.
    #
    # Every job for a request_id is launched with `--config <staging_config_path>` -- one path,
    # shared by every attempt -- and the worker waits up to 120s for that file to appear. So a
    # second create overwrote it with the second run's config, and the first job, if it had not
    # read the file yet, then picked up the SECOND run's config: two paid GPU jobs writing their
    # results into one run directory, with the first run's own directory left empty forever.
    #
    # shortcut: this is read-then-write, so two creates racing inside the same instant can both
    # still pass. Upgrade to a GCS precondition write (ifGenerationMatch=0) on the staging config
    # if that is ever observed; the window here is the width of one GCS write.
    if storage.exists(staging_config_path):
        try:
            previous = json.loads(storage.read_text(staging_config_path))
            run_id = previous["run_id"]
        except Exception:
            # Unreadable or half-written: there is no job name to return, so fall through and
            # create one. Logged because it is the only path that can still double-submit.
            logging.exception(
                "Staging config %s exists but could not be read; treating %s as a new request",
                staging_config_path,
                request.request_id,
            )
        else:
            return CreateJobResponse(
                request_id=request.request_id,
                job_id=f"{parent}/customJobs/{run_id}",
                result_path=storage.join(artifacts, "runs", run_id),
            )
    try:
        brief_stage = {
            "metabolic": "s01_metabolic_therapeutic_product_brief",
            "cosmetic": "s01_cosmetic_therapeutic_product_brief",
        }.get(request.pipeline, "s01_therapeutic_product_brief")
        config = validate_job_config(
            {
                "artifacts_dir": artifacts,
                "model_store": models,
                "seed_candidates_path": seeds,
                "stages": request.stages.model_dump(),
            },
            pipeline=request.pipeline,
        )
        brief_model = {"metabolic": MetabolicBrief, "cosmetic": CosmeticBrief}.get(
            request.pipeline, Brief
        )
        brief_model.model_validate(config.for_stage(brief_stage).params["brief"])
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    custom_job = {
        "display_name": f"ai-peptide-discovery-{ {'metabolic': 'metabolic', 'cosmetic': 'cosmetic'}.get(request.pipeline, 'e2e') }-{request.request_id[:12]}",
        "labels": {
            "service": "battery-ai",
            "pipeline": request.pipeline.replace("_", "-"),
            "request-id": _label_value(request.request_id),
        },
        "job_spec": {
            "service_account": service_account,
            "worker_pool_specs": [
                {
                    "replica_count": 1,
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                    "container_spec": {
                        "image_uri": image,
                        "args": [
                            "--config",
                            staging_config_path,
                            "--wait-for-config",
                            "120",
                        ],
                        "env": [
                            {"name": "DEV_MODE", "value": "false"},
                            *(
                                {"name": key, "value": os.environ[key]}
                                for key in worker_env_keys
                                if key in os.environ
                            ),
                        ],
                    },
                }
            ],
        },
    }
    try:
        client = job_client()
        job = client.create_custom_job(parent=parent, custom_job=custom_job)
        job_id = job.name.rsplit("/", 1)[-1]
        config.run_id = job_id
        result_path = storage.join(artifacts, "runs", job_id)
        config_path = storage.join(result_path, "config.json")
        try:
            body = config.model_dump_json(indent=2)
            storage.write_text(config_path, body)
            storage.write_text(staging_config_path, body)
        except Exception as exc:
            try:
                client.cancel_custom_job(name=job.name, timeout=30)
            except Exception:
                logging.exception(
                    "Could not cancel job %s after config upload failed", job.name
                )
            raise HTTPException(
                502, f"Config upload failed for {job.name}; cancellation requested"
            ) from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return CreateJobResponse(
        request_id=request.request_id,
        job_id=job.name,
        result_path=result_path,
    )


def _immuno_batch_size() -> int:
    """Peptides the worker's models process at a time: IMMUNO_BATCH_SIZE, else the schema default."""
    try:
        return max(1, int(os.environ.get("IMMUNO_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    except ValueError:
        return DEFAULT_BATCH_SIZE


class ImmunoCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, description="Caller's name for the run. A repeat of the same request_id returns the first job instead of starting another.")
    peptides: list[str] = Field(
        min_length=1,
        max_length=IMMUNO_MAX_PEPTIDES,
        description="Sequences or FASTA entries. Each is checked for the 20 standard residues, length 6-50 and duplicates; the worker then scores them in batches.",
    )


class ImmunoRejection(BaseModel):
    id: str
    sequence: str
    reason: str
    detail: str = ""


class ImmunoCreateResponse(CreateJobResponse):
    accepted: int = Field(description="Peptides submitted to the job, after QC and de-duplication.")
    rejected: list[ImmunoRejection] = Field(default_factory=list, description="What QC removed, and why.")


#: DEV_MODE: immunomodulation jobs run as local subprocesses and get ids shaped like a Vertex resource name, so the status,
#: results and cancel routes, the accounts proxy's ownership check and the web page treat them like any other job. The handles
#: live in this process: after an API restart a running job cannot be cancelled, and its status falls back to its results.json.
LOCAL_JOB_PREFIX = "projects/local/locations/local/customJobs/"
_REPO = Path(__file__).resolve().parents[3]
_local_processes: dict[str, subprocess.Popen] = {}
_local_cancelled: set[str] = set()  # run ids a client cancelled: their status reads cancelled, not failed


def _local_artifacts() -> str:
    """VERTEX_ARTIFACTS_DIR when it is a local path, else <repo>/artifacts."""
    configured = os.environ.get("VERTEX_ARTIFACTS_DIR", "")
    return configured if configured and not configured.startswith("gs://") else str(_REPO / "artifacts")


def _create_local_immunomodulation_job(request: ImmunoCreateRequest, peptides: list[str], rejected: list[ImmunoRejection]):
    """DEV_MODE: run the worker as a subprocess with local weights; its output goes to <run dir>/worker.log."""
    artifacts = _local_artifacts()
    run_id = str(time.time_ns() // 1_000_000)
    run_dir = storage.join(artifacts, "runs", run_id)
    config_path = storage.join(run_dir, "config.json")
    config = ImmunoRunConfig(run_id=run_id, artifacts_dir=artifacts, model_store="model_store", peptides=peptides, batch_size=_immuno_batch_size())
    storage.write_text(config_path, config.model_dump_json(indent=2))
    env = {**os.environ, "DEV_MODE": "true", "PYTHONPATH": os.pathsep.join([str(_REPO / "src"), str(_REPO), os.environ.get("PYTHONPATH", "")])}
    with open(Path(run_dir) / "worker.log", "ab") as log:
        _local_processes[run_id] = subprocess.Popen(
            [sys.executable, "-m", "backend.worker_immuno.worker", "--config", config_path],
            cwd=str(_REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return ImmunoCreateResponse(
        request_id=request.request_id,
        job_id=LOCAL_JOB_PREFIX + run_id,
        result_path=run_dir,
        accepted=len(peptides),
        rejected=rejected,
    )


@app.post("/api/v1/jobs/immunomodulation/create", response_model=ImmunoCreateResponse, status_code=202)
def create_immunomodulation_job(request: ImmunoCreateRequest):
    """Start a job that scores peptides on six cytokine-induction models. The worker processes them in batches of IMMUNO_BATCH_SIZE (default 100); progress is reported in the status route's `stage`. With DEV_MODE the job runs as a local process on local weights."""
    parsed = parse_sequence_items(request.peptides)
    rejected = [ImmunoRejection(id=r.id, sequence=r.sequence, reason=r.reason, detail=r.detail) for r in parsed.rejected]
    if not parsed.accepted:
        raise HTTPException(422, {"message": "No valid peptides in the request", "rejected": [r.model_dump() for r in rejected]})
    peptides = parsed.sequences
    if DEV_MODE:
        return _create_local_immunomodulation_job(request, peptides, rejected)

    parent = f"projects/{required('VERTEX_CLOUD_PROJECT')}/locations/{required('VERTEX_LOCATION')}"
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    models = required("VERTEX_MODEL_STORE")
    image = required("WORKER_IMMUNO_IMAGE_URI")
    service_account = required("WORKER_SERVICE_ACCOUNT")
    machine_type = required("MACHINE_TYPE")
    accelerator_type = required("ACCELERATOR_TYPE")
    accelerator_count = int(required("ACCELERATOR_COUNT"))
    if not all(storage.is_gcs_path(p) for p in (artifacts, models)):
        raise HTTPException(503, "Artifacts and model store must use gs:// paths")
    # The worker runs with DEV_MODE=false (set in the job below), so it downloads every model from this bucket.
    # Refuse now if the bucket is the placeholder or lacks a model, rather than failing the GPU job minutes in.
    if "TODO-bucket" in models:
        raise HTTPException(503, "VERTEX_MODEL_STORE is still the placeholder; set it to the bucket that holds the model weights")
    missing = [f for f in REQUIRED_WEIGHT_FILES if not storage.exists(storage.join(models, f))]
    if missing:
        raise HTTPException(503, f"Model weights missing from {models}: {', '.join(missing)}")

    # Its own staging folder, so a request_id shared with another pipeline cannot return that pipeline's job.
    staging_config_path = storage.join(artifacts, "requests", "immunomodulation", request.request_id, "config.json")

    # A retry returns the job the first attempt started rather than submitting a second one; see create_job for why.
    if storage.exists(staging_config_path):
        try:
            run_id = json.loads(storage.read_text(staging_config_path))["run_id"]
        except Exception:
            logging.exception("Staging config %s exists but could not be read; treating %s as new", staging_config_path, request.request_id)
        else:
            return ImmunoCreateResponse(
                request_id=request.request_id,
                job_id=f"{parent}/customJobs/{run_id}",
                result_path=storage.join(artifacts, "runs", run_id),
                accepted=len(peptides),
                rejected=rejected,
            )

    custom_job = {
        "display_name": f"ai-peptide-discovery-immuno-{request.request_id[:12]}",
        "labels": {"service": "battery-ai", "pipeline": "immunomodulation", "request-id": _label_value(request.request_id)},
        "job_spec": {
            "service_account": service_account,
            "worker_pool_specs": [
                {
                    "replica_count": 1,
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                    "container_spec": {
                        "image_uri": image,
                        "args": ["--config", staging_config_path, "--wait-for-config", "120"],
                        "env": [
                            {"name": "DEV_MODE", "value": "false"},
                            {"name": "VERTEX_MODEL_STORE", "value": models},
                        ],
                    },
                }
            ],
        },
    }
    try:
        client = job_client()
        job = client.create_custom_job(parent=parent, custom_job=custom_job)
        job_id = job.name.rsplit("/", 1)[-1]
        result_path = storage.join(artifacts, "runs", job_id)
        config = ImmunoRunConfig(run_id=job_id, artifacts_dir=artifacts, model_store=models, peptides=peptides, batch_size=_immuno_batch_size())
        try:
            body = config.model_dump_json(indent=2)
            storage.write_text(storage.join(result_path, "config.json"), body)
            storage.write_text(staging_config_path, body)
        except Exception as exc:
            try:
                client.cancel_custom_job(name=job.name, timeout=30)
            except Exception:
                logging.exception("Could not cancel job %s after config upload failed", job.name)
            raise HTTPException(502, f"Config upload failed for {job.name}; cancellation requested") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return ImmunoCreateResponse(
        request_id=request.request_id, job_id=job.name, result_path=result_path, accepted=len(peptides), rejected=rejected
    )


#: Model-card fields that describe how a cytokine model is built (which classifiers, how they are combined, how they were
#: trained) and so are never returned to clients. The cards in model_store/ keep them for the team.
CONFIDENTIAL_CARD_KEYS = ("members", "aggregation", "cv_metrics_note", "limits", "provenance")


def _public_card(package: str, card: dict | None) -> dict | None:
    """The card as clients may see it: for the immunomodulation models, without the build details above."""
    if card is None or package not in {p for p, _ in CYTOKINE_MODELS.values()}:
        return card
    public = {k: v for k, v in card.items() if k not in CONFIDENTIAL_CARD_KEYS}
    public["training"] = {k: v for k, v in (card.get("training") or {}).items() if k != "final_fit"}
    return public


class ImmunoModelInfo(BaseModel):
    cytokine: str
    package: str
    name: str
    version: str
    registered: bool  # found in common.model_registry.ModelRegistry (needs a readable model_card.json)
    checksum: str | None = None
    card_status: str | None = None  # e.g. "first version"
    task: str | None = None
    threshold: float | None = None
    cv_metrics: dict = Field(default_factory=dict)  # of the model as served
    training: dict = Field(default_factory=dict)  # n, positive_rate, length_min, length_max, source
    #: None when the weights' info.json cannot be read here (the API image carries no weights and the bucket is unset or unreachable).
    domain: dict | None = None


class ImmunoModelsResponse(BaseModel):
    models: list[ImmunoModelInfo]
    caveat: str


def _immuno_weights_info(package: str) -> dict | None:
    """The weights' info.json: the local weights folder first, else the model store bucket; None if neither is readable."""
    local = weights_dir_for(MODEL_STORE_DIR / package) / "info.json"
    try:
        if local.is_file():
            return json.loads(local.read_text(encoding="utf-8"))
        store = os.environ.get("VERTEX_MODEL_STORE", "")
        if not DEV_MODE and storage.is_gcs_path(store) and "TODO-bucket" not in store:
            return json.loads(storage.read_text(storage.join(store, package, "info.json")))
    except Exception:
        logging.warning("Could not read info.json for %s", package, exc_info=True)
    return None


def _immuno_domain(info: dict | None) -> dict | None:
    if not info:
        return None
    try:
        profile = ReferenceProfile.from_info(info)
    except (KeyError, TypeError, ValueError):
        return None
    domain = info.get("domain") or {}
    return {
        "min_examples_per_class": profile.min_per_class,
        "per_length_counts_recorded": profile.counts is not None,
        "status_ranges": profile.status_ranges(),  # lengths 6-50 -> in_domain / limited_evidence / out_of_domain
        "not_checked": [c for c in NOT_CHECKED if not (c == "sequence_similarity" and info.get("similarity"))],
        "similarity": {k: v for k, v in (info.get("similarity") or {}).items() if k in ("method", "similar_at", "distant_below")} or None,
        "source": domain.get("source"),
    }


@app.get("/api/v1/immunomodulation/models", response_model=ImmunoModelsResponse)
def immunomodulation_models():
    """What the studio's six cytokine models are: version and checksum from the model registry, performance and limits from each model card, and the per-length applicability ranges from the weights' info.json."""
    registry = ModelRegistry(MODEL_STORE_DIR)
    entries = []
    for cytokine, (package, _cls) in CYTOKINE_MODELS.items():
        name, version = package.rsplit("_", 1)
        ref = registry.get(name, version)
        card: dict = {}
        card_path = MODEL_STORE_DIR / package / "model_card.json"
        try:
            card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
        except ValueError:
            logging.warning("Unreadable model card %s", card_path)
        card = _public_card(package, card) or {}
        entries.append(
            ImmunoModelInfo(
                cytokine=cytokine,
                package=package,
                name=name,
                version=version,
                registered=ref is not None,
                checksum=ref.checksum if ref else None,
                card_status=card.get("status"),
                task=card.get("task"),
                threshold=card.get("threshold"),
                cv_metrics=card.get("cv_metrics_5fold_mean", {}),
                training=card.get("training", {}),
                domain=_immuno_domain(_immuno_weights_info(package)),
            )
        )
    return ImmunoModelsResponse(
        models=entries,
        caveat=(
            "Cross-validation figures estimate performance on peptides like the training data, at threshold 0.5. The domain ranges"
            " reflect peptide length, per-length training counts and, where shown, similarity to the training peptides; the embedding distribution is not checked."
        ),
    )


# Declared before the generic /{job_id:path}/... routes below: those would otherwise take "immunomodulation/<job>" as the job id and 422.
@app.get("/api/v1/jobs/immunomodulation/{job_id:path}/status", response_model=JobStatusResponse)
def get_immunomodulation_job_status(job_id: str):
    """Same envelope as the pipeline status route. While running, `stage` reads like `batch 3 of 12 (300/1200 peptides)`."""
    if DEV_MODE and job_id.startswith(LOCAL_JOB_PREFIX):
        return _local_immunomodulation_status(job_name(job_id))
    return get_job_status(job_id)


def _local_immunomodulation_status(name: str) -> JobStatusResponse:
    run_id = _run_id_from_job_id(name)
    status_path = storage.join(_local_artifacts(), "runs", run_id, "results.json")
    results = json.loads(storage.read_text(status_path)) if storage.exists(status_path) else None
    worker_status = (results.get("status") if results else None) or None
    process = _local_processes.get(run_id)
    if process is not None:
        code = process.poll()
        vertex_state = "JOB_STATE_RUNNING" if code is None else "JOB_STATE_SUCCEEDED" if code == 0 else "JOB_STATE_FAILED"
    elif worker_status == "success":
        vertex_state = "JOB_STATE_SUCCEEDED"
    elif worker_status in ("failed", "fail"):
        vertex_state = "JOB_STATE_FAILED"
    else:
        vertex_state = "JOB_STATE_RUNNING"  # unknown: no process handle and no verdict from the worker
    if run_id in _local_cancelled:
        vertex_state = "JOB_STATE_CANCELLED"  # a client asked for it to stop; the process may still be winding down
    return JobStatusResponse(
        job_id=name,
        status=worker_status or "pending",
        stage=(results.get("stage") if results else None) or "pending",
        vertex_state=vertex_state,
        run_state=run_state(worker_status, vertex_state),
        error=(results.get("error") if results else None),
    )


class CancelJobResponse(BaseModel):
    job_id: str
    status: Literal["cancelling", "cancelled", "failed"]


class ImmunoPeptideResult(BaseModel):
    sequence: str | None = None
    predictions: dict = Field(
        default_factory=dict,
        description=(
            "Keyed by cytokine (TNF-alpha, IL-6, IL-10, IL-4, IL-13, IFN-gamma). Each holds probability, label (probability >= 0.5),"
            " in_length_range, model (the package that produced it) and applicability {status, reasons, checks, reference, similarity, not_checked}."
        ),
    )


class ImmunoJobResultsResponse(BaseModel):
    pipeline: Literal["immunomodulation"] = "immunomodulation"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    results: list[ImmunoPeptideResult] = Field(default_factory=list)


@app.get("/api/v1/jobs/immunomodulation/{job_id:path}/results", response_model=ImmunoJobResultsResponse)
def get_immunomodulation_job_results(job_id: str):
    """Per-peptide predictions, in input order. Empty until the job has succeeded. Read each result's `applicability` before its probability."""
    name = job_name(job_id)
    run_id = _run_id_from_job_id(name)
    artifacts = _local_artifacts() if DEV_MODE and name.startswith(LOCAL_JOB_PREFIX) else required("VERTEX_ARTIFACTS_DIR")
    run_dir = storage.join(artifacts, "runs", run_id)
    status_path = storage.join(run_dir, "results.json")
    if not storage.exists(status_path):
        raise HTTPException(404, "Job results not found")
    status_payload = json.loads(storage.read_text(status_path))
    status = status_payload.get("status", "unknown")
    n_final = status_payload.get("n_final")
    results: list[dict] = []
    if status == "success":
        results_path = storage.join(run_dir, "immuno_results.json")
        if not storage.exists(results_path):
            raise HTTPException(404, "Job reported success but immuno_results.json is missing")
        results = json.loads(storage.read_text(results_path))
    return ImmunoJobResultsResponse(
        job_id=name,
        run_id=run_id,
        status=status,
        n_final=n_final,
        results=[
            ImmunoPeptideResult(
                sequence=r.get("sequence"),
                predictions={
                    cytokine: {k: v for k, v in prediction.items() if k != "members"}
                    for cytokine, prediction in (r.get("predictions") or {}).items()
                },
            )
            for r in results
        ],
    )


@app.post("/api/v1/jobs/immunomodulation/{job_id:path}/cancel", response_model=CancelJobResponse)
def cancel_immunomodulation_job(job_id: str):
    if DEV_MODE and job_id.startswith(LOCAL_JOB_PREFIX):
        name = job_name(job_id)
        process = _local_processes.get(_run_id_from_job_id(name))
        if process is None or process.poll() is not None:
            return {"job_id": name, "status": "failed"}  # not started by this API process, or already over
        process.terminate()
        _local_cancelled.add(_run_id_from_job_id(name))
        return {"job_id": name, "status": "cancelling"}
    return cancel_job(job_id)


class FoodCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, description="Caller's name for the run.")
    fasta: str | None = Field(
        default=None, min_length=1,
        description="Upload mode: the proteins as FASTA text. Every record is checked; a record that fails is reported in `rejected`. A record that is exactly a library protein gets the library's family and allergen-source flag.",
    )
    library_proteins: list[str] | None = Field(
        default=None, min_length=1,
        description="Library mode: substrate ids from GET /api/v1/food/library/proteins to run instead of a FASTA. Their family, allergen-source flag and release come from the library.",
    )
    food_source: str | None = Field(default=None, min_length=1, description="Upload mode: food source of the proteins whose header carries no FS=.")
    allergen_source: bool | None = Field(
        default=None, description="Upload mode: whether the food source is a known allergen source, for proteins the library does not match; null when not established."
    )

    @model_validator(mode="after")
    def _one_input(self):
        if (self.fasta is None) == (self.library_proteins is None):
            raise ValueError("send either fasta (upload mode) or library_proteins (library mode), not both and not neither")
        if self.fasta is not None and not self.food_source:
            raise ValueError("food_source is required with a fasta")
        if self.library_proteins is not None and (self.food_source or self.allergen_source is not None):
            raise ValueError("with library_proteins, food source and allergen status come from the library; do not send them")
        return self


class FoodRejection(BaseModel):
    id: str
    severity: str
    code: str
    message: str


class FoodLibraryMatch(BaseModel):
    id: str = Field(description="The uploaded record's id.")
    status: Literal["exact_sequence", "accession_sequence_differs", "not_in_library"]
    library_substrate_id: str | None = None
    protein_family: str | None = None
    allergen_source: bool | None = None
    message: str = ""


class FoodCreateResponse(CreateJobResponse):
    accepted: int = Field(description="Protein records that passed QC and enter the run.")
    rejected: list[FoodRejection] = Field(default_factory=list, description="QC issues: errors skip a record, warnings keep it.")
    input_mode: Literal["upload", "library"] = "upload"
    library_matches: list[FoodLibraryMatch] = Field(
        default_factory=list, description="Upload mode: each record against the library. A record that is not in the library is not rejected; it is reported here as not_in_library."
    )


class LibraryProteinInfo(BaseModel):
    substrate_id: str
    accession: str
    name: str
    food_source: str
    kind: str
    length: int
    protein_family: str | None = None
    organism: str | None = None
    allergen_source: bool | None = Field(default=None, description="The library's flag for the source protein; not a clinical finding.")
    sequence_version: str | None = None
    uniprot_release: str | None = None
    retrieved_at: str | None = None
    n_terminus: str | None = Field(default=None, description="State of the chain's N-terminal end: SIGNAL_CLEAVED, PROPEP_CLEAVED, TRANSIT_CLEAVED, INIT_MET_CLEAVED, CHAIN_JUNCTION, SEQ_START or UNANNOTATED.")
    c_terminus: str | None = Field(default=None, description="State of the chain's C-terminal end: PROPEP_CLEAVED, CHAIN_JUNCTION, SEQ_END or UNANNOTATED (same codes as n_terminus).")


class LibraryProteinsResponse(BaseModel):
    total: int = Field(description="Proteins matching the filters.")
    offset: int
    limit: int
    proteins: list[LibraryProteinInfo]


class LibrarySummaryResponse(BaseModel):
    manifest: dict
    total: int
    by_food_source: dict[str, int]
    families: list[dict]
    allergen_source_flags: dict[str, int]


DIGESTION_MAX_PROTEINS, DIGESTION_MAX_RESIDUES = 50, 30000


class DigestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fasta: str | None = Field(default=None, min_length=1, description="Upload mode: the proteins as FASTA text.")
    library_proteins: list[str] | None = Field(default=None, min_length=1, description="Library mode: substrate ids from GET /api/v1/food/library/proteins.")
    pathways: list[str] | None = Field(default=None, min_length=1, description="Pathway ids from GET /api/v1/food/digestion/settings; all of them when omitted.")
    max_missed_cleavages: int | None = Field(default=None, ge=0, le=4, description="Cleavage sites a fragment may contain; the default settings use 2.")
    min_length: int | None = Field(default=None, ge=1, le=60)
    max_length: int | None = Field(default=None, ge=1, le=60)
    limit: int = Field(default=500, ge=1, le=2000, description="Peptides returned per page.")
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_input(self):
        if (self.fasta is None) == (self.library_proteins is None):
            raise ValueError("send either fasta or library_proteins, not both and not neither")
        return self


class DigestionEnzyme(BaseModel):
    name: str
    pattern: str = Field(description="The rule as a regular expression on the P1 residue (Expasy PeptideCutter, as encoded in pyteomics).")
    ph_range: tuple[float, float]
    source: str
    description: str
    validation: str = Field(description="What the protease benchmark says about the rule; see docs/validation/food/protease_validation.md.")


class DigestionSettingsResponse(BaseModel):
    settings: dict = Field(description="The default digestion settings: pathways with their stages, enzymes and pH, and the length and missed-cleavage limits.")
    enzymes: list[DigestionEnzyme]
    limits: dict
    limitations: list[str]


class DigestionOrigin(BaseModel):
    substrate_id: str
    pathway: str
    stage: str
    enzymes: list[str]
    ph: float | None = None
    start_in_product: int
    end_in_product: int
    start_in_precursor: int
    end_in_precursor: int
    missed_cleavages_final: int
    n_term: str
    c_term: str
    is_limit_product: bool


class DigestionPeptide(BaseModel):
    peptide_id: str
    sequence: str
    length: int
    mono_mass: float
    avg_mass: float
    pathways: list[str]
    n_origins: int
    parents: list[str]
    is_limit_product: bool = Field(description="True if some pathway releases it with no further cleavage site inside it.")
    chemistry_notes: str
    origins: list[DigestionOrigin]


class DigestionSite(BaseModel):
    p1_position: int = Field(description="1-based position of the residue before the cut.")
    p1_residue: str
    p1p_residue: str
    cleaved_by: list[str] = Field(description="stage:enzyme labels of the rules that cut here.")


class DigestionCleavageMap(BaseModel):
    substrate_id: str
    pathway: str
    length: int
    sites: list[DigestionSite] = Field(description="The cut sites after the last stage of the pathway.")


class DigestionResponse(BaseModel):
    evidence_class: Literal["THEORETICAL"] = "THEORETICAL"
    settings: dict
    settings_warnings: list[str]
    rule_validation: dict[str, str]
    n_proteins: int
    n_peptides: int = Field(description="Unique peptides over all proteins and pathways.")
    n_origins: int
    offset: int
    limit: int
    truncated: bool = Field(description="True when more peptides exist than this page returns.")
    peptides: list[DigestionPeptide]
    cleavage_map: list[DigestionCleavageMap]
    limitations: list[str]


@app.get("/api/v1/food/digestion/settings", response_model=DigestionSettingsResponse)
def food_digestion_settings():
    """The digestion rules, pathways and limits, with what the benchmark says about each rule. The run endpoint accepts a subset of these pathways."""
    enzymes = [
        DigestionEnzyme(name=e.name, pattern=e.pattern, ph_range=e.ph_range, source=e.source, description=e.description, validation=RULE_VALIDATION.get(e.name, "not benchmarked"))
        for e in digestion_registry().values()
    ]
    return DigestionSettingsResponse(
        settings=DIGESTION_DEFAULT_SETTINGS.to_dict(), enzymes=enzymes, limitations=DIGESTION_LIMITATIONS,
        limits={"max_proteins": DIGESTION_MAX_PROTEINS, "max_residues": DIGESTION_MAX_RESIDUES, "max_peptides_per_page": 2000},
    )


@app.post("/api/v1/food/digestion/run", response_model=DigestionResponse)
def run_food_digestion(request: DigestionRequest):
    """Run the digestion engine now, on uploaded or library proteins, and return the theoretical fragments with their positions plus a cleavage map. It is the same engine and default
    settings as stage s03 of the food pipeline, so these fragments are the ones a pipeline run starts from. Synchronous: at most 50 proteins and 30,000 residues."""
    library = food_library()
    if request.library_proteins is not None:
        try:
            substrates = library.substrates(library.resolve(request.library_proteins))
        except UnknownLibraryProteins as exc:
            raise HTTPException(422, {"message": str(exc), "missing": exc.missing}) from exc
    else:
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "proteins.fasta"
            fasta.write_text(request.fasta, encoding="utf-8")
            try:
                substrates, issues = read_protein_fasta(fasta)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        if not substrates:
            raise HTTPException(422, {"message": "No usable protein records in the FASTA", "rejected": [{"id": i.substrate_id, "code": i.code, "message": i.message} for i in issues]})
    if len(substrates) > DIGESTION_MAX_PROTEINS or sum(len(s.sequence) for s in substrates) > DIGESTION_MAX_RESIDUES:
        raise HTTPException(422, f"At most {DIGESTION_MAX_PROTEINS} proteins and {DIGESTION_MAX_RESIDUES} residues per digestion run")
    base = DIGESTION_DEFAULT_SETTINGS
    known = {p.id: p for p in base.pathways}
    unknown = [i for i in (request.pathways or []) if i not in known]
    if unknown:
        raise HTTPException(422, f"Unknown pathway {', '.join(unknown)}; available: {', '.join(known)}")
    try:
        settings = dataclasses.replace(
            base, pathways=tuple(known[i] for i in dict.fromkeys(request.pathways)) if request.pathways else base.pathways,
            max_missed_cleavages=base.max_missed_cleavages if request.max_missed_cleavages is None else request.max_missed_cleavages,
            min_length=request.min_length or base.min_length, max_length=request.max_length or base.max_length,
        )
        settings.validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    origins, peptides, cleavage_map = run_digestion(substrates, settings)
    page = peptides[request.offset:request.offset + request.limit]
    by_peptide: dict[str, list[dict]] = {}
    for o in origins:
        by_peptide.setdefault(o["peptide_id"], []).append(o)
    stages = {(sub.substrate_id, pw.id): len(pw.stages) for sub in substrates for pw in settings.pathways}
    sites: dict[tuple[str, str], list[DigestionSite]] = {}
    for row in cleavage_map:
        if row["stage_index"] == stages[(row["substrate_id"], row["pathway"])]:
            sites.setdefault((row["substrate_id"], row["pathway"]), []).append(DigestionSite(
                p1_position=row["p1_position"], p1_residue=row["p1_residue"], p1p_residue=row["p1p_residue"], cleaved_by=row["cleaved_by"].split(";")))
    lengths = {s.substrate_id: len(s.sequence) for s in substrates}
    return DigestionResponse(
        settings=settings.to_dict(), settings_warnings=settings.ph_warnings(),
        rule_validation={e: RULE_VALIDATION.get(e, "not benchmarked") for e in sorted({e for pw in settings.pathways for st in pw.stages for e in st.enzymes})},
        n_proteins=len(substrates), n_peptides=len(peptides), n_origins=len(origins), offset=request.offset, limit=request.limit,
        truncated=request.offset + request.limit < len(peptides),
        peptides=[
            DigestionPeptide(
                peptide_id=p["peptide_id"], sequence=p["sequence"], length=p["length"], mono_mass=p["mono_mass"], avg_mass=p["avg_mass"], pathways=p["pathways"].split(";"),
                n_origins=p["n_origins"], parents=p["parent_substrates"].split(";"), is_limit_product=p["is_limit_product_any"], chemistry_notes=p["chemistry_notes"],
                origins=[DigestionOrigin(
                    substrate_id=o["substrate_id"], pathway=o["pathway"], stage=o["stage"], enzymes=o["stage_enzymes"].split("+"), ph=None if o["stage_ph"] == "" else float(o["stage_ph"]),
                    start_in_product=o["start_in_product"], end_in_product=o["end_in_product"], start_in_precursor=o["start_in_precursor"], end_in_precursor=o["end_in_precursor"],
                    missed_cleavages_final=o["missed_cleavages_final"], n_term=o["n_term"], c_term=o["c_term"], is_limit_product=o["is_limit_product"]) for o in by_peptide[p["peptide_id"]]],
            )
            for p in page
        ],
        cleavage_map=[DigestionCleavageMap(substrate_id=s.substrate_id, pathway=pw.id, length=lengths[s.substrate_id], sites=sites.get((s.substrate_id, pw.id), []))
                      for s in substrates for pw in settings.pathways],
        limitations=DIGESTION_LIMITATIONS,
    )


@app.get("/api/v1/food/library/summary", response_model=LibrarySummaryResponse)
def food_library_summary():
    """What the food protein library holds: its release, the proteins per food source and family, and the allergen-source flags."""
    return food_library().summary()


@app.get("/api/v1/food/library/proteins", response_model=LibraryProteinsResponse)
def food_library_proteins(food_source: str | None = None, family: str | None = None, q: str | None = None, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Browse the library. `q` matches accession, substrate id, protein name or family. Pass the `substrate_id` values to `food/create` as `library_proteins`."""
    total, page = food_library().search(food_source, family, q, limit, offset)
    return {"total": total, "offset": offset, "limit": limit, "proteins": [p.info() for p in page]}


@dataclass
class FoodInput:
    fasta: str
    records_csv: str | None  # source records the library supplies for these proteins
    accepted: int
    rejected: list[FoodRejection]
    mode: str
    matches: list[LibraryMatchResult]
    library_proteins: list[str]


def _prepare_food_input(request: FoodCreateRequest) -> FoodInput:
    """Library mode: the chosen proteins and their records. Upload mode: QC with the reader the worker uses (422 when no record is usable), then each record against the library.
    Standard library only, so the API does not import the pipeline."""
    library = food_library()
    if request.library_proteins is not None:
        try:
            chosen = library.resolve(request.library_proteins)
        except UnknownLibraryProteins as exc:
            raise HTTPException(422, {"message": str(exc), "missing": exc.missing}) from exc
        return FoodInput(library.fasta_text(chosen), library.records_csv(chosen), len(chosen), [], "library", [], [p.substrate_id for p in chosen])
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "proteins.fasta"
        fasta.write_text(request.fasta, encoding="utf-8")
        try:
            proteins, issues = read_protein_fasta(fasta)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    rejected = [FoodRejection(id=i.substrate_id, severity=i.severity, code=i.code, message=i.message) for i in issues]
    if not proteins:
        raise HTTPException(422, {"message": "No usable protein records in the FASTA", "rejected": [r.model_dump() for r in rejected]})
    matches = library.match_upload(proteins)
    return FoodInput(request.fasta, library.matched_records_csv(proteins, matches), len(proteins), rejected, "upload", matches, [])


def _store_food_input(run_dir: str, fi: FoodInput, request: FoodCreateRequest) -> FoodRunConfig:
    """Write the FASTA, the library's source records and the library match record into the run folder; return the run config that points at them."""
    library = food_library()
    storage.write_text(storage.join(run_dir, "proteins.fasta"), fi.fasta)
    records_path = ""
    if fi.records_csv:
        records_path = storage.join(run_dir, "source_records.csv")
        storage.write_text(records_path, fi.records_csv)
    storage.write_text(
        storage.join(run_dir, "library_match.json"),
        json.dumps({"mode": fi.mode, "library": {k: library.manifest[k] for k in ("name", "uniprot_release", "retrieved_at")}, "library_proteins": fi.library_proteins,
                    "matches": [m.__dict__ for m in fi.matches]}),
    )
    return FoodRunConfig(fasta_path=storage.join(run_dir, "proteins.fasta"), source_records_path=records_path, food_source=request.food_source or "",
                         allergen_source=request.allergen_source, run_id="", artifacts_dir="", model_store="", stages={})


def _food_response(request: FoodCreateRequest, fi: FoodInput, job_id: str, result_path: str) -> FoodCreateResponse:
    return FoodCreateResponse(request_id=request.request_id, job_id=job_id, result_path=result_path, accepted=fi.accepted, rejected=fi.rejected, input_mode=fi.mode,
                              library_matches=[FoodLibraryMatch(**m.__dict__) for m in fi.matches])


def _create_local_food_job(request: FoodCreateRequest, fi: FoodInput) -> FoodCreateResponse:
    """DEV_MODE: run the food worker as a subprocess with local weights; its output goes to <run dir>/worker.log."""
    artifacts = _local_artifacts()
    run_id = str(time.time_ns() // 1_000_000)
    run_dir = storage.join(artifacts, "runs", run_id)
    config = _store_food_input(run_dir, fi, request).model_copy(update={"run_id": run_id, "artifacts_dir": artifacts, "model_store": "model_store"})
    config_path = storage.join(run_dir, "config.json")
    storage.write_text(config_path, config.model_dump_json(indent=2))
    env = {**os.environ, "DEV_MODE": "true", "PYTHONPATH": os.pathsep.join([str(_REPO / "src"), str(_REPO), os.environ.get("PYTHONPATH", "")])}
    with open(Path(run_dir) / "worker.log", "ab") as log:
        _local_processes[run_id] = subprocess.Popen(
            [sys.executable, "-m", "backend.worker_food.worker", "--config", config_path],
            cwd=str(_REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return _food_response(request, fi, LOCAL_JOB_PREFIX + run_id, run_dir)


def _create_vertex_food_job(request: FoodCreateRequest, fi: FoodInput) -> FoodCreateResponse:
    """Submit the food worker as a Vertex AI Custom Job. The input and the run config go to the run folder; the worker waits for its staging config."""
    parent = f"projects/{required('VERTEX_CLOUD_PROJECT')}/locations/{required('VERTEX_LOCATION')}"
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    models = required("VERTEX_MODEL_STORE")
    image = required("WORKER_FOOD_IMAGE_URI")
    service_account = required("WORKER_SERVICE_ACCOUNT")
    machine_type = required("MACHINE_TYPE")
    accelerator_type = required("ACCELERATOR_TYPE")
    accelerator_count = int(required("ACCELERATOR_COUNT"))
    if not all(storage.is_gcs_path(p) for p in (artifacts, models)):
        raise HTTPException(503, "Artifacts and model store must use gs:// paths")
    # The worker runs with DEV_MODE=false, so it downloads every model from this bucket: refuse now, not minutes into a GPU job.
    if "TODO-bucket" in models:
        raise HTTPException(503, "VERTEX_MODEL_STORE is still the placeholder; set it to the bucket that holds the model weights")
    missing = [f for f in FOOD_REQUIRED_WEIGHT_FILES if not storage.exists(storage.join(models, f))]
    if missing:
        raise HTTPException(503, f"Model weights missing from {models}: {', '.join(missing)}")

    # Its own staging folder, so a request_id shared with another pipeline cannot return that pipeline's job.
    staging_config_path = storage.join(artifacts, "requests", "food", request.request_id, "config.json")

    # A retry returns the job the first attempt started rather than submitting a second one; see create_job for why.
    if storage.exists(staging_config_path):
        try:
            run_id = json.loads(storage.read_text(staging_config_path))["run_id"]
        except Exception:
            logging.exception("Staging config %s exists but could not be read; treating %s as new", staging_config_path, request.request_id)
        else:
            return _food_response(request, fi, f"{parent}/customJobs/{run_id}", storage.join(artifacts, "runs", run_id))

    custom_job = {
        "display_name": f"ai-peptide-discovery-food-{request.request_id[:12]}",
        "labels": {"service": "battery-ai", "pipeline": "food", "request-id": _label_value(request.request_id)},
        "job_spec": {
            "service_account": service_account,
            "worker_pool_specs": [
                {
                    "replica_count": 1,
                    "machine_spec": {"machine_type": machine_type, "accelerator_type": accelerator_type, "accelerator_count": accelerator_count},
                    "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                    "container_spec": {
                        "image_uri": image,
                        "args": ["--config", staging_config_path, "--wait-for-config", "120"],
                        "env": [{"name": "DEV_MODE", "value": "false"}, {"name": "VERTEX_MODEL_STORE", "value": models}],
                    },
                }
            ],
        },
    }
    try:
        client = job_client()
        job = client.create_custom_job(parent=parent, custom_job=custom_job)
        job_id = job.name.rsplit("/", 1)[-1]
        result_path = storage.join(artifacts, "runs", job_id)
        try:
            config = _store_food_input(result_path, fi, request).model_copy(update={"run_id": job_id, "artifacts_dir": artifacts, "model_store": models})
            body = config.model_dump_json(indent=2)
            storage.write_text(storage.join(result_path, "config.json"), body)
            storage.write_text(staging_config_path, body)
        except Exception as exc:
            try:
                client.cancel_custom_job(name=job.name, timeout=30)
            except Exception:
                logging.exception("Could not cancel job %s after config upload failed", job.name)
            raise HTTPException(502, f"Config upload failed for {job.name}; cancellation requested") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return _food_response(request, fi, job.name, result_path)


@app.post("/api/v1/jobs/food/create", response_model=FoodCreateResponse, status_code=202)
def create_food_job(request: FoodCreateRequest):
    """Start a food-peptide run: digestion, generation, screening, gut stability, ranking and a provenance check.

    Two input modes. **Upload**: send `fasta` and `food_source`; every record is checked (a record that fails is reported in `rejected`, and a FASTA with no usable record is a 422),
    and each record is compared with the food protein library: a protein that is not in the library is not rejected, it is reported as `not_in_library` and keeps what you supplied.
    **Library**: send `library_proteins` (substrate ids from `GET /api/v1/food/library/proteins`); their family, allergen-source flag and release come from the library.
    With DEV_MODE the job runs as a local process on local weights, otherwise as a Vertex AI Custom Job."""
    fi = _prepare_food_input(request)
    return _create_local_food_job(request, fi) if DEV_MODE else _create_vertex_food_job(request, fi)


@app.post("/api/v1/jobs/food/{job_id:path}/cancel", response_model=CancelJobResponse)
def cancel_food_job(job_id: str):
    """Stop a running food job. `cancelling` while it winds down, `failed` when it had already ended; the status route then reads `cancelled`."""
    if DEV_MODE and job_id.startswith(LOCAL_JOB_PREFIX):
        name = job_name(job_id)
        run_id = _run_id_from_job_id(name)
        process = _local_processes.get(run_id)
        if process is None or process.poll() is not None:
            return {"job_id": name, "status": "failed"}  # not started by this API process, or already over
        process.terminate()
        _local_cancelled.add(run_id)
        return {"job_id": name, "status": "cancelling"}
    return cancel_job(job_id)


# Declared before the generic /{job_id:path}/status route below, as the immunomodulation route is.
@app.get("/api/v1/jobs/food/{job_id:path}/status", response_model=JobStatusResponse)
def get_food_job_status(job_id: str):
    """Same envelope as the pipeline status route; `stage` names the food stage that is running."""
    if DEV_MODE and job_id.startswith(LOCAL_JOB_PREFIX):
        return _local_immunomodulation_status(job_name(job_id))
    return get_job_status(job_id)


class FoodRankedPeptide(BaseModel):
    id: str
    sequence: str
    rank: int
    origin_type: str
    evidence_class: str
    shortlist_position: int | None = Field(default=None, description="Position in the biology shortlist; null when the peptide is not on it.")
    pareto_front: int | None = Field(default=None, description="Pareto front of the peptide in the model-stage pool (1: no pooled peptide beats it on every objective). Separate from rank and shortlist_position.")
    objective_values: dict[str, float | None] = Field(default_factory=dict, description="The peptide's value on each objective the run used (0 to 1, higher is the favourable outcome; for a model it is the ranking value, not the model's own output). Empty when the run did not record its objectives.")
    shortlist_reason: str | None = Field(default=None, description="Why the peptide is on the shortlist or not: not eligible (its evidence), left off (a model calls it unfavourable), skipped (same family or designed-peptide limit) or not needed (list full).")
    parents: list[dict] = Field(default_factory=list, description="Every protein the peptide comes from, with position, pathway, enzymes and pH.")
    parent_clusters: list[str] = Field(default_factory=list)
    any_parent_allergen_source: bool | None = Field(default=None, description="True if a parent is an allergen source, false if none is, null if unknown.")
    digestion_product_type: str | None = Field(default=None, description="FINAL_PRODUCT or INTERMEDIATE_PRODUCT for a digestion-derived peptide; null for a designed peptide.")
    scores: dict = Field(default_factory=dict, description="Objectives the ranking compared: safety (the safety models other than IgE binding), developability, digestion_plausibility, uncut_bond_fraction, evidence, coverage. uncut_bond_fraction does not measure survival in the gut.")
    models: dict = Field(default_factory=dict, description="Per screening model: score, label, applicability, detail. A model that did not run is absent and named in `predictions_not_run`.")
    applicability: dict = Field(default_factory=dict, description="Per dimension and overall, worst case; read before the scores.")
    cleavage: dict = Field(default_factory=dict, description="Cut sites per protease from the Expasy rules: counts, the predicted positions per protease, the uncut-bond fraction and a statement of what it does not measure.")
    evidence: dict = Field(default_factory=dict, description="Evidence for the candidate: evidence_type (EXACT_SEQUENCE, CONTAINS_REFERENCE, CONTAINED_IN_REFERENCE, SEQUENCE_SIMILARITY or NO_EVIDENCE), the evidence peptide and its source, and whether the evidence gate passed; for a designed peptide the template's evidence under template_* keys, never transferred to the candidate.")
    predictions_not_run: list[str] = Field(default_factory=list)
    glp1_classifier_status: str | None = None
    rationale: str = ""
    limitations: str = ""
    review: dict | None = Field(default=None, description="The biology lead's current decision on this peptide (decision, reason, reviewer, decided_at); null until they decide. It never changes the rank.")


class FoodJobResultsResponse(BaseModel):
    pipeline: Literal["food"] = "food"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    peptides: list[FoodRankedPeptide] = Field(default_factory=list, description="Ranked peptides, best first.")
    shortlist: list[FoodRankedPeptide] = Field(default_factory=list, description="The ranked peptides on the biology shortlist, in shortlist order.")
    controls: list[dict] = Field(default_factory=list, description="Reference positive and negative controls proposed by rule; the biology lead finalizes them.")
    gate: dict = Field(default_factory=dict, description="Evidence gate and diversity report of the selection, including any shortfall.")
    facts: dict = Field(default_factory=dict)
    protein_qc: list[dict] = Field(default_factory=list, description="FASTA records that were skipped or warned about.")
    clusters: dict = Field(default_factory=dict, description="How each protein's cluster was assigned: notes and the number of proteins per source.")
    library: dict = Field(default_factory=dict, description="How the proteins relate to the food protein library: the mode (upload or library), the library release, and for an upload each record's match (exact_sequence, accession_sequence_differs or not_in_library).")
    ranking_summary: dict = Field(default_factory=dict, description="How the run narrowed its candidates: the stage counts, the hard rules and what they removed, the objectives, the selection and tie-break rules, the evidence gate and the caps, with the peptides each dropped. A value an older run did not record is null.")
    digestion_warnings: list[dict] = Field(default_factory=list, description="Where the digestion run again disagrees with a recorded origin of a ranked peptide; a warning only, it never fails a run.")
    review_gate: dict = Field(default_factory=dict, description="The biological approval gate: its status, the approved, on-hold, rejected and pending shortlisted peptides, and the rule that ranks are never changed by the review.")
    provenance: dict = Field(default_factory=dict, description="Outcome of the provenance check of the final peptides: how many were checked and any issue. A run with an issue fails, so it is normally empty.")


_FOOD_SCORE_KEYS = ("safety", "developability", "digestion_plausibility", "uncut_bond_fraction", "evidence", "coverage", "pareto_front", "cluster")
_FOOD_EVIDENCE_KEYS = ("evidence_type", "evidence_gate_passed", "evidence_applies_to", "evidence_direction", "evidence_peptide", "evidence_similarity", "evidence_study", "evidence_assay",
                       "evidence_food_derivation", "evidence_tier", "nearest_positive", "nearest_positive_sim", "nearest_negative", "nearest_negative_sim", "prior_art_alert",
                       "template_evidence_applies_to", "template_evidence_type", "template_evidence_direction", "template_evidence_peptide", "template_evidence_similarity",
                       "template_evidence_study", "template_evidence_assay", "template_evidence_food_derivation", "template_evidence_tier")


def _food_peptide(candidate: dict, review: dict | None = None, objectives: list[str] | None = None) -> FoodRankedPeptide:
    preds = candidate.get("predictions") or {}
    ranking, screening, cleavage = preds.get("ranking") or {}, preds.get("screening") or {}, preds.get("cleavage") or {}
    return FoodRankedPeptide(
        id=candidate["id"],
        sequence=candidate["sequence"],
        rank=ranking["rank"],
        origin_type=candidate["origin_type"],
        evidence_class=candidate["evidence_class"],
        digestion_product_type=candidate.get("digestion_product_type"),
        shortlist_position=ranking.get("shortlist_position"),
        pareto_front=ranking.get("pareto_front"),
        objective_values={o: ranking.get(o, screening.get(o)) for o in objectives or []},
        shortlist_reason=ranking.get("shortlist_reason"),
        parents=candidate.get("parents") or [],
        parent_clusters=candidate.get("parent_clusters") or [],
        any_parent_allergen_source=candidate.get("any_parent_allergen_source"),
        scores={k: ranking.get(k, cleavage.get(k)) for k in _FOOD_SCORE_KEYS if ranking.get(k, cleavage.get(k)) is not None},
        models=preds.get("models") or {},
        applicability={k.removeprefix("applicability_"): v for k, v in ranking.items() if k.startswith("applicability_")},
        cleavage=cleavage,
        evidence={k: ranking.get(k, screening.get(k)) for k in _FOOD_EVIDENCE_KEYS if ranking.get(k, screening.get(k)) is not None},
        predictions_not_run=[n for n in (ranking.get("predictions_not_run") or "").split(";") if n],
        glp1_classifier_status=ranking.get("glp1_classifier_status"),
        rationale=ranking.get("rationale", ""),
        limitations=ranking.get("limitations", ""),
        review=review,
    )


def _nan_to_none(value):
    """Run files are written with NaN for a missing number; the response must be valid JSON, so those become null."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _read_json_if_present(path: str, default):
    return _nan_to_none(json.loads(storage.read_text(path))) if storage.exists(path) else default


# Declared before the generic /{job_id:path}/... routes, as the status route above.
@app.get("/api/v1/jobs/food/{job_id:path}/results", response_model=FoodJobResultsResponse)
def get_food_job_results(job_id: str):
    """The ranked peptides of a finished food run with their traceability, applicability and limitations, the shortlist and the controls. Empty until the job has succeeded."""
    name = job_name(job_id)
    run_id = _run_id_from_job_id(name)
    artifacts = _local_artifacts() if DEV_MODE and name.startswith(LOCAL_JOB_PREFIX) else required("VERTEX_ARTIFACTS_DIR")
    run_dir = storage.join(artifacts, "runs", run_id)
    status_path = storage.join(run_dir, "results.json")
    if not storage.exists(status_path):
        raise HTTPException(404, "Job results not found")
    status_payload = json.loads(storage.read_text(status_path))
    status = status_payload.get("status", "unknown")
    response = FoodJobResultsResponse(job_id=name, run_id=run_id, status=status, n_final=status_payload.get("n_final"))
    if status != "success":
        return response
    ranked_path = storage.join(run_dir, "candidates", "s07_food_ranking_decision.jsonl")
    if not storage.exists(ranked_path):
        raise HTTPException(404, "Job reported success but the ranked candidates are missing")
    current = latest(_food_review_log(run_dir))
    report = _read_json_if_present(storage.join(run_dir, "ranking_report.json"), {})
    objectives = (report.get("facts") or {}).get("objectives_used")
    peptides = [_food_peptide(c, current.get(c["id"]), objectives) for c in (_nan_to_none(json.loads(line)) for line in storage.read_text(ranked_path).splitlines() if line.strip())]
    clusters = _read_json_if_present(storage.join(run_dir, "protein_clusters.json"), {})
    sources: dict[str, int] = {}
    for assigned in (clusters.get("clusters") or {}).values():
        sources[assigned["source"]] = sources.get(assigned["source"], 0) + 1
    response.peptides = peptides
    response.shortlist = sorted((p for p in peptides if p.shortlist_position is not None), key=lambda p: p.shortlist_position)
    response.review_gate = gate_summary([p.id for p in response.shortlist], current)
    response.controls, response.gate, response.facts = report.get("controls", []), report.get("gate", {}), report.get("facts", {})
    response.protein_qc = _read_json_if_present(storage.join(run_dir, "protein_qc.json"), [])
    response.clusters = {"notes": clusters.get("notes", []), "proteins_per_source": sources}
    response.provenance = _read_json_if_present(storage.join(run_dir, "provenance_check.json"), {})
    response.ranking_summary = _food_run_summary(run_dir, len(response.shortlist))
    response.digestion_warnings = report.get("digestion_warnings", [])
    response.library = _read_json_if_present(storage.join(run_dir, "library_match.json"), {})
    return response


def _food_run_summary(run_dir: str, n_shortlisted: int | None = None) -> dict:
    audit_path = storage.join(run_dir, "audit_log.jsonl")
    audit = [json.loads(line) for line in storage.read_text(audit_path).splitlines() if line.strip()] if storage.exists(audit_path) else []
    return ranking_summary(_read_json_if_present(storage.join(run_dir, "ranking_report.json"), {}), audit, n_shortlisted)


def _finished_food_run(job_id: str) -> tuple[str, str]:
    """(job name, run directory) of a food run that succeeded; 404 when the run is unknown or not finished."""
    name = job_name(job_id)
    run_id = _run_id_from_job_id(name)
    artifacts = _local_artifacts() if DEV_MODE and name.startswith(LOCAL_JOB_PREFIX) else required("VERTEX_ARTIFACTS_DIR")
    run_dir = storage.join(artifacts, "runs", run_id)
    status_path = storage.join(run_dir, "results.json")
    if not storage.exists(status_path):
        raise HTTPException(404, "Job results not found")
    if json.loads(storage.read_text(status_path)).get("status") != "success":
        raise HTTPException(404, "The job has not succeeded; there is no dossier yet")
    return name, run_dir


def _food_ranked_candidates(run_dir: str) -> list[dict]:
    ranked_path = storage.join(run_dir, "candidates", "s07_food_ranking_decision.jsonl")
    if not storage.exists(ranked_path):
        raise HTTPException(404, "Job reported success but the ranked candidates are missing")
    return [_nan_to_none(json.loads(line)) for line in storage.read_text(ranked_path).splitlines() if line.strip()]


def _food_controls(run_dir: str) -> list[dict]:
    return _read_json_if_present(storage.join(run_dir, "ranking_report.json"), {}).get("controls", [])


def _food_review_log(run_dir: str) -> list[ReviewRecord]:
    path = storage.join(run_dir, "review", "decisions.jsonl")
    return parse_log(storage.read_text(path)) if storage.exists(path) else []


def _food_reviews(run_dir: str) -> dict[str, dict]:
    """The biology lead's current decision on each peptide, by peptide id."""
    return latest(_food_review_log(run_dir))


# Declared before the generic /{job_id:path}/... routes, as the status route above.
@app.get("/api/v1/jobs/food/{job_id:path}/dossiers/{peptide_id}", response_model=FoodDossier)
def get_food_dossier(job_id: str, peptide_id: str, format: Literal["json", "html", "pdf"] = "json"):
    """The review dossier of one ranked peptide: provenance, evidence, screening with applicability, gut stability, limitations, controls and an empty assay plan. `format=html` returns a readable page, `format=pdf` a PDF: an overview page, the shared-information page, then one page for the peptide."""
    _, run_dir = _finished_food_run(job_id)
    candidate = next((c for c in _food_ranked_candidates(run_dir) if c["id"] == peptide_id), None)
    if candidate is None:
        raise HTTPException(404, f"{peptide_id} is not among the ranked peptides of this run")
    dossier = build_dossier(candidate, _food_controls(run_dir), _food_reviews(run_dir).get(peptide_id), _food_run_summary(run_dir))
    if format == "pdf":
        return _pdf_response(dossier_pdf([dossier]), f"dossier_{dossier.sequence}_{dossier.peptide_id}.pdf")
    return HTMLResponse(dossier_html(dossier)) if format == "html" else dossier


def _pdf_response(body: bytes, filename: str) -> Response:
    return Response(body, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/v1/jobs/food/{job_id:path}/shortlist-dossiers.pdf")
def get_food_shortlist_dossiers_pdf(job_id: str):
    """One PDF for the biology lead: an overview page of the shortlisted peptides, a page with what is the same for every peptide (controls, assay-plan cells, shared limitations), then one page per shortlisted peptide, in shortlist order."""
    name, run_dir = _finished_food_run(job_id)
    controls = _food_controls(run_dir)
    shortlisted = sorted((c for c in _food_ranked_candidates(run_dir) if c["predictions"]["ranking"].get("shortlist_position") is not None),
                         key=lambda c: c["predictions"]["ranking"]["shortlist_position"])
    if not shortlisted:
        raise HTTPException(404, "This run has no shortlisted peptides")
    reviews, summary = _food_reviews(run_dir), _food_run_summary(run_dir, len(shortlisted))
    return _pdf_response(dossier_pdf([build_dossier(c, controls, reviews.get(c["id"]), summary) for c in shortlisted]), f"shortlist_dossiers_{_run_id_from_job_id(name)}.pdf")


@app.get("/api/v1/jobs/food/{job_id:path}/experiment-sheet")
def get_food_experiment_sheet(job_id: str):
    """CSV for the biology review: the shortlisted peptides in shortlist order, then the reference controls. Assay design, decision, reviewer and comments are left empty."""
    name, run_dir = _finished_food_run(job_id)
    body = experiment_sheet_csv(_food_ranked_candidates(run_dir), _food_controls(run_dir), _food_reviews(run_dir))
    return Response(body, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="experiment_sheet_{_run_id_from_job_id(name)}.csv"'})


class FoodReviewLogResponse(BaseModel):
    gate: dict = Field(description="The biological approval gate: status, approved, on-hold, rejected and pending shortlisted peptides, and the rule that ranks are never changed by the review.")
    decisions: list[ReviewRecord] = Field(description="The current decision on each reviewed peptide.")
    history: list[ReviewRecord] = Field(description="Every decision ever saved, oldest first. Nothing is overwritten.")


class FoodReviewSavedResponse(BaseModel):
    record: ReviewRecord
    gate: dict


def _food_shortlisted_ids(candidates: list[dict]) -> list[str]:
    ranked = [c for c in candidates if (c.get("predictions") or {}).get("ranking", {}).get("shortlist_position") is not None]
    return [c["id"] for c in sorted(ranked, key=lambda c: c["predictions"]["ranking"]["shortlist_position"])]


@app.get("/api/v1/jobs/food/{job_id:path}/review", response_model=FoodReviewLogResponse)
def get_food_review(job_id: str):
    """The biological approval gate of a finished run: where it stands, the current decision on each reviewed peptide, and the full history of decisions."""
    _, run_dir = _finished_food_run(job_id)
    records = _food_review_log(run_dir)
    return FoodReviewLogResponse(gate=gate_summary(_food_shortlisted_ids(_food_ranked_candidates(run_dir)), latest(records)),
                                 decisions=[ReviewRecord(**r) for r in latest(records).values()], history=records)


@app.put("/api/v1/jobs/food/{job_id:path}/review/{peptide_id}", response_model=FoodReviewSavedResponse)
def put_food_review(job_id: str, peptide_id: str, request: ReviewRequest):
    """Save the biology lead's decision (approve, hold or reject), the reason and the reviewer for one ranked peptide. The decision is appended to the run's review log; earlier decisions stay
    in the history and the latest one is current. The ranking is never changed by a review. There is no sign-in yet, so the reviewer is the name sent."""
    _, run_dir = _finished_food_run(job_id)
    candidates = _food_ranked_candidates(run_dir)
    candidate = next((c for c in candidates if c["id"] == peptide_id), None)
    if candidate is None:
        raise HTTPException(404, f"{peptide_id} is not among the ranked peptides of this run")
    record = new_record(peptide_id, candidate["sequence"], request)
    storage.ensure_dir(storage.join(run_dir, "review"))
    storage.append_text(storage.join(run_dir, "review", "decisions.jsonl"), record.model_dump_json() + "\n")
    return FoodReviewSavedResponse(record=record, gate=gate_summary(_food_shortlisted_ids(candidates), _food_reviews(run_dir)))


def run_state(worker_status: str | None, vertex_state: str | None) -> str:
    """What the run actually is, from the worker's self-report AND the live Vertex state.

    A NAMED FUNCTION ON PURPOSE. This rule was written inline inside get_job_status(), and
    web/src/api/peptide.ts carries a second implementation of it whose comment says it "mirrors
    run_state() in src/backend/api_e2e/api.py exactly" -- naming something that was not there.
    Two copies of a rule in two languages need at least one of them to be a thing you can point
    at, test directly, and diff by eye against the other. web/src/test/runstate.test.ts holds the
    TypeScript copy to the same table.

    THE ORDER IS THE RULE, not an accident of how it was written:
      - the worker's own verdict wins when it has one, because it is the only thing that knows
        whether results were written;
      - then Vertex's terminal states, which are the only thing that knows the machine is gone;
      - `abandoned` has NO status condition, deliberately. A worker that reported progress and was
        then killed leaves status at "running" forever, and an earlier version checked
        `status in (None, "pending")` here, which sent that case to `running` -- the UI showing a
        finished job as still working, which is the defect this reconciliation exists to end.
    """
    terminal = vertex_state in TERMINAL_VERTEX_STATES
    if worker_status == "success":
        return "succeeded"
    if worker_status in ("failed", "fail"):
        return "failed"
    if vertex_state == "JOB_STATE_CANCELLED":
        return "cancelled"
    if vertex_state in ("JOB_STATE_FAILED", "JOB_STATE_EXPIRED"):
        return "failed"
    if terminal:
        return "abandoned"
    if worker_status == "running":
        return "running"
    return "submitted"


@app.get("/api/v1/jobs/{job_id:path}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    name = job_name(job_id)
    try:
        job = job_client().get_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    vertex_state = job.state.name

    artifacts = required("VERTEX_ARTIFACTS_DIR")
    run_id = _run_id_from_job_id(job_id)
    run_dir = storage.join(artifacts, "runs", run_id)
    status_path = storage.join(run_dir, "results.json")

    results = (
        json.loads(storage.read_text(status_path))
        if storage.exists(status_path)
        else None
    )

    worker_status = (results.get("status") if results else None) or None

    return JobStatusResponse(
        job_id=job.name,
        status=worker_status or "pending",
        stage=(results.get("stage") if results else None) or "pending",
        vertex_state=vertex_state,
        run_state=run_state(worker_status, vertex_state),
        error=job.error.message or None,
    )


class ComponentStats(BaseModel):
    count: int
    mean: float
    median: float
    min: float
    max: float


class CandidateResponse(BaseModel):
    id: str = "?"
    sequence: str | None = None  # 'sequence' key - sibling of predictions
    final_score_rank: int | None = None  # 'ranking.final_score_rank' key in predictions
    pareto_rank: int | None = None  # 'ranking.pareto_rank' key in predictions
    final_score: float | None = None  # 'ranking.final_score' key in predictions
    evidence_coverage: float | None = None  # 'ranking.evidence_coverage' key
    missing_modules: list[str] | None = None  # 'ranking.missing_modules' key
    amp_probability: float | None = None  # 'amp_prob' key in predictions
    hemolytic_activity_phc50: float | None = (
        None  # 'hemolysis_phc50' key in predictions
    )
    molecular_weight: float | None = (
        None  # 'molecular_weight' key in predictions (unit: Daltons)
    )
    net_charge: float | None = None  # 'net_charge' key in predictions
    instability_index: float | None = None  # 'instability_index' key in predictions
    deamidation_risk: Literal["low", "medium", "high"] | None = (
        None  # deamidation_risk.risk_category
    )
    oxidation_risk: Literal["low", "medium", "high"] | None = (
        None  # oxidation_risk.risk_category
    )
    solubility: float | None = None  # 'solubility.score' key in predictions
    aggregation_tendency: float | None = (
        None  # 'aggregation_tendency.score' key in predictions
    )
    anti_inflammatory_probability: float | None = (
        None  # 'anti_inflammatory_prob' key in predictions
    )
    angiogenic_activity: float | None = (
        None  # 'angiogenic_activity.angiogenic' key in predictions
    )
    proliferation_probability: float | None = (
        0  # proliferation_migration.proliferation in predictions
    )
    migration_probability: float | None = (
        0  # proliferation_migration.migration in predictions
    )
    cytotoxicity_probability: float | None = (
        None  # 'cytotoxicity.score' key in predictions
    )
    cleavage_stability: float | None = None  # cleavage_stability.score in predictions
    log_mic_um: dict[
        Literal["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"],
        float,
    ]  # 'mic.log_mic_um' key in predictions
    pmbic: dict[
        Literal[
            "Acinetobacter baumannii",
            "Candida albicans",
            "Candida tropicalis",
            "Cutibacterium acnes",
            "Enterococcus faecium",
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Pseudomonas aeruginosa",
            "Salmonella enterica",
            "Staphylococcus aureus",
            "Staphylococcus epidermidis",
            "Streptococcus mutans",
            "Streptococcus sanguinis",
        ],
        float,
    ]
    activated_pathways: list[str] | None = (
        None  # 'mechanism.activated_pathways' key in predictions
    )
    undetermined_pathways: list[str] | None = (
        None  # 'mechanism.undetermined_pathways' key in predictions
    )
    inhibited_pathways: list[str] | None = (
        None  # 'mechanism.inhibited_pathways' key in predictions
    )


Receptor = Literal["GLP1R", "GIPR", "GCGR"]


class MetabolicCandidateResponse(BaseModel):
    """A metabolic candidate: the ranks and shared physicochemical and safety values, plus the receptor predictions."""

    id: str = "?"
    sequence: str | None = None
    final_score_rank: int | None = None  # 'ranking.final_score_rank' key in predictions
    pareto_rank: int | None = None  # 'ranking.pareto_rank' key in predictions
    final_score: float | None = None  # 'ranking.final_score' key in predictions
    evidence_coverage: float | None = None  # 'ranking.evidence_coverage' key
    missing_modules: list[str] | None = None  # 'ranking.missing_modules' key
    pec50: dict[Receptor, float | None] = Field(default_factory=dict)  # predicted potency, -log10(EC50 [M])
    ec50_nm: dict[Receptor, float | None] = Field(default_factory=dict)  # the same potency as EC50 in nM
    binding_pkd: dict[Receptor, float | None] = Field(default_factory=dict)  # predicted binding, 9 - log10(Kd [nM])
    agonism_probability: dict[Receptor, float | None] = Field(default_factory=dict)  # P(EC50 < 1 nM)
    pec50_pkd_gap: float | None = None  # disagreement of the pEC50 and pKd models; Route A candidates only
    #: Receptors among this run's targets whose pec50/ec50_nm are returned but carry no ranking
    #: information (the potency model's leave-one-cluster-out R2 is below zero for them). Empty
    #: unless the run targeted one. Read it the way you would read `not_screened`: the number is
    #: present, the evidence is not.
    potency_unreliable_receptors: list[str] = Field(default_factory=list)
    potency_unreliable_note: str | None = None
    metabolic_activity_score: float | None = None  # Stage 6 combined potency score
    anti_inflammatory_probability: float | None = None
    half_life_minutes: float | None = None  # 'plasma_half_life.minutes' key in predictions
    hemolytic_activity_phc50: float | None = None  # 'hemolysis.phc50' key in predictions
    p_hemolytic: float | None = None  # 'hemolysis.p_hemolytic' key in predictions
    cleavage_stability: float | None = None  # cleavage_stability.score in predictions
    molecular_weight: float | None = None  # Daltons
    net_charge: float | None = None
    instability_index: float | None = None
    deamidation_risk: Literal["low", "medium", "high"] | None = None
    oxidation_risk: Literal["low", "medium", "high"] | None = None
    solubility: float | None = None
    aggregation_tendency: float | None = None
    cytotoxicity_probability: float | None = None
    activated_pathways: list[str] | None = None
    undetermined_pathways: list[str] | None = None
    inhibited_pathways: list[str] | None = None


class RejectionSummary(BaseModel):
    """Candidates removed from stage 5 on. Only runs made after this was added carry it."""

    n_removed: int
    reasons: dict[str, int] = Field(
        default_factory=dict,
        description="Removals per reason. A candidate can fail several checks, so these can add up to more than n_removed.",
    )
    cutoffs: dict[str, str] = Field(
        default_factory=dict,
        description="The cutoff each reason was judged against, from the run's own stage settings, e.g. 'P(hemolytic) above 0.5'. A reason with no entry has no single cutoff to state.",
    )


#: The stages whose rejections are counted, per pipeline: stage 5, 6, 8, 9.
_CUTOFF_STAGES = {
    "wound_healing": ("s05_physchem_screening", "s06_functional_models", "s08_safety_developability", "s09_synthesis_cmc"),
    "metabolic": (
        "s05_metabolic_physchem_screening",
        "s06_metabolic_functional_screening",
        "s08_metabolic_safety_developability",
        "s09_synthesis_cmc",
    ),
    "cosmetic": (
        "s05_cosmetic_physchem_screening",
        "s06_cosmetic_functional_models",
        "s08_cosmetic_safety_developability",
        "s09_cosmetic_synthesis_cmc",
    ),
}

#: Wound-healing stage 5 reads these two from constants in the stage, not from its params schema.
_WOUND_S5_MW_REJECT_MAX = 5000.0
_WOUND_S5_HYDROPHOBICITY_REJECT_MAX = 0.5


def _cutoff_text(pipeline: str, reason: str, s5: dict, s6: dict, s8: dict, s9: dict) -> str | None:
    """The cutoff behind one rejection reason, as words; None when no single value applies."""
    g = lambda x: f"{x:g}"  # noqa: E731
    base, _, qualifier = reason.partition(":")
    try:
        if pipeline == "cosmetic":
            thresholds = [s6.get(f"{f}_threshold") for f in ("collagen_ecm", "antioxidant", "pigmentation")]
            table = {
                "molecular_weight_too_high": lambda: f"above {g(s5['molecular_weight_reject_max'])} Da",
                "net_charge_out_of_range": lambda: f"outside {g(s5['net_charge_reject_min'])} to {g(s5['net_charge_reject_max'])}",
                "hydrophobicity_outside_seed_range": lambda: f"outside the {g(s5['hydrophobicity_percentile_low'])}th to {g(s5['hydrophobicity_percentile_high'])}th percentile of the seed peptides",
                "membrane_active_profile": lambda: f"hydrophobic moment above the seed {g(s5['moment_percentile'])}th percentile with net charge of {g(s5['membrane_active_min_charge'])} or more",
                "repeated_residues": lambda: f"{s5['repeat_run_length']} or more identical residues in a row",
                "solubility_insoluble": lambda: f"P(soluble) below {g(s5['solubility_threshold'])}",
                "primary_function_below_threshold": lambda: f"score below {g(thresholds[0])}" if len(set(thresholds)) == 1 else None,
                "antimicrobial": lambda: f"P(antimicrobial) at or above {g(s6['antimicrobial_reject_min'])}",
                "irritation_high": lambda: f"irritation score above the seed {g(s8['irritation_percentile_high'])}th percentile",
                "irritation_medium": lambda: f"irritation score above the seed {g(s8['irritation_percentile_medium'])}th percentile",
                "skin_sensitization": lambda: "predicted sensitizer (the model's own cutoff, 0.55)",
                "hemolysis": lambda: f"P(hemolytic) at or above {g(s8['hemolysis_reject_min'])}",
                "half_life": lambda: f"log10 half-life below {g(s8['half_life_reject_min'])} s",
                "allergenicity": lambda: f"P(allergen) at or above {g(s8['allergenicity_reject_min'])}",
                "cytotoxicity": lambda: f"P(cytotoxic) at or above {g(s8['cytotoxicity_reject_min'])}",
                "cleavage_stability": lambda: f"stability below {g(s8['cleavage_stability_reject_min'])}",
                "synthesis_feasibility_below_min": lambda: f"P(feasible) below {g(s9['synthesis_feasibility_min'])}",
            }
        else:
            metabolic = pipeline == "metabolic"
            t6 = s6.get("stage6_thresholds") or {}
            table = {
                "molecular_weight": lambda: f"above {g(s5['molecular_weight_reject_max'] if metabolic else _WOUND_S5_MW_REJECT_MAX)} Da",
                "hydrophobicity": lambda: f"mean hydrophobicity above {g(s5['hydrophobicity_reject_max'] if metabolic else _WOUND_S5_HYDROPHOBICITY_REJECT_MAX)}",
                "cytotoxicity": lambda: f"P(cytotoxic) above {g(s8['cytotoxicity_reject_max'])}",
                "aggregation_tendency": lambda: f"P(aggregation-prone) above {g(s8['aggregation_reject_max'])}",
                "cleavage_stability": lambda: f"stability below {g(s8['cleavage_stability_reject_min'])}",
                "hemolysis": lambda: (
                    f"P(hemolytic) above {g(s8['hemolysis_probability_reject_max'])}"
                    if metabolic
                    else f"pHC50 above {g(s8['hemolysis_phc50_reject_max'])}"
                ),
                "plasma_half_life": lambda: f"below {g(s8['plasma_half_life_min_minutes'])} min",
                "difficulty_class": lambda: "difficulty class: " + ", ".join(s9["rejected_difficulty_classes"]),
                "high_ec50": lambda: f"above {g(s6['ec50_max_nm'][qualifier])} nM",
                "low_agonism_probability": lambda: f"P(agonist) below {g(s6['agonism_probability_min'][qualifier])}",
                "low_anti_inflammatory_probability": lambda: f"P(anti-inflammatory) below {g(s6['anti_inflammatory_probability_min'] if metabolic else t6['min_anti_inflammatory_probability'])}",
                "low_amp_probability": lambda: f"P(antimicrobial) below {g(t6['min_amp_probability'])}",
                "low_proliferation_migration": lambda: f"below {g(t6['min_proliferation_migration'])}",
                "low_angiogenic_activity": lambda: f"below {g(t6['min_angiogenic_activity'])}",
                "high_mic": lambda: f"above {g(t6['max_mic'][qualifier])} µM",
                "high_mbic": lambda: f"above {g(t6['max_mbic'][qualifier])} µM",
            }
        return table[base]()
    except (KeyError, TypeError, IndexError):
        return None


def _rejection_cutoffs(run_dir: str, pipeline: str, reasons: list[str]) -> dict[str, str]:
    """Cutoff text per rejection reason: the run's stored stage settings over the schema defaults."""
    try:
        stages = json.loads(storage.read_text(storage.join(run_dir, "config.json"))).get("stages") or {}
    except Exception:  # noqa: BLE001 - an older run or an unreadable config: the defaults still apply
        stages = {}

    def params(name: str) -> dict:
        try:
            return resolve_params(name, (stages.get(name) or {}).get("params") or {})
        except Exception:  # noqa: BLE001 - stored params that no longer validate: report no cutoffs for that stage
            return {}

    s5, s6, s8, s9 = (params(name) for name in _CUTOFF_STAGES[pipeline])
    texts = {reason: _cutoff_text(pipeline, reason, s5, s6, s8, s9) for reason in reasons}
    return {reason: text for reason, text in texts.items() if text}


class MetabolicJobResultsResponse(BaseModel):
    pipeline: Literal["metabolic"] = "metabolic"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
    rejections: RejectionSummary | None = None
    candidates: list[MetabolicCandidateResponse] = []


class JobResultsResponse(BaseModel):
    pipeline: Literal["wound_healing"] = "wound_healing"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
    rejections: RejectionSummary | None = None
    candidates: list[CandidateResponse] = []


def _run_id_from_job_id(job_id: str) -> str:
    return job_id.rsplit("/", 1)[-1]


def _candidate_response(candidate: dict) -> CandidateResponse:
    preds = candidate.get("predictions") or {}

    def nested(key: str, field: str):
        return (preds.get(key) or {}).get(field)

    return CandidateResponse(
        id=candidate.get("id", "?"),
        sequence=candidate.get("sequence"),
        final_score_rank=nested("ranking", "final_score_rank"),
        pareto_rank=nested("ranking", "pareto_rank"),
        final_score=nested("ranking", "final_score"),
        evidence_coverage=nested("ranking", "evidence_coverage"),
        missing_modules=nested("ranking", "missing_modules"),
        amp_probability=preds.get("amp_probability"),
        hemolytic_activity_phc50=nested("hemolysis", "phc50"),
        molecular_weight=preds.get("molecular_weight"),
        net_charge=preds.get("net_charge"),
        instability_index=preds.get("instability_index"),
        deamidation_risk=nested("deamidation_risk", "risk_category"),
        oxidation_risk=nested("oxidation_risk", "risk_category"),
        solubility=nested("solubility", "score"),
        aggregation_tendency=nested("aggregation_tendency", "score"),
        anti_inflammatory_probability=preds.get("anti_inflammatory_probability"),
        angiogenic_activity=nested("angiogenic_activity", "angiogenic"),
        proliferation_probability=nested("proliferation_migration", "proliferation"),
        migration_probability=nested("proliferation_migration", "migration"),
        cytotoxicity_probability=nested("cytotoxicity", "score"),
        cleavage_stability=nested("cleavage_stability", "score"),
        log_mic_um=nested("mic", "log_mic_um") or {},
        pmbic=nested("mbic", "pmbic") or {},
        activated_pathways=nested("mechanism", "activated_pathways"),
        inhibited_pathways=nested("mechanism", "inhibited_pathways"),
        undetermined_pathways=nested("mechanism", "undetermined_pathways"),
    )


def _metabolic_candidate_response(candidate: dict) -> MetabolicCandidateResponse:
    preds = candidate.get("predictions") or {}

    def nested(key: str, field: str):
        return (preds.get(key) or {}).get(field)

    # THE FLAG IS DERIVED WHEN THE RUN DID NOT RECORD IT. Stage 6 only began writing
    # potency_unreliable_receptors recently; every run from before that has GIPR pEC50 and EC50
    # values and no flag, so the numbers reached the web app with nothing marking them as
    # unrankable. Reading the stored list when it is there and computing it from the receptors
    # this candidate actually has potency for when it is not fixes every client at once, and
    # fixes OLD runs, which re-running stage 6 could not.
    #
    # `is None`, not falsy: a run that recorded an EMPTY list targeted no unreliable receptor and
    # must stay empty, while a run that recorded nothing at all has to be worked out here.
    unreliable = preds.get("potency_unreliable_receptors")
    if unreliable is None:
        scored = {
            receptor
            for key in ("pec50", "ec50_nm", "binding_pkd", "agonism_probability")
            for receptor, value in (preds.get(key) or {}).items()
            if value is not None
        }
        unreliable = [r for r in UNRELIABLE_POTENCY_RECEPTORS if r in scored]

    return MetabolicCandidateResponse(
        id=candidate.get("id", "?"),
        sequence=candidate.get("sequence"),
        final_score_rank=nested("ranking", "final_score_rank"),
        pareto_rank=nested("ranking", "pareto_rank"),
        final_score=nested("ranking", "final_score"),
        evidence_coverage=nested("ranking", "evidence_coverage"),
        missing_modules=nested("ranking", "missing_modules"),
        pec50=preds.get("pec50") or {},
        ec50_nm=preds.get("ec50_nm") or {},
        binding_pkd=preds.get("binding_pkd") or {},
        agonism_probability=preds.get("agonism_probability") or {},
        pec50_pkd_gap=preds.get("pec50_pkd_gap"),
        potency_unreliable_receptors=unreliable,
        # The note is what tells a reader the values still count toward the gates and the score,
        # so a flagged receptor without it is a worse half-truth than no flag at all.
        potency_unreliable_note=(
            preds.get("potency_unreliable_note")
            or (UNRELIABLE_POTENCY_NOTE if unreliable else None)
        ),
        metabolic_activity_score=preds.get("metabolic_activity_score"),
        anti_inflammatory_probability=preds.get("anti_inflammatory_probability"),
        half_life_minutes=nested("plasma_half_life", "minutes"),
        hemolytic_activity_phc50=nested("hemolysis", "phc50"),
        p_hemolytic=nested("hemolysis", "p_hemolytic"),
        cleavage_stability=nested("cleavage_stability", "score"),
        molecular_weight=preds.get("molecular_weight"),
        net_charge=preds.get("net_charge"),
        instability_index=preds.get("instability_index"),
        deamidation_risk=nested("deamidation_risk", "risk_category"),
        oxidation_risk=nested("oxidation_risk", "risk_category"),
        solubility=nested("solubility", "score"),
        aggregation_tendency=nested("aggregation_tendency", "score"),
        cytotoxicity_probability=nested("cytotoxicity", "score"),
        activated_pathways=nested("mechanism", "activated_pathways"),
        inhibited_pathways=nested("mechanism", "inhibited_pathways"),
        undetermined_pathways=nested("mechanism", "undetermined_pathways"),
    )


class CosmeticComponentScore(BaseModel):
    score: float | None
    screened: bool


class CosmeticCandidateResponse(BaseModel):
    id: str = "?"
    sequence: str | None = None
    final_score: float | None = None
    final_score_rank: int | None = None
    pareto_rank: int | None = None
    evidence_coverage: float | None = None
    not_screened: list[str] = []
    component_scores: dict[str, CosmeticComponentScore] = {}
    primary_function: str | None = None
    secondary_functions: list[str] = []
    functional_scores: dict[str, dict] = {}
    antimicrobial_probability: float | None = None
    molecular_weight: float | None = None
    net_charge: float | None = None
    hydrophobic_moment: float | None = None
    physchem_flags: list[str] = []
    physchem_flag_count: int | None = None
    aggregation_tendency: float | None = None
    solubility: float | None = None
    irritation_tier: str | None = None
    irritation_score: float | None = None
    # penetration_passed: REMINDER, do not add back without restoring the Stage 8 reject. The penetration filter is
    # commented out (no penetration model yet; see docs/TODO.md and docs/LIMITATIONS.md).
    skin_sensitizer: bool | None = None
    skin_sensitization_probability: float | None = None
    hemolysis_probability: float | None = None
    cytotoxicity: dict[str, dict] = {}
    half_life_log10_seconds: float | None = None
    allergenicity_probability: float | None = None
    cleavage_stability: float | None = None
    safety_flags: list[str] = []
    synthesis_feasibility_probability: float | None = None
    synthesis_flags: dict[str, list[int]] = {}  # flag -> 0-based residue positions; the web UI shows them +1


class CosmeticJobResultsResponse(BaseModel):
    pipeline: Literal["cosmetic"] = "cosmetic"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
    rejections: RejectionSummary | None = None
    candidates: list[CosmeticCandidateResponse] = []


#: The results envelope each pipeline answers with. Keyed by every value Pipeline can take, so
#: adding a fourth pipeline is a KeyError here rather than a response quietly tagged
#: "wound_healing" -- which is exactly what the previous `if metabolic else wound` ternary did to
#: cosmetic.
RESULTS_RESPONSE_FOR: dict[str, type[BaseModel]] = {
    "wound_healing": JobResultsResponse,
    "metabolic": MetabolicJobResultsResponse,
    "cosmetic": CosmeticJobResultsResponse,
}


def _run_pipeline(
    run_dir: str, candidates: list[dict], stage: str | None = None
) -> Pipeline:
    """The run's pipeline, from the best evidence available.

    Three sources, in order of how much they prove:

    1. the config the API stored with the job, which names the brief stage outright;
    2. the stage the WORKER last reported, for an unfinished run whose config cannot be read --
       `s08_cosmetic_safety_developability` can only be a cosmetic run;
    3. the shape of the candidates, for a finished older run.

    Source 2 exists because the unfinished branch of get_job_results() has no candidates to
    inspect, so without it an unfinished run with no readable config fell through to
    "wound_healing" however cosmetic it actually was -- and the web app now chooses how to render
    a run from this field.
    """
    try:
        config = json.loads(storage.read_text(storage.join(run_dir, "config.json")))
        stages = config.get("stages") or {}
        for pipeline, brief_stage in (
            ("cosmetic", "s01_cosmetic_therapeutic_product_brief"),
            ("metabolic", "s01_metabolic_therapeutic_product_brief"),
        ):
            if (stages.get(brief_stage) or {}).get("enabled"):
                return pipeline
        return "wound_healing"
    except Exception:  # noqa: BLE001 - an older run or an unreadable config: infer from what is left
        for pipeline in ("cosmetic", "metabolic"):
            # The pipelines' own stages are named for them; the stages they share are not, so a
            # run sitting in s07_structure_mechanism says nothing here and falls through.
            if stage and f"_{pipeline}_" in stage:
                return pipeline
        if any("ec50_nm" in (c.get("predictions") or {}) for c in candidates):
            return "metabolic"
        if any("components" in ((c.get("predictions") or {}).get("ranking") or {}) for c in candidates):
            return "cosmetic"
        return "wound_healing"


def _cosmetic_component_stats(candidates: list[dict]) -> dict[str, ComponentStats]:
    """Mean/median/min/max per ranking component over the candidates that screened it."""
    values_by_component: dict[str, list[float]] = {}
    for candidate in candidates:
        components = ((candidate.get("predictions") or {}).get("ranking") or {}).get("components") or {}
        for name, component in components.items():
            if component.get("screened") and component.get("score") is not None:
                values_by_component.setdefault(name, []).append(component["score"])
    return {
        name: ComponentStats(
            count=len(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        for name, values in values_by_component.items()
    }


def _cosmetic_candidate_response(candidate: dict) -> CosmeticCandidateResponse:
    preds = candidate.get("predictions") or {}
    ranking = preds.get("ranking") or {}
    components = ranking.get("components") or {}
    verdict = preds.get("functional_verdict") or {}
    irritation = preds.get("irritation") or {}
    skin = preds.get("skin_sensitization") or {}
    cleavage = preds.get("cleavage_stability") or {}
    half_life = preds.get("half_life") or {}
    allergen = preds.get("allergenicity") or {}
    synthesis = preds.get("synthesis_feasibility") or {}
    hemolysis = preds.get("hemolysis") or {}
    aggregation = preds.get("aggregation_tendency") or {}
    solubility = preds.get("solubility") or {}
    return CosmeticCandidateResponse(
        id=candidate.get("id", "?"),
        sequence=candidate.get("sequence"),
        final_score=ranking.get("final_score"),
        final_score_rank=ranking.get("final_score_rank"),
        pareto_rank=ranking.get("pareto_rank"),
        evidence_coverage=ranking.get("coverage"),
        not_screened=ranking.get("not_screened") or [],
        component_scores={
            name: CosmeticComponentScore(score=c.get("score"), screened=bool(c.get("screened")))
            for name, c in components.items()
        },
        primary_function=verdict.get("primary_function"),
        secondary_functions=verdict.get("secondary_functions") or [],
        functional_scores=preds.get("functional_scores") or {},
        antimicrobial_probability=(preds.get("antimicrobial") or {}).get("probability"),
        molecular_weight=preds.get("molecular_weight"),
        net_charge=preds.get("net_charge"),
        hydrophobic_moment=preds.get("hydrophobic_moment"),
        physchem_flags=sorted(preds.get("physchem_flags") or {}),
        physchem_flag_count=preds.get("physchem_flag_count"),
        aggregation_tendency=aggregation.get("score"),
        solubility=solubility.get("score"),
        irritation_tier=irritation.get("tier"),
        irritation_score=irritation.get("score"),
        skin_sensitizer=skin.get("sensitizer"),
        skin_sensitization_probability=skin.get("p_sensitizer"),
        hemolysis_probability=hemolysis.get("p_hemolytic"),
        cytotoxicity=(preds.get("cytotoxicity") or {}).get("by_cell_type") or {},
        half_life_log10_seconds=half_life.get("log10_seconds"),
        allergenicity_probability=allergen.get("p_allergen"),
        cleavage_stability=cleavage.get("score"),
        safety_flags=preds.get("safety_flags") or [],
        synthesis_feasibility_probability=synthesis.get("probability"),
        synthesis_flags=synthesis.get("flags") or {},
    )


def _component_stats(candidates: list[dict]) -> dict[str, ComponentStats]:
    """Mean/median/min/max per ranking module over ranked candidates' post-deduction `score`; missing scores are excluded."""
    values_by_component: dict[str, list[float]] = {}
    for candidate in candidates:
        ranking = candidate.get("predictions", {}).get("ranking")
        if not ranking or ranking.get("status") != "ranked":
            continue
        for component, module in (ranking.get("modules") or {}).items():
            value = module.get("score")
            if value is not None:
                values_by_component.setdefault(component, []).append(value)

    return {
        component: ComponentStats(
            count=len(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
        for component, values in values_by_component.items()
    }


@app.get(
    "/api/v1/jobs/{job_id:path}/results",
    response_model=Annotated[
        Union[JobResultsResponse, MetabolicJobResultsResponse, CosmeticJobResultsResponse],
        Field(discriminator="pipeline"),
    ],
)
def get_job_results(job_id: str):
    name = job_name(job_id)
    run_id = _run_id_from_job_id(name)
    artifacts = required("VERTEX_ARTIFACTS_DIR")
    run_dir = storage.join(artifacts, "runs", run_id)
    print("Fetching results for job %s in %s", name, run_dir)
    status_path = storage.join(run_dir, "results.json")
    if not storage.exists(status_path):
        raise HTTPException(404, "Job results not found")
    status_payload = json.loads(storage.read_text(status_path))
    status = status_payload.get("status", "unknown")

    if status != "success":
        # ALL THREE PIPELINES, not metabolic versus everything else. This was a ternary that
        # sent cosmetic down the wound-healing branch, so an unfinished cosmetic run came back
        # tagged `pipeline: "wound_healing"`. Nothing rendered wrongly, because `candidates` is
        # empty here -- but the tag was false, and the web app now DISCRIMINATES ON IT to choose
        # how to render a run, so a wrong tag is a wrong screen as soon as these envelopes carry
        # anything pipeline-specific of their own.
        # The worker's last reported stage is the only pipeline evidence an unfinished run has
        # besides its config, and it is often there when the config is not yet readable.
        unfinished = RESULTS_RESPONSE_FOR[
            _run_pipeline(run_dir, [], stage=status_payload.get("stage"))
        ]
        return unfinished(
            job_id=name,
            run_id=run_id,
            status=status,
            n_final=status_payload.get("n_final"),
            candidates=[],
        )

    candidates_path = storage.join(run_dir, "candidates_final.json")
    if not storage.exists(candidates_path):
        raise HTTPException(
            404, "Job reported success but candidates_final.json is missing"
        )
    candidates = json.loads(storage.read_text(candidates_path))

    ranked = sum(
        1
        for c in candidates
        if c.get("predictions", {}).get("ranking", {}).get("status") == "ranked"
    )

    common = dict(
        job_id=name,
        run_id=run_id,
        status=status,
        n_final=status_payload.get("n_final"),
        ranked_candidates=ranked,
        insufficient_evidence_candidates=len(candidates) - ranked,
        component_stats=_component_stats(candidates),
        rejections=status_payload.get("rejections"),
    )
    pipeline = _run_pipeline(run_dir, candidates)
    rejections = status_payload.get("rejections")
    if rejections:
        rejections = {**rejections, "cutoffs": _rejection_cutoffs(run_dir, pipeline, list(rejections.get("reasons") or {}))}
    common["rejections"] = rejections
    if pipeline == "cosmetic":
        ranking_of = lambda c: (c.get("predictions") or {}).get("ranking") or {}  # noqa: E731
        cosmetic_common = {
            **common,
            "ranked_candidates": sum(1 for c in candidates if ranking_of(c).get("final_score") is not None),
            "insufficient_evidence_candidates": sum(1 for c in candidates if ranking_of(c).get("not_screened")),
            "component_stats": _cosmetic_component_stats(candidates),
        }
        return CosmeticJobResultsResponse(
            **cosmetic_common, candidates=[_cosmetic_candidate_response(c) for c in candidates]
        )
    if pipeline == "metabolic":
        return MetabolicJobResultsResponse(
            **common, candidates=[_metabolic_candidate_response(c) for c in candidates]
        )
    return JobResultsResponse(
        **common, candidates=[_candidate_response(c) for c in candidates]
    )


@app.post("/api/v1/jobs/{job_id:path}/cancel", response_model=CancelJobResponse)
def cancel_job(job_id: str):
    name = job_name(job_id)
    client = job_client()
    try:
        job = client.get_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc

    state = job.state.name
    if state in ("JOB_STATE_CANCELLING", "JOB_STATE_CANCELLED"):
        return {"job_id": name, "status": "cancelled"}
    if state in TERMINAL_VERTEX_STATES and state != "JOB_STATE_SUCCEEDED":
        return {"job_id": name, "status": "failed"}

    try:
        client.cancel_custom_job(name=name, timeout=30)
    except NotFound as exc:
        raise HTTPException(404, "Job not found") from exc
    except GoogleAuthError as exc:
        raise HTTPException(503, "Google credentials are unavailable") from exc
    except GoogleAPICallError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"job_id": name, "status": "cancelling"}


#: model_store/ holds each predictor's code; weights live on the worker, synced from VERTEX_MODEL_STORE by common/model_sync.py.
MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"


class ModelEntry(BaseModel):
    name: str
    version: str
    predictor_sha256: str | None = None
    model_card: dict | None = None
    model_card_present: bool = False
    #: "documented" | "placeholder" | None. A card file exists for every predictor, but eight of
    #: them are PLACEHOLDERS recording that the model is undescribed. Without this field
    #: `model_card_present` alone would read as "documented" for all sixteen, which is the
    #: opposite of what those files say.
    model_card_status: str | None = None


class ModelManifest(BaseModel):
    models: list[ModelEntry]
    weights_source: str | None = None
    caveat: str


#: Read through required() for EVERY create, whatever the pipeline.
SHARED_SETTINGS = (
    "VERTEX_CLOUD_PROJECT",
    "VERTEX_LOCATION",
    "VERTEX_ARTIFACTS_DIR",
    "VERTEX_MODEL_STORE",
    "WORKER_SERVICE_ACCOUNT",
    "MACHINE_TYPE",
    "ACCELERATOR_TYPE",
    "ACCELERATOR_COUNT",
)

#: The image and seed FASTA each pipeline reads, in step with create_job's two lookups above.
#: A pipeline whose pair is absent answers 503 and the other pipelines are unaffected, so these
#: are NOT in settings_missing -- a wound-healing-only deployment is correctly configured, and
#: reporting it as missing four settings would say otherwise.
PIPELINE_SETTINGS: dict[str, tuple[str, ...]] = {
    "wound_healing": ("WORKER_IMAGE_URI", "SEED_CANDIDATES_FILE"),
    "metabolic": ("WORKER_METABOLIC_IMAGE_URI", "SEED_CANDIDATES_FILE_METABOLIC"),
    "cosmetic": ("WORKER_COSMETIC_IMAGE_URI", "SEED_CANDIDATES_FILE_COSMETIC"),
    "immunomodulation": ("WORKER_IMMUNO_IMAGE_URI",),  # takes peptides in the request, so no seed file
    "food": ("WORKER_FOOD_IMAGE_URI",),  # takes its proteins in the request, so no seed file
}


@app.get("/healthz")
def healthz():
    """Liveness only. It deliberately does NOT check Vertex or GCS: a readiness probe that
    calls out to Google turns a Google incident into an unhealthy container, and this process
    is perfectly able to answer 503s explaining itself. Whether the settings are present is
    reported here as a fact rather than as health.

    `settings_missing` covers only what EVERY pipeline needs, so it stays comparable with what
    it has always meant. Each pipeline's own image and seed file are reported per pipeline:
    they are required for requests naming that pipeline and irrelevant to the others, and a
    deployment that runs one pipeline is not misconfigured for lacking another's settings.
    """
    missing = [name for name in SHARED_SETTINGS if not os.environ.get(name)]
    pipelines = {
        name: {
            "ready": not (absent := [s for s in settings if not os.environ.get(s)])
            and not missing,
            "settings_missing": absent,
        }
        for name, settings in PIPELINE_SETTINGS.items()
    }
    unavailable = sorted(n for n, p in pipelines.items() if not p["ready"])
    return {
        "ok": True,
        "settings_missing": missing,
        "pipelines": pipelines,
        "note": (
            "ok reports only that this process is answering. Job creation returns 503 while"
            " settings_missing is non-empty."
            if missing
            else "ok reports only that this process is answering; Vertex and Cloud Storage are"
            " not contacted by this endpoint."
        )
        + (
            f" Pipelines that would answer 503: {', '.join(unavailable)}."
            if unavailable
            else ""
        ),
    }


@app.get("/api/v1/models", response_model=ModelManifest)
def model_manifest():
    """What this deployment declares it will run, read from model_store/ on disk.

    HONEST LIMITS, because a provenance endpoint that overstates is worse than none:
    - `predictor_sha256` digests the predictor's CODE (predictor.py), not its weights. The API
      container does not hold the weights at all -- the worker syncs them from
      VERTEX_MODEL_STORE at run time -- so this process cannot digest them without downloading
      them, which a status endpoint has no business doing.
    - `version` is the suffix of the directory name, which is how this project versions a
      predictor. It is not derived from the weights either.
    - `model_card` is whatever the predictor ships, and `model_card_status` says whether it is a
      real description or a PLACEHOLDER. Every predictor now has a card file, but half of them
      record only that the model is undescribed -- so `model_card_present` alone would read as
      "documented" for all sixteen. Read the status, not the presence.
    """
    entries: list[ModelEntry] = []
    for directory in sorted(MODEL_STORE_DIR.glob("*/")):
        predictor = directory / "predictor.py"
        if not predictor.is_file():
            continue
        name = directory.name
        card_path = directory / "model_card.json"
        card = None
        if card_path.is_file():
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except ValueError:
                card = None
        card = _public_card(name, card)
        entries.append(
            ModelEntry(
                name=name,
                version=name.rsplit("_", 1)[-1] if "_v" in name else "unversioned",
                predictor_sha256=hashlib.sha256(predictor.read_bytes()).hexdigest(),
                model_card=card,
                model_card_present=card is not None,
                model_card_status=(
                    None
                    if card is None
                    else (
                        "placeholder"
                        if card.get("status") == "placeholder"
                        else "documented"
                    )
                ),
            )
        )
    return ModelManifest(
        models=entries,
        weights_source=os.environ.get("VERTEX_MODEL_STORE") or None,
        caveat=(
            "Digests cover predictor code, not model weights: the weights are synced to the"
            " worker from weights_source at run time and are not present in this process. A"
            " matching digest therefore means the same scoring code, not provably the same"
            " weights."
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("PORT", 8080)), log_level="info"
    )

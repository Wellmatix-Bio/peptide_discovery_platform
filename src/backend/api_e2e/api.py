"""Start and inspect complete-pipeline Vertex AI Custom Jobs."""

import hashlib
import json
import os
import re
import statistics
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union
from fastapi import FastAPI, HTTPException
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

# Imported for its side effect: common/env.py loads .env into os.environ; do not remove as unused.
from common import env  # noqa: F401
from schemas.brief import Brief, CosmeticBrief, MetabolicBrief
from schemas.e2e_config import validate_job_config
from schemas.stage_configs import CosmeticE2ERequest, E2ERequest, MetabolicE2ERequest

app = FastAPI(title="End-to-end Pipeline Job API")


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


class MetabolicJobResultsResponse(BaseModel):
    pipeline: Literal["metabolic"] = "metabolic"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
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
    synthesis_flags: dict[str, list[int]] = {}


class CosmeticJobResultsResponse(BaseModel):
    pipeline: Literal["cosmetic"] = "cosmetic"
    job_id: str
    run_id: str
    status: str
    n_final: int | None = None
    ranked_candidates: int | None = None
    insufficient_evidence_candidates: int | None = None
    component_stats: dict[str, ComponentStats] = Field(default_factory=dict)
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
    )
    pipeline = _run_pipeline(run_dir, candidates)
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


class CancelJobResponse(BaseModel):
    job_id: str
    status: Literal["cancelling", "cancelled", "failed"]


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
PIPELINE_SETTINGS: dict[str, tuple[str, str]] = {
    "wound_healing": ("WORKER_IMAGE_URI", "SEED_CANDIDATES_FILE"),
    "metabolic": ("WORKER_METABOLIC_IMAGE_URI", "SEED_CANDIDATES_FILE_METABOLIC"),
    "cosmetic": ("WORKER_COSMETIC_IMAGE_URI", "SEED_CANDIDATES_FILE_COSMETIC"),
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

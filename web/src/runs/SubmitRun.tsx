import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createJob } from "../api/peptide";
import {
  CYTOTOXICITY_CELL_TYPES,
  DESIRED_FUNCTIONS,
  DOSING_BOUND,
  LENGTH_BOUND,
  METABOLIC_LENGTH_BOUND,
  PATHOGENS,
  TARGET_RECEPTORS,
  WOUND_CONTEXTS,
  humanise,
} from "../api/schema";
import type {
  BriefFields,
  MetabolicBrief,
  MetabolicCreateJobRequest,
  WoundHealingCreateJobRequest,
} from "../api/types";
import { ErrorBox, Hero, Pill } from "../components/ui";
import { lastRun } from "../workspace";
import { EMPTY_BRIEF, EMPTY_METABOLIC_BRIEF, EXAMPLES, METABOLIC_EXAMPLES } from "./examples";

/* Start a run.
 *
 * Every option and bound comes from the OpenAPI snapshot (src/api/schema.ts), never from a list
 * typed here, so the form can only offer what the API accepts. Both submit paths -- the form's
 * own and the JSON tab's -- run the same checks() before sending anything.
 *
 * The examples are this app's own, because none of the repository's 32 briefs is a valid request
 * (docs/BRIEF_VALIDITY.md).
 *
 * TWO PIPELINES, TWO BRIEF SCHEMAS. The API routes wound-healing, metabolic and cosmetic jobs
 * through one endpoint, discriminated by `pipeline`, and each one accepts its OWN stage names: a
 * wound-healing brief sent as `pipeline: "metabolic"` is a 422, not a reinterpretation. So the
 * two briefs are separate state with separate checks, and switching pipeline does not carry one
 * schema's answers into the other. Cosmetic is not offered here yet; its results render, and the
 * JSON tab can submit one. */

type Tab = "form" | "json";
type Pipeline = "wound_healing" | "metabolic";

interface Advanced {
  seed: number;
  n_peptides: number;
  ph: number;
  cytotoxicity_cell_type: string;
  stages: Record<string, boolean>;
}

const STAGES = [
  ["s04_candidate_generation", "Generation", "Route A (genetic algorithm) and Route B (ProtGPT2)."],
  ["s05_physchem_screening", "Physicochemical screening", "Charge, hydrophobicity, stability."],
  ["s06_functional_models", "Functional models", "Antimicrobial, migration, angiogenesis."],
  ["s07_structure_mechanism", "Structure & mechanism", "ESMFold. The slowest stage by far."],
  ["s08_safety_developability", "Safety & developability", "Haemolysis, cytotoxicity, aggregation."],
  ["s09_synthesis_cmc", "Synthesis & CMC", "Difficulty, cost band, purity ceiling."],
] as const;

/* The metabolic pipeline's stages. Stages 7 and 9 are the SAME stages as wound healing's and keep
   their names; the other four are metabolic peers that replace their wound-healing counterparts,
   and the API refuses the wound-healing name under this pipeline. s11_metabolic_ranking is not
   offered for the same reason s11_ranking is not offered above: turning the ranking stage off
   leaves a run with no shortlist, which is not a thing this form should make easy. */
const METABOLIC_STAGES = [
  [
    "s04_metabolic_candidate_generation",
    "Generation",
    "Route A (genetic algorithm) and Route B, seeded from known incretin analogues.",
  ],
  [
    "s05_metabolic_physchem_screening",
    "Physicochemical screening",
    "Charge, hydrophobicity, amphipathicity, stability.",
  ],
  [
    "s06_metabolic_functional_screening",
    "Receptor potency & function",
    "pEC50 and EC50 per receptor, binding, agonism probability, plasma half-life.",
  ],
  ["s07_structure_mechanism", "Structure & mechanism", "ESMFold. The slowest stage by far."],
  [
    "s08_metabolic_safety_developability",
    "Safety & developability",
    "Haemolysis, cytotoxicity, aggregation, cleavage.",
  ],
  ["s09_synthesis_cmc", "Synthesis & CMC", "Difficulty, cost band, purity ceiling."],
] as const;

const ADVANCED_DEFAULTS: Advanced = {
  seed: 42,
  n_peptides: 100,
  ph: 7.4,
  cytotoxicity_cell_type: "DRAMP_aggregate",
  // Both pipelines' stage names, so switching pipeline does not read an undefined toggle.
  stages: Object.fromEntries(
    [...STAGES, ...METABOLIC_STAGES].map(([name]) => [name, true]),
  ),
};

/** The receptor whose potency model cannot rank candidates. Mirrors
 *  UNRELIABLE_POTENCY_RECEPTORS in model_store/functional_potency_predictor_v1; the API returns
 *  the same list per candidate in `potency_unreliable_receptors`, and RunView shows it there. It
 *  is repeated here so the warning arrives BEFORE a GPU job is paid for, not after. */
const UNRELIABLE_RECEPTORS = ["GIPR"];

/** Everything wrong with a brief, as a list a person can act on. Runs on EVERY submit path. */
function checks(brief: BriefFields, advanced: Advanced): string[] {
  const problems: string[] = [];
  if (brief.wound_context.length === 0) problems.push("Choose at least one wound context.");
  if (brief.desired_functions.length === 0) problems.push("Choose at least one desired function.");
  if (brief.pathogens.length === 0) {
    problems.push("Choose at least one pathogen — the API requires a non-empty list.");
  }
  if (brief.min_length > brief.max_length) {
    problems.push("Shortest peptide cannot be longer than the longest.");
  }
  for (const [label, value] of [
    ["Shortest peptide", brief.min_length],
    ["Longest peptide", brief.max_length],
  ] as const) {
    if (value < LENGTH_BOUND.min || value > LENGTH_BOUND.max) {
      problems.push(`${label} must be between ${LENGTH_BOUND.min} and ${LENGTH_BOUND.max}.`);
    }
  }
  if (
    brief.dosing_interval_hours < DOSING_BOUND.min ||
    brief.dosing_interval_hours > DOSING_BOUND.max
  ) {
    problems.push(`Dosing interval must be between ${DOSING_BOUND.min} and ${DOSING_BOUND.max} hours.`);
  }
  if (advanced.n_peptides < 1) problems.push("Generate at least 1 candidate.");
  return problems;
}

/** The metabolic brief's own rules. A SEPARATE function rather than a flag on checks(): the two
 *  schemas share two field names and nothing else, and the metabolic brief has no dosing
 *  interval, no wound context and no pathogen requirement. */
function metabolicChecks(brief: MetabolicBrief, advanced: Advanced): string[] {
  const problems: string[] = [];
  if (brief.target_receptors.length === 0) {
    problems.push("Choose at least one target receptor — the API requires a non-empty list.");
  }
  for (const receptor of brief.target_receptors) {
    if (!TARGET_RECEPTORS.includes(receptor)) {
      problems.push(`${receptor} is not a receptor the API accepts.`);
    }
  }
  if (brief.min_length > brief.max_length) {
    problems.push("Shortest peptide cannot be longer than the longest.");
  }
  for (const [label, value] of [
    ["Shortest peptide", brief.min_length],
    ["Longest peptide", brief.max_length],
  ] as const) {
    if (value < METABOLIC_LENGTH_BOUND.min || value > METABOLIC_LENGTH_BOUND.max) {
      problems.push(
        `${label} must be between ${METABOLIC_LENGTH_BOUND.min} and ${METABOLIC_LENGTH_BOUND.max}.`,
      );
    }
  }
  if (advanced.n_peptides < 1) problems.push("Generate at least 1 candidate.");
  return problems;
}

function buildRequest(name: string, brief: BriefFields, advanced: Advanced): WoundHealingCreateJobRequest {
  const enabled = (stage: string) => ({ enabled: advanced.stages[stage] ?? true });
  return {
    /* The proxy replaces this with u<user id>:<uuid> before it reaches the API and keeps the text
       as this run's name. Sending the name is how the name is recorded. */
    request_id: name.trim() || "unnamed run",
    pipeline: "wound_healing",
    stages: {
      s01_therapeutic_product_brief: { brief, seed: advanced.seed },
      s04_candidate_generation: { ...enabled("s04_candidate_generation"), n_peptides: advanced.n_peptides },
      s05_physchem_screening: { ...enabled("s05_physchem_screening"), ph: advanced.ph },
      s06_functional_models: enabled("s06_functional_models"),
      s07_structure_mechanism: enabled("s07_structure_mechanism"),
      s08_safety_developability: {
        ...enabled("s08_safety_developability"),
        cytotoxicity_cell_type: advanced.cytotoxicity_cell_type,
      },
      s09_synthesis_cmc: enabled("s09_synthesis_cmc"),
    },
  } as WoundHealingCreateJobRequest;
}

function buildMetabolicRequest(
  name: string,
  brief: MetabolicBrief,
  advanced: Advanced,
): MetabolicCreateJobRequest {
  const enabled = (stage: string) => ({ enabled: advanced.stages[stage] ?? true });
  return {
    request_id: name.trim() || "unnamed run",
    pipeline: "metabolic",
    stages: {
      s01_metabolic_therapeutic_product_brief: { brief, seed: advanced.seed },
      s04_metabolic_candidate_generation: {
        ...enabled("s04_metabolic_candidate_generation"),
        n_peptides: advanced.n_peptides,
      },
      s05_metabolic_physchem_screening: enabled("s05_metabolic_physchem_screening"),
      s06_metabolic_functional_screening: enabled("s06_metabolic_functional_screening"),
      s07_structure_mechanism: enabled("s07_structure_mechanism"),
      s08_metabolic_safety_developability: enabled("s08_metabolic_safety_developability"),
      s09_synthesis_cmc: enabled("s09_synthesis_cmc"),
    },
    /* Every stage not named above, s11_metabolic_ranking included, takes the API's own default.
       The casts are the same one buildRequest() makes: each stage schema has many more fields
       than this form sets, and all of them are optional server-side. */
  } as unknown as MetabolicCreateJobRequest;
}

function Chooser({
  legend,
  hint,
  values,
  chosen,
  onChange,
}: {
  legend: string;
  hint: string;
  values: string[];
  chosen: string[];
  onChange: (next: string[]) => void;
}) {
  /* The native checkbox stays, for the keyboard and for screen readers, and is hidden visually;
     the pill is the control. Previously these were bare checkboxes, which the design system's
     `.field input` rule rendered as full-width text inputs stacked above their own labels. */
  return (
    <div className="field">
      <label>
        {legend}
        {chosen.length > 0 ? (
          <span style={{ fontWeight: 400, color: "var(--muted)" }}> — {chosen.length} selected</span>
        ) : null}
      </label>
      <div className="choices">
        {values.map((value) => {
          const on = chosen.includes(value);
          return (
            <label key={value} className={`choice${on ? " on" : ""}`}>
              <input
                type="checkbox"
                checked={on}
                onChange={() =>
                  onChange(on ? chosen.filter((one) => one !== value) : [...chosen, value])
                }
              />
              <span className="mark" aria-hidden="true" />
              {humanise(value)}
            </label>
          );
        })}
      </div>
      <p className="help">{hint}</p>
    </div>
  );
}

function NameAndExample({
  name,
  setName,
  placeholder,
  examples,
  onPick,
  exampleHelp,
}: {
  name: string;
  setName: (next: string) => void;
  placeholder: string;
  examples: { id: string; label: string }[];
  onPick: (id: string) => void;
  exampleHelp: React.ReactNode;
}) {
  return (
    <>
      <div className="field">
        <label htmlFor="name">Name this run</label>
        <input
          id="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder={placeholder}
        />
        <p className="help">
          For your own history only. It never reaches the pipeline &mdash; the submission id is
          generated for you.
        </p>
      </div>

      <div className="field">
        <label htmlFor="example">Start from an example</label>
        <select id="example" defaultValue="" onChange={(event) => onPick(event.target.value)}>
          <option value="">Choose an example…</option>
          {examples.map((example) => (
            <option key={example.id} value={example.id}>
              {example.label}
            </option>
          ))}
        </select>
        <p className="help">{exampleHelp}</p>
      </div>
    </>
  );
}

function StageToggles({
  stages,
  advanced,
  setAdvanced,
}: {
  stages: readonly (readonly [string, string, string])[];
  advanced: Advanced;
  setAdvanced: (next: Advanced) => void;
}) {
  return (
    <div className="toggles">
      {stages.map(([stage, label, note]) => {
        const on = advanced.stages[stage] ?? true;
        return (
          <label key={stage} className={`toggle${on ? " on" : " off"}`}>
            <input
              type="checkbox"
              checked={on}
              onChange={(event) =>
                setAdvanced({
                  ...advanced,
                  stages: { ...advanced.stages, [stage]: event.target.checked },
                })
              }
            />
            <span className="switch" aria-hidden="true" />
            <span>
              <b>{label}</b>
              <small>{note}</small>
            </span>
          </label>
        );
      })}
    </div>
  );
}

export function SubmitRun() {
  const navigate = useNavigate();
  const [pipeline, setPipeline] = useState<Pipeline>("wound_healing");
  const [tab, setTab] = useState<Tab>("form");
  const [name, setName] = useState("");
  const [brief, setBrief] = useState<BriefFields>(EMPTY_BRIEF);
  const [metabolicBrief, setMetabolicBrief] = useState<MetabolicBrief>(EMPTY_METABOLIC_BRIEF);
  const [advanced, setAdvanced] = useState<Advanced>(ADVANCED_DEFAULTS);
  const [json, setJson] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const request = useMemo(
    () =>
      pipeline === "metabolic"
        ? buildMetabolicRequest(name, metabolicBrief, advanced)
        : buildRequest(name, brief, advanced),
    [pipeline, name, brief, metabolicBrief, advanced],
  );

  const unreliableChosen = metabolicBrief.target_receptors.filter((receptor) =>
    UNRELIABLE_RECEPTORS.includes(receptor),
  );

  const submit = async (payload: CriticalPayload) => {
    setError(null);
    setProblems([]);
    let body: WoundHealingCreateJobRequest | MetabolicCreateJobRequest;
    if (payload.kind === "form") {
      const found =
        pipeline === "metabolic"
          ? metabolicChecks(metabolicBrief, advanced)
          : checks(brief, advanced);
      if (found.length > 0) {
        setProblems(found);
        return;
      }
      body = request;
    } else {
      let parsed: unknown;
      try {
        parsed = JSON.parse(json);
      } catch (problem) {
        setProblems([`That is not valid JSON: ${(problem as Error).message}`]);
        return;
      }
      /* THE PASTED JSON NAMES ITS OWN PIPELINE, and the tag is read from the text rather than
         from the picker: a request pasted from elsewhere describes what it is, and silently
         retagging it to whatever the form happens to show would send the API a brief under a
         pipeline that rejects its stage names. An absent tag means wound healing, which is the
         API's own default for requests written before there was more than one pipeline. */
      const tag = (parsed as { pipeline?: unknown })?.pipeline ?? "wound_healing";
      if (tag === "metabolic") {
        const candidate = parsed as MetabolicCreateJobRequest;
        const candidateBrief = candidate?.stages?.s01_metabolic_therapeutic_product_brief?.brief;
        if (!candidateBrief) {
          setProblems([
            "A metabolic request must contain" +
              " stages.s01_metabolic_therapeutic_product_brief.brief.",
          ]);
          return;
        }
        const found = metabolicChecks(candidateBrief as MetabolicBrief, advanced);
        if (found.length > 0) {
          setProblems(found);
          return;
        }
        body = { ...candidate, pipeline: "metabolic" };
      } else if (tag === "wound_healing") {
        const candidate = parsed as WoundHealingCreateJobRequest;
        const candidateBrief = candidate?.stages?.s01_therapeutic_product_brief?.brief;
        if (!candidateBrief) {
          setProblems(["The JSON must contain stages.s01_therapeutic_product_brief.brief."]);
          return;
        }
        /* The SAME checks as the form, not a shortcut past them. */
        const found = checks(candidateBrief as BriefFields, advanced);
        if (found.length > 0) {
          setProblems(found);
          return;
        }
        body = { ...candidate, pipeline: "wound_healing" };
      } else {
        /* Refused rather than forwarded. The cosmetic pipeline is real and the API would accept
           it, but nothing here can check a cosmetic brief, and submitting an unchecked request
           starts a GPU job that bills. */
        setProblems([
          `This app cannot submit a "${String(tag)}" request: it has no checks for that` +
            " pipeline's brief, and it will not start a billed GPU job it could not check.",
        ]);
        return;
      }
    }
    setBusy(true);
    try {
      const { body: created } = await createJob(body);
      lastRun.set(created.job_id);
      navigate(`/runs/${encodeURIComponent(created.job_id)}`);
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  };

  const loadExample = (id: string) => {
    const example = EXAMPLES.find((one) => one.id === id);
    if (!example) return;
    setBrief(example.brief);
    setName(example.label);
    setProblems([]);
  };

  const loadMetabolicExample = (id: string) => {
    const example = METABOLIC_EXAMPLES.find((one) => one.id === id);
    if (!example) return;
    setMetabolicBrief(example.brief);
    setName(example.label);
    setProblems([]);
  };

  return (
    <>
      <Hero
        title="Start a run"
        sub={
          pipeline === "metabolic" ? (
            <>
              Choose the receptors the peptide should agonise. The pipeline generates candidates,
              screens them for potency and safety, and returns a ranked shortlist.
            </>
          ) : (
            <>
              Describe the wound and what the peptide should do. The pipeline generates candidates,
              screens them and returns a ranked shortlist.
            </>
          )
        }
        badge="Each run starts a GPU job"
      />

      <div className="card">
        <h2>Which pipeline</h2>
        <div className="tabs" style={{ marginTop: 0 }} role="group" aria-label="pipeline">
          <button
            className={`tab${pipeline === "wound_healing" ? " on" : ""}`}
            onClick={() => {
              setPipeline("wound_healing");
              setProblems([]);
            }}
          >
            Wound healing
          </button>
          <button
            className={`tab${pipeline === "metabolic" ? " on" : ""}`}
            onClick={() => {
              setPipeline("metabolic");
              setProblems([]);
            }}
          >
            Metabolic
          </button>
        </div>
        <p className="help">
          Two different briefs, not one brief with extra fields &mdash; each pipeline accepts its
          own stages, so switching keeps the two sets of answers apart rather than carrying one
          across. The cosmetic pipeline&rsquo;s results render in this app, but submitting one is
          only possible through the JSON tab.
        </p>
      </div>

      <div className="tabs">
        <button className={`tab${tab === "form" ? " on" : ""}`} onClick={() => setTab("form")}>
          Brief
        </button>
        <button
          className={`tab${tab === "json" ? " on" : ""}`}
          onClick={() => {
            setJson(JSON.stringify(request, null, 2));
            setTab("json");
          }}
        >
          Advanced: JSON
        </button>
      </div>

      {tab === "json" ? (
        <div className="card">
          <h2>Advanced: the request as JSON</h2>
          <p className="help" style={{ marginTop: 0 }}>
            Everything the API accepts, including stage parameters this form does not show. Edit or
            paste, then submit. The same checks run either way, and the{" "}
            <span className="mono">pipeline</span> field in the text decides which checks those are
            &mdash; not the picker above.
          </p>
          <div className="field">
            <textarea
              className="json"
              value={json}
              onChange={(event) => setJson(event.target.value)}
              spellCheck={false}
              aria-label="request JSON"
            />
          </div>
          <div className="btns">
            <button
              className="btn secondary"
              type="button"
              onClick={() => setJson(JSON.stringify(request, null, 2))}
            >
              Reset from the form
            </button>
          </div>
        </div>
      ) : pipeline === "metabolic" ? (
        <div className="grid">
          <div>
            <div className="card">
              <h2>The target</h2>
              <NameAndExample
                name={name}
                setName={setName}
                placeholder="e.g. dual agonist, attempt 2"
                examples={METABOLIC_EXAMPLES}
                onPick={loadMetabolicExample}
                exampleHelp={
                  <>
                    Fills the fields below, which you can then change. Each example says which
                    receptors it targets and why.
                  </>
                }
              />

              <Chooser
                legend="Target receptors"
                hint={`The API's own enum, and the one thing a metabolic brief must say. ${TARGET_RECEPTORS.length} values.`}
                values={TARGET_RECEPTORS}
                chosen={metabolicBrief.target_receptors}
                onChange={(next) =>
                  setMetabolicBrief({
                    ...metabolicBrief,
                    target_receptors: next as MetabolicBrief["target_receptors"],
                  })
                }
              />

              {/* SAID BEFORE THE JOB IS PAID FOR, not only on the results page. The potency model
                  for this receptor scores a negative R2 under leave-one-cluster-out, so its
                  ordering of two candidates carries no information -- and the value still counts
                  toward the gates and the ranking score, which is why this is worth a warning
                  rather than a footnote. It does not block submission: wanting the numbers anyway
                  is a legitimate reason to run it. */}
              {unreliableChosen.length > 0 ? (
                <div className="field">
                  <Pill tone="warn">{unreliableChosen.join(", ")} potency is not predictive</Pill>
                  <p className="help">
                    Leave-one-cluster-out R&sup2; for {unreliableChosen.join(" and ")} is negative
                    &mdash; worse than always predicting the training mean &mdash; on 55 labels
                    against 271 for GLP1R. The values are returned and flagged, and they{" "}
                    <b>still count toward the gates and the ranking score</b>. You can submit this;
                    just do not read the resulting order as evidence about{" "}
                    {unreliableChosen.join(" or ")}.
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          <div>
            <div className="card">
              <h2>Constraints</h2>
              <div className="row">
                <div className="field">
                  <label htmlFor="m-min-length">Shortest peptide</label>
                  <input
                    id="m-min-length"
                    type="number"
                    min={METABOLIC_LENGTH_BOUND.min}
                    max={METABOLIC_LENGTH_BOUND.max}
                    value={metabolicBrief.min_length}
                    onChange={(event) =>
                      setMetabolicBrief({
                        ...metabolicBrief,
                        min_length: Number(event.target.value),
                      })
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="m-max-length">Longest peptide</label>
                  <input
                    id="m-max-length"
                    type="number"
                    min={METABOLIC_LENGTH_BOUND.min}
                    max={METABOLIC_LENGTH_BOUND.max}
                    value={metabolicBrief.max_length}
                    onChange={(event) =>
                      setMetabolicBrief({
                        ...metabolicBrief,
                        max_length: Number(event.target.value),
                      })
                    }
                  />
                </div>
              </div>
              <p className="help">
                Residues, {METABOLIC_LENGTH_BOUND.min}&ndash;{METABOLIC_LENGTH_BOUND.max}. The
                incretin analogues this pipeline is seeded from are around 30 residues, so a range
                far below that asks for something unlike its training data.
              </p>
              <p className="help">
                A metabolic brief has no dosing interval, no wound context and no pathogen list.
                Its three optional free-text fields take no enum, so this form leaves them unset
                &mdash; the JSON tab reaches them.
              </p>
            </div>

            <div className="card">
              <h2>Generation</h2>
              <div className="row">
                <div className="field">
                  <label htmlFor="m-n-peptides">Candidates to generate</label>
                  <input
                    id="m-n-peptides"
                    type="number"
                    min={1}
                    value={advanced.n_peptides}
                    onChange={(event) =>
                      setAdvanced({ ...advanced, n_peptides: Number(event.target.value) })
                    }
                  />
                  <p className="help">The main driver of how long the run takes, and what it costs.</p>
                </div>
                <div className="field">
                  <label htmlFor="m-seed">Random seed</label>
                  <input
                    id="m-seed"
                    type="number"
                    value={advanced.seed}
                    onChange={(event) => setAdvanced({ ...advanced, seed: Number(event.target.value) })}
                  />
                  <p className="help">Same seed and same brief, same candidates.</p>
                </div>
              </div>
            </div>

            <div className="card">
              <h2>Stages</h2>
              <p className="help" style={{ marginTop: 0 }}>
                All on by default. Turning one off skips its scoring &mdash; and its filtering, so
                candidates it would have rejected survive. Every stage parameter not shown here
                takes the API&rsquo;s default; the JSON tab has them all.
              </p>
              <StageToggles
                stages={METABOLIC_STAGES}
                advanced={advanced}
                setAdvanced={setAdvanced}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="grid">
          <div>
            <div className="card">
              <h2>The wound</h2>
              <NameAndExample
                name={name}
                setName={setName}
                placeholder="e.g. diabetic foot ulcer, attempt 3"
                examples={EXAMPLES}
                onPick={loadExample}
                exampleHelp={
                  <>
                    Fills the fields below, which you can then change. These are written for this
                    form; the briefs in <span className="mono">data/briefs/</span> are not valid API
                    requests (see Models &amp; health).
                  </>
                }
              />

              <Chooser
                legend="Wound context"
                hint={`${WOUND_CONTEXTS.length} values, all of them from the API's own vocabulary. Choose every one that applies.`}
                values={WOUND_CONTEXTS}
                chosen={brief.wound_context}
                onChange={(next) => setBrief({ ...brief, wound_context: next as BriefFields["wound_context"] })}
              />

              <Chooser
                legend="Desired functions"
                hint="What the peptide should do. Ranking weights follow from these."
                values={DESIRED_FUNCTIONS}
                chosen={brief.desired_functions}
                onChange={(next) =>
                  setBrief({ ...brief, desired_functions: next as BriefFields["desired_functions"] })
                }
              />

              <Chooser
                legend="Target pathogens"
                hint="Only these three are accepted — the MIC model supports no others, and a pathogen outside its vocabulary is skipped rather than scored."
                values={PATHOGENS}
                chosen={brief.pathogens}
                onChange={(next) => setBrief({ ...brief, pathogens: next as BriefFields["pathogens"] })}
              />
            </div>
          </div>

          <div>
            <div className="card">
              <h2>Constraints</h2>
              <div className="row">
                <div className="field">
                  <label htmlFor="min-length">Shortest peptide</label>
                  <input
                    id="min-length"
                    type="number"
                    min={LENGTH_BOUND.min}
                    max={LENGTH_BOUND.max}
                    value={brief.min_length}
                    onChange={(event) =>
                      setBrief({ ...brief, min_length: Number(event.target.value) })
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="max-length">Longest peptide</label>
                  <input
                    id="max-length"
                    type="number"
                    min={LENGTH_BOUND.min}
                    max={LENGTH_BOUND.max}
                    value={brief.max_length}
                    onChange={(event) =>
                      setBrief({ ...brief, max_length: Number(event.target.value) })
                    }
                  />
                </div>
              </div>
              <p className="help">
                Residues, {LENGTH_BOUND.min}&ndash;{LENGTH_BOUND.max}. The floor is the models&rsquo;,
                not a preference: shorter peptides are outside what they were trained to score.
              </p>

              <div className="field">
                <label htmlFor="dosing">Dosing interval (hours)</label>
                <input
                  id="dosing"
                  type="number"
                  min={DOSING_BOUND.min}
                  max={DOSING_BOUND.max}
                  value={brief.dosing_interval_hours}
                  onChange={(event) =>
                    setBrief({ ...brief, dosing_interval_hours: Number(event.target.value) })
                  }
                />
                <p className="help">
                  {DOSING_BOUND.min}&ndash;{DOSING_BOUND.max}. Feeds the stability requirement.
                </p>
              </div>
            </div>

            <div className="card">
              <h2>Generation</h2>
              <div className="row">
                <div className="field">
                  <label htmlFor="n-peptides">Candidates to generate</label>
                  <input
                    id="n-peptides"
                    type="number"
                    min={1}
                    value={advanced.n_peptides}
                    onChange={(event) =>
                      setAdvanced({ ...advanced, n_peptides: Number(event.target.value) })
                    }
                  />
                  <p className="help">The main driver of how long the run takes, and what it costs.</p>
                </div>
                <div className="field">
                  <label htmlFor="seed">Random seed</label>
                  <input
                    id="seed"
                    type="number"
                    value={advanced.seed}
                    onChange={(event) => setAdvanced({ ...advanced, seed: Number(event.target.value) })}
                  />
                  <p className="help">Same seed and same brief, same candidates.</p>
                </div>
              </div>
            </div>

            <div className="card">
              <h2>Stages</h2>
              <p className="help" style={{ marginTop: 0 }}>
                All on by default. Turning one off skips its scoring &mdash; and its filtering, so
                candidates it would have rejected survive.
              </p>
              <StageToggles stages={STAGES} advanced={advanced} setAdvanced={setAdvanced} />
              <div className="field">
                <label htmlFor="cell-type">Cytotoxicity reference cells</label>
                <select
                  id="cell-type"
                  value={advanced.cytotoxicity_cell_type}
                  onChange={(event) =>
                    setAdvanced({ ...advanced, cytotoxicity_cell_type: event.target.value })
                  }
                >
                  {CYTOTOXICITY_CELL_TYPES.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
                <p className="help">Which cell line the safety stage scores against.</p>
              </div>
              <div className="field">
                <label htmlFor="ph">Screening pH</label>
                <input
                  id="ph"
                  type="number"
                  step="0.1"
                  value={advanced.ph}
                  onChange={(event) => setAdvanced({ ...advanced, ph: Number(event.target.value) })}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {problems.length > 0 ? (
        <div className="err" style={{ marginTop: 14 }}>
          <b>Fix these first</b>
          {"\n"}
          {problems.map((problem) => `• ${problem}`).join("\n")}
        </div>
      ) : null}
      <ErrorBox error={error} />

      <div className="card" style={{ marginTop: 14 }}>
        <div className="btns" style={{ marginTop: 0 }}>
          <button
            className="btn primary"
            type="button"
            disabled={busy}
            onClick={() => submit({ kind: tab === "json" ? "json" : "form" })}
          >
            {busy ? "Submitting…" : "Submit this run →"}
          </button>
        </div>
        <p className="help">
          Submitting starts a GPU job on Vertex AI that bills to this project, and it runs for
          minutes to hours. You can close the tab &mdash; the run continues, and it will be in your
          run history.
        </p>
      </div>
    </>
  );
}

type CriticalPayload = { kind: "form" | "json" };

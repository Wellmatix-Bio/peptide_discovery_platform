import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createJob } from "../api/peptide";
import {
  CYTOTOXICITY_CELL_TYPES,
  DESIRED_FUNCTIONS,
  DOSING_BOUND,
  LENGTH_BOUND,
  PATHOGENS,
  WOUND_CONTEXTS,
  humanise,
} from "../api/schema";
import type { BriefFields, CreateJobRequest } from "../api/types";
import { ErrorBox, Hero } from "../components/ui";
import { lastRun } from "../workspace";
import { EMPTY_BRIEF, EXAMPLES } from "./examples";

/* Start a run.
 *
 * Every option and bound comes from the OpenAPI snapshot (src/api/schema.ts), never from a list
 * typed here, so the form can only offer what the API accepts. Both submit paths -- the form's
 * own and the JSON tab's -- run the same checks() before sending anything.
 *
 * The examples are this app's own, because none of the repository's 32 briefs is a valid request
 * (docs/BRIEF_VALIDITY.md). */

type Tab = "form" | "json";

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

const ADVANCED_DEFAULTS: Advanced = {
  seed: 42,
  n_peptides: 100,
  ph: 7.4,
  cytotoxicity_cell_type: "DRAMP_aggregate",
  stages: Object.fromEntries(STAGES.map(([name]) => [name, true])),
};

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

function buildRequest(name: string, brief: BriefFields, advanced: Advanced): CreateJobRequest {
  const enabled = (stage: string) => ({ enabled: advanced.stages[stage] ?? true });
  return {
    /* The proxy replaces this with u<user id>:<uuid> before it reaches the API and keeps the text
       as this run's name. Sending the name is how the name is recorded. */
    request_id: name.trim() || "unnamed run",
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
  } as CreateJobRequest;
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

export function SubmitRun() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>("form");
  const [name, setName] = useState("");
  const [brief, setBrief] = useState<BriefFields>(EMPTY_BRIEF);
  const [advanced, setAdvanced] = useState<Advanced>(ADVANCED_DEFAULTS);
  const [json, setJson] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const request = useMemo(() => buildRequest(name, brief, advanced), [name, brief, advanced]);

  const submit = async (payload: CriticalPayload) => {
    setError(null);
    setProblems([]);
    let body: CreateJobRequest;
    if (payload.kind === "form") {
      const found = checks(brief, advanced);
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
      const candidate = parsed as CreateJobRequest;
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
      body = candidate;
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

  return (
    <>
      <Hero
        title="Start a run"
        sub={
          <>
            Describe the wound and what the peptide should do. The pipeline generates candidates,
            screens them and returns a ranked shortlist.
          </>
        }
        badge="Each run starts a GPU job"
      />

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

      {tab === "form" ? (
        <div className="grid">
          <div>
            <div className="card">
              <h2>The wound</h2>
              <div className="field">
                <label htmlFor="name">Name this run</label>
                <input
                  id="name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. diabetic foot ulcer, attempt 3"
                />
                <p className="help">
                  For your own history only. It never reaches the pipeline &mdash; the submission
                  id is generated for you.
                </p>
              </div>

              <div className="field">
                <label htmlFor="example">Start from an example</label>
                <select
                  id="example"
                  defaultValue=""
                  onChange={(event) => loadExample(event.target.value)}
                >
                  <option value="">Choose an example…</option>
                  {EXAMPLES.map((example) => (
                    <option key={example.id} value={example.id}>
                      {example.label}
                    </option>
                  ))}
                </select>
                <p className="help">
                  Fills the fields below, which you can then change. These are written for this
                  form; the briefs in <span className="mono">data/briefs/</span> are not valid API
                  requests (see Models &amp; health).
                </p>
              </div>

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
              <div className="toggles">
                {STAGES.map(([stage, label, note]) => {
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
      ) : (
        <div className="card">
          <h2>Advanced: the request as JSON</h2>
          <p className="help" style={{ marginTop: 0 }}>
            Everything the API accepts, including stage parameters this form does not show. Edit or
            paste, then submit. The same checks run either way.
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

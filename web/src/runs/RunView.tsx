import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import {
  STATE_EXPLANATION,
  cancelJob,
  isFinished,
  jobResults,
  jobStatus,
  serverRunState,
  type RunState,
} from "../api/peptide";
import type {
  AnyCandidateResponse,
  AnyJobResultsResponse,
  CandidateResponse,
  CosmeticCandidateResponse,
  JobStatusResponse,
  MetabolicCandidateResponse,
} from "../api/types";
import { resultsPipeline } from "../api/types";
import { ErrorBox, Hero, Kv, Loading, Pill, downloadJson, num, words } from "../components/ui";
import { lastRun } from "../workspace";

/* One run: what it is doing, and what it produced.
 *
 * The §7 rules this page exists to keep:
 * - Nothing is computed here that the API did not return. The API returns run_state itself;
 *   serverRunState() prefers it and only classifies the two older fields
 *   fields; every number shown is a field of a response.
 * - Withheld is not zero. A missing prediction renders "No value", never a dash and never 0 --
 *   a dash reads as "about zero", which for a cytotoxicity score is the opposite of the truth.
 * - No false order. The API returns a `ranking` per candidate; candidates without one are shown
 *   separately and never numbered, and shared positions are shown as ties.
 * - A score is not called a probability unless the API's own field name says so.
 * - vertex_state is always shown, because it is the only field that tells the truth about a
 *   worker that died without writing results.json. */

const POLL_MS = 20_000;

function StatePill({ state }: { state: RunState }) {
  const tone =
    state === "succeeded" ? "good" : state === "running" || state === "submitted" ? "muted" : "bad";
  const label = state === "abandoned" ? "no result written" : state;
  return <Pill tone={tone}>{label}</Pill>;
}

/** A value the API withheld. Never a blank, a dash, or 0. */
function NoValue({ why = "The API returned no value for this." }: { why?: string }) {
  return (
    <span className="help" title={why}>
      No value
    </span>
  );
}

function Scored({
  label,
  value,
  meaning,
  digits = 3,
  scale,
}: {
  label: string;
  value: number | null | undefined;
  meaning: string;
  digits?: number;
  /** Present only where the API's value genuinely lies on a known range. The bar is a reading
   *  aid for THAT value; it is never drawn for a number whose range is not known, because a bar
   *  implies a scale and inventing one would misrepresent the score. */
  scale?: [number, number];
}) {
  return (
    <div className="stat" title={meaning}>
      <div className="stat-head">
        <span>{label}</span>
        {value == null ? <NoValue /> : <b>{num(value, digits)}</b>}
      </div>
      {value != null && scale ? (
        <div className="stat-bar">
          <i
            style={{
              width: `${Math.max(0, Math.min(1, (value - scale[0]) / (scale[1] - scale[0]))) * 100}%`,
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

/* ONE SHELL, THREE BODIES. The three pipelines return three different candidate shapes that
   share a header (rank, sequence, mass, id, ties) and share nothing below it. The shell holds
   what is common so that adding a pipeline cannot quietly change how the other two are read. */
function CandidateShell({
  rank,
  sequence,
  id,
  facts,
  chips,
  children,
}: {
  rank: number | null | undefined;
  sequence: string | null | undefined;
  id: string;
  facts: ReactNode;
  chips: ReactNode[];
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="cand">
      <div className="cand-top">
        {rank == null ? (
          <span className="cand-rank none" title="The pipeline declined to rank this candidate">
            unranked
          </span>
        ) : (
          <span className="cand-rank">#{rank}</span>
        )}
        <div className="cand-main">
          <div className="cand-seq">
            {sequence ?? <NoValue why="The API returned no sequence for this candidate." />}
          </div>
          <div className="cand-facts" style={{ marginTop: 6 }}>
            {facts}
            <span className="mono" style={{ fontSize: 10, opacity: 0.65 }}>
              {id}
            </span>
          </div>
          {chips.length > 0 ? <div className="cand-sub">{chips}</div> : null}
        </div>
        <button
          className="btn outline"
          style={{ padding: "6px 11px", flex: "0 0 auto" }}
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? "Less" : "More"}
        </button>
      </div>
      {open ? (
        <div className="cand-body">
          <div className="cand-cols">{children}</div>
        </div>
      ) : null}
    </div>
  );
}

function tieChip(tiedWith: number): ReactNode[] {
  return tiedWith > 1
    ? [
        <Pill tone="warn" key="tie">
          tied &mdash; {tiedWith} share this position
        </Pill>,
      ]
    : [];
}

/** Flattened from the API's three separate lists, keeping the direction attached to each pathway.
 *  Nothing is computed here that the API did not report: a pathway appears only if it named it.
 *  Shared by the wound-healing and metabolic cards; the cosmetic response has no pathways. */
function pathwayChips(candidate: {
  activated_pathways?: string[] | null;
  inhibited_pathways?: string[] | null;
  undetermined_pathways?: string[] | null;
}): ReactNode[] {
  const pathways = [
    ...(candidate.activated_pathways ?? []).map((pathway) => ({
      pathway,
      direction: "activates" as const,
    })),
    ...(candidate.inhibited_pathways ?? []).map((pathway) => ({
      pathway,
      direction: "inhibits" as const,
    })),
    // Named rather than omitted. The model assigns a direction to every pathway, so a label
    // missing from both lists means it could not be called with confidence -- not that the
    // peptide leaves that pathway alone.
    ...(candidate.undetermined_pathways ?? []).map((pathway) => ({
      pathway,
      direction: "unclear" as const,
    })),
  ];
  if (pathways.length === 0) return [];
  return [
    /* The API reports activation and inhibition separately (pathway predictor v2). They are
       rendered with their direction rather than as one neutral list: a pathway the peptide
       INHIBITS shown identically to one it ACTIVATES would read as the opposite of what the
       model said. */
    ...pathways.map(({ pathway, direction }) => (
      <span className="chip" key={`${direction}:${pathway}`}>
        {direction} {words(pathway)}
      </span>
    )),
    /* The direction is the model's call, not a measurement, and at the shipped threshold of 0.5
       it has no "unsure" band. Said here rather than only in docs/PATHWAY_THRESHOLD.md, because
       this is where someone reads the result. */
    <span
      key="how-to-read"
      className="help"
      title={
        "The pathway model reports P(activator): it was trained on activators versus" +
        " inhibitors with the non-interacting examples dropped, so it cannot say a" +
        " peptide leaves a pathway alone. At the shipped threshold of 0.5 every" +
        " pathway is given a direction, including probabilities near a coin flip." +
        " Treat these as a weak signal, not a finding. See docs/PATHWAY_THRESHOLD.md."
      }
    >
      how to read these
    </span>,
  ];
}

/** Final score, evidence coverage, and the modules that produced nothing. Identical in the
 *  wound-healing and metabolic responses, so it is written once. */
function RankingEvidence({
  finalScore,
  evidenceCoverage,
  missingModules,
}: {
  finalScore: number | null | undefined;
  evidenceCoverage: number | null | undefined;
  missingModules: string[] | null | undefined;
}) {
  return (
    <div className="cand-col">
      <h4>Ranking evidence</h4>
      {/* The API reports how much of the evidence behind a rank actually existed. Showing a score
          without it invites the reader to treat a rank computed from half the modules as one
          computed from all of them. */}
      <Scored
        label="Final score"
        value={finalScore}
        meaning="The weighted multi-objective score from the ranking stage, after flag deductions."
        scale={[0, 1]}
      />
      <Scored
        label="Evidence coverage"
        value={evidenceCoverage}
        meaning="Activated modules that produced a score, over activated modules. Below 1 means the rank was computed from part of the intended evidence, with the remaining weights renormalised."
        scale={[0, 1]}
      />
      <div className="kv">
        <span>Modules with no score</span>
        <span>
          {missingModules == null ? (
            <NoValue why="The API reported no missing_modules field for this candidate." />
          ) : missingModules.length === 0 ? (
            "None — every activated module scored"
          ) : (
            missingModules.map((module) => (
              <span className="chip" key={module}>
                {words(module)}
              </span>
            ))
          )}
        </span>
      </div>
    </div>
  );
}

function WoundCandidate({
  candidate,
  tiedWith,
}: {
  candidate: CandidateResponse;
  tiedWith: number;
}) {
  return (
    <CandidateShell
      rank={candidate.final_score_rank}
      sequence={candidate.sequence}
      id={candidate.id}
      chips={[...tieChip(tiedWith), ...pathwayChips(candidate)]}
      facts={
        <>
          {candidate.molecular_weight != null ? (
            <span>
              mass <b>{num(candidate.molecular_weight, 1)} Da</b>
            </span>
          ) : null}
          {candidate.net_charge != null ? (
            <span>
              net charge <b>{num(candidate.net_charge, 1)}</b>
            </span>
          ) : null}
          {candidate.amp_probability != null ? (
            <span>
              antimicrobial <b>{num(candidate.amp_probability, 2)}</b>
            </span>
          ) : null}
        </>
      }
    >
      <div className="cand-col">
        <h4>Function</h4>
        <Scored
          label="Antimicrobial probability"
          value={candidate.amp_probability}
          meaning="P(antimicrobial peptide) from amp_classifier_v1. The API calls this a probability."
          scale={[0, 1]}
        />
        <Scored
          label="Anti-inflammatory probability"
          value={candidate.anti_inflammatory_probability}
          meaning="From anti_inflammatory_predictor_v1."
          scale={[0, 1]}
        />
        <Scored
          label="Proliferation probability"
          value={candidate.proliferation_probability}
          meaning="From proliferation_migration_predictor_v1."
          scale={[0, 1]}
        />
        <Scored
          label="Migration probability"
          value={candidate.migration_probability}
          meaning="From proliferation_migration_predictor_v1."
          scale={[0, 1]}
        />
        <Scored
          label="Angiogenic activity"
          value={candidate.angiogenic_activity}
          meaning="Score from angiogenic_activity_predictor_v1. The API does not call this a probability, so it is not labelled as one."
          scale={[0, 1]}
        />
      </div>

      <RankingEvidence
        finalScore={candidate.final_score}
        evidenceCoverage={candidate.evidence_coverage}
        missingModules={candidate.missing_modules}
      />

      <div className="cand-col">
        <h4>Safety &amp; developability</h4>
        <Scored
          label="Cytotoxicity score"
          value={candidate.cytotoxicity_probability}
          meaning="Score from cytotoxicity_predictor_v1, against the cell line this run selected. Lower is better."
          scale={[0, 1]}
        />
        <Scored
          label="Solubility score"
          value={candidate.solubility}
          meaning="From solubility_predictor_v1. Higher is better."
          scale={[0, 1]}
        />
        <Scored
          label="Aggregation tendency"
          value={candidate.aggregation_tendency}
          meaning="From aggregation_predictor_v1. Higher is more aggregation-prone."
          scale={[0, 1]}
        />
        <Scored
          label="Cleavage stability"
          value={candidate.cleavage_stability}
          meaning="From cleavage_site_predictor_v1. Higher is more stable."
          scale={[0, 1]}
        />
        <Scored
          label="Haemolysis pHC50"
          value={candidate.hemolytic_activity_phc50}
          meaning="pHC50 from hemolysis_predictor_v1. Higher is less haemolytic. No bar: the API states no range for this value."
        />
        <Scored
          label="Instability index"
          value={candidate.instability_index}
          meaning="Computed physicochemically in stage 5, not by a learned model. No bar: it is unbounded and can be negative."
          digits={2}
        />
        <div className="stat">
          <div className="stat-head">
            <span>Deamidation / oxidation risk</span>
            <b>
              {candidate.deamidation_risk ?? "no value"} /{" "}
              {candidate.oxidation_risk ?? "no value"}
            </b>
          </div>
        </div>
      </div>

      <div className="cand-col">
        <h4>MIC by pathogen (log10 µM)</h4>
        {Object.keys(candidate.log_mic_um ?? {}).length === 0 ? (
          <p className="stat-none">No MIC was returned for this candidate.</p>
        ) : (
          Object.entries(candidate.log_mic_um ?? {}).map(([organism, value]) => (
            <div className="pathogen" key={organism}>
              <i>{words(organism)}</i>
              <b>{num(value as number, 3)}</b>
            </div>
          ))
        )}
      </div>

      <div className="cand-col">
        <h4>Biofilm pMBIC by pathogen</h4>
        {Object.keys(candidate.pmbic ?? {}).length === 0 ? (
          <p className="stat-none">No biofilm potency was returned for this candidate.</p>
        ) : (
          Object.entries(candidate.pmbic ?? {}).map(([organism, value]) => (
            <div className="pathogen" key={organism}>
              <i>{words(organism)}</i>
              <b>{num(value as number, 3)}</b>
            </div>
          ))
        )}
        <p className="help" style={{ marginTop: 8 }}>
          The biofilm model scores more organisms than a brief may request.
        </p>
      </div>
    </CandidateShell>
  );
}

function MetabolicCandidate({
  candidate,
  tiedWith,
}: {
  candidate: MetabolicCandidateResponse;
  tiedWith: number;
}) {
  /* Every receptor the API said anything about, from all four of its per-receptor maps. Built as
     a union rather than from one map, so a receptor present in only one of them is still shown
     instead of silently dropping out of the table. */
  const receptors = [
    ...new Set([
      ...Object.keys(candidate.pec50 ?? {}),
      ...Object.keys(candidate.ec50_nm ?? {}),
      ...Object.keys(candidate.binding_pkd ?? {}),
      ...Object.keys(candidate.agonism_probability ?? {}),
    ]),
  ].sort();
  const unreliable = new Set(candidate.potency_unreliable_receptors ?? []);
  return (
    <CandidateShell
      rank={candidate.final_score_rank}
      sequence={candidate.sequence}
      id={candidate.id}
      chips={[
        ...tieChip(tiedWith),
        /* THE FLAG IS NOT A FOOTNOTE. The API marks receptors whose potency model cannot rank
           candidates, and the value is still returned and still counted, so a reader who sees
           the number must see this beside it rather than only in an expanded panel. */
        ...[...unreliable].map((receptor) => (
          <Pill tone="bad" key={`unreliable:${receptor}`}>
            {receptor} potency not predictive
          </Pill>
        )),
        ...pathwayChips(candidate),
      ]}
      facts={
        <>
          {candidate.molecular_weight != null ? (
            <span>
              mass <b>{num(candidate.molecular_weight, 1)} Da</b>
            </span>
          ) : null}
          {candidate.net_charge != null ? (
            <span>
              net charge <b>{num(candidate.net_charge, 1)}</b>
            </span>
          ) : null}
          {candidate.metabolic_activity_score != null ? (
            <span>
              activity <b>{num(candidate.metabolic_activity_score, 2)}</b>
            </span>
          ) : null}
        </>
      }
    >
      <div className="cand-col">
        <h4>Receptor potency</h4>
        {receptors.length === 0 ? (
          <p className="stat-none">The API returned no per-receptor potency for this candidate.</p>
        ) : (
          receptors.map((receptor) => (
            <div className="stat" key={receptor}>
              <div className="stat-head">
                <span>
                  {receptor}
                  {unreliable.has(receptor) ? (
                    <>
                      {" "}
                      <span
                        className="help"
                        title={
                          candidate.potency_unreliable_note ??
                          "The API flagged this receptor's potency prediction as not usable for ranking."
                        }
                      >
                        not predictive
                      </span>
                    </>
                  ) : null}
                </span>
                <b>
                  {candidate.pec50?.[receptor] == null ? (
                    <NoValue why={`No pEC50 was returned for ${receptor}.`} />
                  ) : (
                    <>pEC50 {num(candidate.pec50[receptor] as number, 2)}</>
                  )}
                </b>
              </div>
              <div className="cand-facts" style={{ marginTop: 4 }}>
                {candidate.ec50_nm?.[receptor] != null ? (
                  <span>
                    EC50 <b>{num(candidate.ec50_nm[receptor] as number, 2)} nM</b>
                  </span>
                ) : null}
                {candidate.binding_pkd?.[receptor] != null ? (
                  <span>
                    pKd <b>{num(candidate.binding_pkd[receptor] as number, 2)}</b>
                  </span>
                ) : null}
                {candidate.agonism_probability?.[receptor] != null ? (
                  <span>
                    P(agonist) <b>{num(candidate.agonism_probability[receptor] as number, 2)}</b>
                  </span>
                ) : null}
              </div>
            </div>
          ))
        )}
        {unreliable.size > 0 && candidate.potency_unreliable_note ? (
          <p className="help" style={{ marginTop: 8 }}>
            {candidate.potency_unreliable_note}
          </p>
        ) : null}
      </div>

      <div className="cand-col">
        <h4>Function</h4>
        <Scored
          label="Metabolic activity score"
          value={candidate.metabolic_activity_score}
          meaning="The stage 6 combined potency score. The API does not call this a probability, so it is not labelled as one."
          scale={[0, 1]}
        />
        <Scored
          label="Anti-inflammatory probability"
          value={candidate.anti_inflammatory_probability}
          meaning="From anti_inflammatory_predictor_v1."
          scale={[0, 1]}
        />
        <Scored
          label="Plasma half-life"
          value={candidate.half_life_minutes}
          meaning="Minutes, from half_life_predictor_v1. No bar: the API states no range for this value."
          digits={1}
        />
        <Scored
          label="pEC50 − pKd gap"
          value={candidate.pec50_pkd_gap}
          meaning="The API's own difference between predicted potency and predicted binding. No bar: it is a signed difference, not a score on a range."
          digits={2}
        />
      </div>

      <RankingEvidence
        finalScore={candidate.final_score}
        evidenceCoverage={candidate.evidence_coverage}
        missingModules={candidate.missing_modules}
      />

      <div className="cand-col">
        <h4>Safety &amp; developability</h4>
        <Scored
          label="Haemolysis probability"
          value={candidate.p_hemolytic}
          meaning="P(haemolytic) from hemolysis_predictor_v2."
          scale={[0, 1]}
        />
        <Scored
          label="Haemolysis pHC50"
          value={candidate.hemolytic_activity_phc50}
          meaning="pHC50 from hemolysis_predictor_v1. Higher is less haemolytic. No bar: the API states no range for this value."
        />
        <Scored
          label="Cytotoxicity score"
          value={candidate.cytotoxicity_probability}
          meaning="Score from cytotoxicity_predictor_v1, against the cell line this run selected. Lower is better."
          scale={[0, 1]}
        />
        <Scored
          label="Solubility score"
          value={candidate.solubility}
          meaning="From solubility_predictor_v1. Higher is better."
          scale={[0, 1]}
        />
        <Scored
          label="Aggregation tendency"
          value={candidate.aggregation_tendency}
          meaning="From aggregation_predictor_v1. Higher is more aggregation-prone."
          scale={[0, 1]}
        />
        <Scored
          label="Cleavage stability"
          value={candidate.cleavage_stability}
          meaning="From cleavage_site_predictor_v1. Higher is more stable."
          scale={[0, 1]}
        />
        <Scored
          label="Instability index"
          value={candidate.instability_index}
          meaning="Computed physicochemically in stage 5, not by a learned model. No bar: it is unbounded and can be negative."
          digits={2}
        />
        <div className="stat">
          <div className="stat-head">
            <span>Deamidation / oxidation risk</span>
            <b>
              {candidate.deamidation_risk ?? "no value"} /{" "}
              {candidate.oxidation_risk ?? "no value"}
            </b>
          </div>
        </div>
      </div>
    </CandidateShell>
  );
}

function CosmeticCandidate({
  candidate,
  tiedWith,
}: {
  candidate: CosmeticCandidateResponse;
  tiedWith: number;
}) {
  const components = Object.entries(candidate.component_scores ?? {});
  return (
    <CandidateShell
      rank={candidate.final_score_rank}
      sequence={candidate.sequence}
      id={candidate.id}
      chips={[
        ...tieChip(tiedWith),
        /* `not_screened` is the API saying a screen DID NOT RUN. Rendered as a flag rather than
           folded into the scores, because a screen that did not run is not a screen that
           passed. */
        ...(candidate.not_screened ?? []).map((screen) => (
          <Pill tone="warn" key={`not-screened:${screen}`}>
            {words(screen)} not screened
          </Pill>
        )),
        ...(candidate.safety_flags ?? []).map((flag) => (
          <Pill tone="bad" key={`safety:${flag}`}>
            {words(flag)}
          </Pill>
        )),
        ...(candidate.physchem_flags ?? []).map((flag) => (
          <span className="chip" key={`physchem:${flag}`}>
            {words(flag)}
          </span>
        )),
      ]}
      facts={
        <>
          {candidate.primary_function != null ? (
            <span>
              function <b>{words(candidate.primary_function)}</b>
            </span>
          ) : null}
          {candidate.molecular_weight != null ? (
            <span>
              mass <b>{num(candidate.molecular_weight, 1)} Da</b>
            </span>
          ) : null}
          {candidate.net_charge != null ? (
            <span>
              net charge <b>{num(candidate.net_charge, 1)}</b>
            </span>
          ) : null}
        </>
      }
    >
      <div className="cand-col">
        <h4>Function</h4>
        <Scored
          label="Antimicrobial probability"
          value={candidate.antimicrobial_probability}
          meaning="P(antimicrobial peptide) from amp_classifier_v1. The API calls this a probability."
          scale={[0, 1]}
        />
        <Scored
          label="Half-life (log10 seconds)"
          value={candidate.half_life_log10_seconds}
          meaning="From half_life_predictor_v1, in the log10 seconds the API returns. No bar: the API states no range."
          digits={2}
        />
        <div className="kv">
          <span>Functional screens reported</span>
          <span>
            {Object.keys(candidate.functional_scores ?? {}).length === 0 ? (
              <NoValue why="The API returned no functional_scores for this candidate." />
            ) : (
              Object.keys(candidate.functional_scores).map((name) => (
                <span className="chip" key={name}>
                  {words(name)}
                </span>
              ))
            )}
          </span>
        </div>
        <p className="help" style={{ marginTop: 8 }}>
          The per-screen detail the API returns under these names is in the exported results JSON;
          this page does not summarise it into a single number.
        </p>
      </div>

      <div className="cand-col">
        <h4>Ranking evidence</h4>
        <Scored
          label="Final score"
          value={candidate.final_score}
          meaning="The weighted multi-objective score from s11_cosmetic_ranking, after flag deductions."
          scale={[0, 1]}
        />
        <Scored
          label="Evidence coverage"
          value={candidate.evidence_coverage}
          meaning="Activated modules that produced a score, over activated modules. Below 1 means the rank was computed from part of the intended evidence."
          scale={[0, 1]}
        />
        {components.length === 0 ? (
          <p className="stat-none">The API returned no component scores for this candidate.</p>
        ) : (
          components.map(([component, score]) => (
            <div className="stat" key={component}>
              <div className="stat-head">
                <span>
                  {words(component)}
                  {score.screened ? null : (
                    <>
                      {" "}
                      <span
                        className="help"
                        title="The API reports this component as not screened: the screen did not run, so there is no score to read as a pass."
                      >
                        not screened
                      </span>
                    </>
                  )}
                </span>
                {score.score == null ? (
                  <NoValue
                    why={
                      score.screened
                        ? "The screen ran but the API returned no score."
                        : "The screen did not run, so the API returned no score."
                    }
                  />
                ) : (
                  <b>{num(score.score, 3)}</b>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="cand-col">
        <h4>Skin safety</h4>
        <div className="stat">
          <div className="stat-head">
            <span>Irritation tier</span>
            <b>{candidate.irritation_tier ?? "no value"}</b>
          </div>
        </div>
        <Scored
          label="Irritation score"
          value={candidate.irritation_score}
          meaning="From the stage 8 cosmetic irritation screen. Higher is more irritant."
          scale={[0, 1]}
        />
        <div className="stat">
          <div className="stat-head">
            <span>Skin sensitizer</span>
            <b>
              {candidate.skin_sensitizer == null
                ? "no value"
                : candidate.skin_sensitizer
                  ? "yes"
                  : "no"}
            </b>
          </div>
        </div>
        <Scored
          label="Skin sensitization probability"
          value={candidate.skin_sensitization_probability}
          meaning="From skin_sensitization_predictor_v1. The call above is this value against the model's own threshold."
          scale={[0, 1]}
        />
        <Scored
          label="Allergenicity probability"
          value={candidate.allergenicity_probability}
          meaning="From allergenicity_predictor_v1."
          scale={[0, 1]}
        />
        <Scored
          label="Haemolysis probability"
          value={candidate.hemolysis_probability}
          meaning="P(haemolytic) from hemolysis_predictor_v2."
          scale={[0, 1]}
        />
        <div className="kv">
          <span>Cytotoxicity cell lines reported</span>
          <span>
            {Object.keys(candidate.cytotoxicity ?? {}).length === 0 ? (
              <NoValue why="The API returned no cytotoxicity entries for this candidate." />
            ) : (
              Object.keys(candidate.cytotoxicity).map((line) => (
                <span className="chip" key={line}>
                  {words(line)}
                </span>
              ))
            )}
          </span>
        </div>
      </div>

      <div className="cand-col">
        <h4>Developability</h4>
        <Scored
          label="Solubility score"
          value={candidate.solubility}
          meaning="From solubility_predictor_v1. Higher is better."
          scale={[0, 1]}
        />
        <Scored
          label="Aggregation tendency"
          value={candidate.aggregation_tendency}
          meaning="From aggregation_predictor_v1. Higher is more aggregation-prone."
          scale={[0, 1]}
        />
        <Scored
          label="Cleavage stability"
          value={candidate.cleavage_stability}
          meaning="From cleavage_site_predictor_v1. Higher is more stable."
          scale={[0, 1]}
        />
        <Scored
          label="Hydrophobic moment"
          value={candidate.hydrophobic_moment}
          meaning="Computed physicochemically in stage 5, not by a learned model. No bar: the API states no range."
          digits={3}
        />
        <Scored
          label="Synthesis feasibility probability"
          value={candidate.synthesis_feasibility_probability}
          meaning="From the stage 9 cosmetic synthesis screen."
          scale={[0, 1]}
        />
        <div className="kv">
          <span>Physicochemical flags</span>
          <span>
            {candidate.physchem_flag_count == null ? (
              <NoValue why="The API reported no physchem_flag_count for this candidate." />
            ) : (
              candidate.physchem_flag_count
            )}
          </span>
        </div>
        <div className="kv">
          <span>Synthesis flags, by residue position</span>
          <span>
            {Object.keys(candidate.synthesis_flags ?? {}).length === 0 ? (
              "None reported"
            ) : (
              Object.entries(candidate.synthesis_flags).map(([flag, positions]) => (
                <span className="chip" key={flag}>
                  {words(flag)} @ {positions.join(", ")}
                </span>
              ))
            )}
          </span>
        </div>
      </div>
    </CandidateShell>
  );
}

/** Dispatches on the envelope's `pipeline` tag.
 *
 *  The cast is the one place this file has to assert something TypeScript cannot follow: the tag
 *  and the candidate come from the same response object, but that association is lost once the
 *  array is mapped. It is confined to this function, and every branch is a different component
 *  with a different prop type, so a shape that gains a field cannot silently render as another
 *  pipeline's. */
function CandidateCard({
  pipeline,
  candidate,
  tiedWith,
}: {
  pipeline: "wound_healing" | "metabolic" | "cosmetic";
  candidate: AnyCandidateResponse;
  tiedWith: number;
}) {
  if (pipeline === "metabolic") {
    return (
      <MetabolicCandidate
        candidate={candidate as MetabolicCandidateResponse}
        tiedWith={tiedWith}
      />
    );
  }
  if (pipeline === "cosmetic") {
    return (
      <CosmeticCandidate candidate={candidate as CosmeticCandidateResponse} tiedWith={tiedWith} />
    );
  }
  return <WoundCandidate candidate={candidate as CandidateResponse} tiedWith={tiedWith} />;
}

export function RunView() {
  const { jobId = "" } = useParams();
  const decoded = decodeURIComponent(jobId);
  /* EVERY PIECE OF FETCHED STATE CARRIES THE RUN IT BELONGS TO, and is read back only when that
     run is the one in the URL. Clearing state in an effect is not enough: react-router wraps a
     navigation in startTransition, so the component renders with the NEW route parameter before
     the clearing effect has committed, and that render pairs the new run's id with the previous
     run's stage and results -- exactly the mismatch review reported. Tying the data to its run
     makes the pairing impossible to express, rather than merely unlikely to be observed. */
  const [held, setHeld] = useState<{
    run: string;
    status: JobStatusResponse | null;
    results: AnyJobResultsResponse | null;
    notYet: boolean;
    error: unknown;
  }>({ run: decoded, status: null, results: null, notYet: false, error: null });

  const mine = held.run === decoded ? held : null;
  const status = mine?.status ?? null;
  const results = mine?.results ?? null;
  const notYet = mine?.notYet ?? false;
  const error = mine?.error ?? null;

  /** Apply a change only if the run it is for is still the one on screen. */
  const update = useCallback(
    (run: string, change: Partial<Omit<typeof held, "run">>) =>
      setHeld((prev) =>
        prev.run === run
          ? { ...prev, ...change }
          : { run, status: null, results: null, notYet: false, error: null, ...change },
      ),
    [],
  );

  const setStatus = useCallback(
    (run: string, body: JobStatusResponse | null) => update(run, { status: body }),
    [update],
  );
  const setResults = useCallback(
    (run: string, body: AnyJobResultsResponse | null) => update(run, { results: body }),
    [update],
  );
  const setNotYet = useCallback((run: string, v: boolean) => update(run, { notYet: v }), [update]);
  const setError = useCallback((run: string, v: unknown) => update(run, { error: v }), [update]);
  const [cancelling, setCancelling] = useState(false);
  const [cancelNote, setCancelNote] = useState<string | null>(null);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (decoded) lastRun.set(decoded);
  }, [decoded]);

  /* The run currently on screen. A response is applied only if it is still for THIS run.
     Without it, navigating straight from one run to another lets the abandoned run's late
     response land after the new one's and overwrite the page -- reported by review, and
     reproduced in src/test/runview-switch.test.tsx. */
  const showing = useRef(decoded);

  /* The route parameter changes without remounting. `held` is keyed by run so a stale value can
     never be READ, and this effect discards it so it is not retained either. The guard above is
     what makes correctness independent of when this runs. */
  useEffect(() => {
    showing.current = decoded;
    setHeld({ run: decoded, status: null, results: null, notYet: false, error: null });
    setCheckedAt(null);
  }, [decoded]);

  const poll = useCallback(
    async (signal: AbortSignal) => {
      const mine = decoded;
      const stale = () => signal.aborted || showing.current !== mine;
      try {
        const { body } = await jobStatus(mine, signal);
        if (stale()) return null;
        setStatus(mine, body);
        setCheckedAt(new Date().toISOString());
        const state = serverRunState(body);
        if (state === "succeeded") {
          const { status: code, body: got } = await jobResults(mine, signal);
          if (stale()) return null;
          if (code === 404) setNotYet(mine, true);
          else setResults(mine, got);
        }
        return state;
      } catch (problem) {
        /* An abort is this component tidying up, not something to show the reader. */
        if (stale() || (problem as Error)?.name === "AbortError") return null;
        setError(mine, problem);
        return null;
      }
    },
    [decoded],
  );

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    const tick = async () => {
      const state = await poll(controller.signal);
      if (!live) return;
      /* Stop polling once there is nothing left to learn. */
      if (state && !isFinished(state)) {
        timer.current = window.setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      live = false;
      controller.abort();
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [poll]);

  if (!status && !error) return <Loading>Reading this run…</Loading>;

  /* No status at all. Everything below describes a run; rendering it would invent one -- the
     default state is "submitted", so a run that is not yours, or does not exist, would appear as
     accepted by Vertex and still working, complete with a Cancel button. Found by walking the
     app as a second account (docs/WEB_WALKTHROUGH.md). */
  if (!status) {
    return (
      <>
        <Hero title="This run is not available" sub={<span className="mono">{decoded}</span>} />
        <div className="card">
          <ErrorBox error={error} />
          <p className="help">
            You will see the same answer whether the run belongs to another account or never
            existed &mdash; which is deliberate, so this page cannot be used to find out which.
          </p>
          <div className="btns">
            <Link className="btn secondary" to="/runs">
              Your runs
            </Link>
            <Link className="btn secondary" to="/">
              Start a run
            </Link>
          </div>
        </div>
      </>
    );
  }

  const state = serverRunState(status);
  /* `candidates` is the list; `ranked_candidates` and `insufficient_evidence_candidates` are the
     API's own COUNTS (api.py computes them from each candidate's ranking.status). The grouping
     below is for display; the counts shown are always the API's, never re-derived here. */
  /* The pipeline tag is read ONCE, here, and passed down. resultsPipeline() answers
     "wound_healing" for a response that carries no tag, which every captured fixture does. */
  const pipeline = resultsPipeline(results);
  const all: AnyCandidateResponse[] = results?.candidates ?? [];
  const ranked = all.filter((one) => one.final_score_rank != null);
  const unranked = all.filter((one) => one.final_score_rank == null);
  const tallies = new Map<number, number>();
  for (const one of ranked) {
    tallies.set(one.final_score_rank!, (tallies.get(one.final_score_rank!) ?? 0) + 1);
  }

  return (
    <>
      <Hero
        title={status?.stage && status.stage !== "pending" ? `Run — ${words(status.stage)}` : "Run"}
        sub={<span className="mono">{decoded}</span>}
        badge={<StatePill state={state} />}
      />

      <ErrorBox error={error} />

      <div className="card">
        <h2>What this run is doing</h2>
        <div className={`status${state === "succeeded" ? " good" : isFinished(state) ? " bad" : ""}`}>
          {STATE_EXPLANATION[state]}
        </div>
        <dl className="kv" style={{ marginTop: 14 }}>
          <Kv k="status (from the worker)" v={status?.status ?? "no value"} />
          <Kv k="stage (last written)" v={status?.stage ? words(status.stage) : "no value"} />
          <Kv k="vertex_state (live from Vertex)" v={status?.vertex_state ?? "no value"} />
          <Kv k="error reported" v={status?.error ?? "none"} />
          <Kv k="checked at" v={checkedAt ?? "not yet"} />
        </dl>
        <p className="footer-note">
          Two fields, because they can disagree. <span className="mono">status</span> comes only
          from the file the worker writes; <span className="mono">vertex_state</span> is queried
          live. A worker that stops before writing anything leaves{" "}
          <span className="mono">status</span> at &ldquo;pending&rdquo; permanently, which is why
          both are shown and why this page does not summarise them into one word.
        </p>
        {!isFinished(state) ? (
          <p className="help">
            Checking again every {POLL_MS / 1000} seconds. Safe to close this tab &mdash; the run
            keeps going and stays in your history.
          </p>
        ) : null}
        <div className="btns">
          {!isFinished(state) ? (
            <button
              className="btn outline"
              disabled={cancelling}
              onClick={async () => {
                if (!window.confirm("Cancel this run? Its GPU time so far is not recoverable, and no results will be produced.")) return;
                setCancelling(true);
                try {
                  await cancelJob(decoded);
                  setCancelNote("Cancellation requested. Vertex may take a moment to report it.");
                  /* Its own controller: this refresh is not the polling loop's, and must not be
                     cancelled by the loop's cleanup or cancel the loop's request. */
                  await poll(new AbortController().signal);
                } catch (problem) {
                  setError(decoded, problem);
                } finally {
                  setCancelling(false);
                }
              }}
            >
              {cancelling ? "Cancelling…" : "Cancel this run"}
            </button>
          ) : null}
          <Link className="btn secondary" to="/runs">
            All runs
          </Link>
          <Link className="btn secondary" to="/">
            Start another
          </Link>
        </div>
        {cancelNote ? <p className="help">{cancelNote}</p> : null}
      </div>

      {notYet ? (
        <div className="card">
          <h2>Results</h2>
          <div className="status">
            Vertex reports this job finished and the worker recorded success, but the results file
            is not readable yet. This is usually a moment&rsquo;s lag while it is written. Nothing
            is lost &mdash; check again shortly.
          </div>
        </div>
      ) : null}

      {results ? (
        <>
          <div className="card">
            <h2>Shortlist</h2>
            <div className="metric-grid">
              <div className="metric">
                <small>Final candidates</small>
                <b>{results.n_final ?? <NoValue why="The worker reported no final count." />}</b>
              </div>
              <div className="metric">
                <small>Ranked</small>
                <b>{results.ranked_candidates ?? <NoValue />}</b>
              </div>
              <div className="metric">
                <small>Insufficient evidence</small>
                <b>{results.insufficient_evidence_candidates ?? <NoValue />}</b>
              </div>
            </div>
            <p className="footer-note">
              Every count here is the API&rsquo;s own field, not counted from the list.
              &ldquo;Insufficient evidence&rdquo; means the pipeline declined to rank that
              candidate; it is not a low rank, and it is listed below without a position rather
              than placed last.
            </p>
            {results.ranked_candidates != null && results.ranked_candidates !== ranked.length ? (
              <div className="status">
                The API reports {results.ranked_candidates} ranked candidate
                {results.ranked_candidates === 1 ? "" : "s"}, but {ranked.length} of the{" "}
                {all.length} returned carry a rank. Both figures are shown as given; this app does
                not reconcile them.
              </div>
            ) : null}
            <div className="btns">
              <button
                className="btn secondary"
                onClick={() => downloadJson(`run-${results.run_id}.json`, results)}
              >
                Export results JSON
              </button>
            </div>
          </div>

          <div className="card">
            <h2>Ranked candidates ({ranked.length})</h2>
            {ranked.length === 0 ? (
              <p className="help">The pipeline ranked none of the candidates it returned.</p>
            ) : (
              ranked
                .slice()
                .sort((a, b) => (a.final_score_rank ?? 0) - (b.final_score_rank ?? 0))
                .map((candidate) => (
                  <CandidateCard
                    key={candidate.id}
                    pipeline={pipeline}
                    candidate={candidate}
                    tiedWith={tallies.get(candidate.final_score_rank!) ?? 1}
                  />
                ))
            )}
          </div>

          {unranked.length > 0 ? (
            <div className="card">
              <h2>Returned without a rank ({unranked.length})</h2>
              <p className="help" style={{ marginTop: 0 }}>
                Deliberately unordered. The pipeline did not rank these, so this app does not
                either.
              </p>
              {unranked.map((candidate) => (
                <CandidateCard
                  key={candidate.id}
                  pipeline={pipeline}
                  candidate={candidate}
                  tiedWith={1}
                />
              ))}
            </div>
          ) : null}

          {results.component_stats && Object.keys(results.component_stats).length > 0 ? (
            <div className="card">
              <h2>Ranking components</h2>
              <p className="help" style={{ marginTop: 0 }}>
                Spread of each component across ranked candidates, as the API computed it.
                Candidates with no normalised score are excluded rather than counted as zero.
              </p>
              <table className="runs">
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Min</th>
                    <th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.component_stats).map(([component, stats]) => {
                    const s = stats as Record<string, number | null>;
                    return (
                      <tr key={component}>
                        <td>{words(component)}</td>
                        {(["mean", "median", "min", "max"] as const).map((key) => (
                          <td className="num mono" key={key}>
                            {s[key] == null ? <NoValue /> : num(s[key], 3)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
        </>
      ) : null}

      <div className="card">
        <h2>Provenance</h2>
        <p className="help" style={{ marginTop: 0 }}>
          The API returns no model identity with a run &mdash; not a version, not a fingerprint.
          What was serving at the time is therefore not recorded per run, and this page will not
          imply otherwise. <Link to="/models">Models &amp; health</Link> shows what is serving{" "}
          <b>now</b>, which is the closest honest answer available.
        </p>
      </div>
    </>
  );
}

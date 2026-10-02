import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  STATE_EXPLANATION,
  cancelJob,
  isFinished,
  jobResults,
  jobStatus,
  runState,
  type RunState,
} from "../api/peptide";
import type { CandidateResponse, JobResultsResponse, JobStatusResponse } from "../api/types";
import { ErrorBox, Hero, Kv, Loading, Pill, downloadJson, num, words } from "../components/ui";
import { lastRun } from "../workspace";

/* One run: what it is doing, and what it produced.
 *
 * The §7 rules this page exists to keep:
 * - Nothing is computed here that the API did not return. runState() classifies two returned
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

function Candidate({
  candidate,
  tiedWith,
}: {
  candidate: CandidateResponse;
  tiedWith: number;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="cand">
      <div className="cand-top">
        {candidate.ranking == null ? (
          <span className="cand-rank none" title="The pipeline declined to rank this candidate">
            unranked
          </span>
        ) : (
          <span className="cand-rank">#{candidate.ranking}</span>
        )}
        <div className="cand-main">
          <div className="cand-seq">
            {candidate.sequence ?? (
              <NoValue why="The API returned no sequence for this candidate." />
            )}
          </div>
          <div className="cand-facts" style={{ marginTop: 6 }}>
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
            <span className="mono" style={{ fontSize: 10, opacity: 0.65 }}>
              {candidate.id}
            </span>
          </div>
          {(candidate.engaged_pathways ?? []).length > 0 || tiedWith > 1 ? (
            <div className="cand-sub">
              {tiedWith > 1 ? (
                <Pill tone="warn">tied &mdash; {tiedWith} share this position</Pill>
              ) : null}
              {(candidate.engaged_pathways ?? []).map((pathway) => (
                <span className="chip" key={pathway}>
                  {words(pathway)}
                </span>
              ))}
            </div>
          ) : null}
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
          <div className="cand-cols">
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
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function RunView() {
  const { jobId = "" } = useParams();
  const decoded = decodeURIComponent(jobId);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [results, setResults] = useState<JobResultsResponse | null>(null);
  const [notYet, setNotYet] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelNote, setCancelNote] = useState<string | null>(null);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (decoded) lastRun.set(decoded);
  }, [decoded]);

  const poll = useCallback(async () => {
    try {
      const { body } = await jobStatus(decoded);
      setStatus(body);
      setCheckedAt(new Date().toISOString());
      const state = runState(body.status, body.vertex_state);
      if (state === "succeeded") {
        const { status: code, body: got } = await jobResults(decoded);
        if (code === 404) setNotYet(true);
        else setResults(got);
      }
      return state;
    } catch (problem) {
      setError(problem);
      return null;
    }
  }, [decoded]);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      const state = await poll();
      if (!live) return;
      /* Stop polling once there is nothing left to learn. */
      if (state && !isFinished(state)) {
        timer.current = window.setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      live = false;
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

  const state = runState(status.status, status.vertex_state);
  /* `candidates` is the list; `ranked_candidates` and `insufficient_evidence_candidates` are the
     API's own COUNTS (api.py computes them from each candidate's ranking.status). The grouping
     below is for display; the counts shown are always the API's, never re-derived here. */
  const all = results?.candidates ?? [];
  const ranked = all.filter((one) => one.ranking != null);
  const unranked = all.filter((one) => one.ranking == null);
  const tallies = new Map<number, number>();
  for (const one of ranked) {
    tallies.set(one.ranking!, (tallies.get(one.ranking!) ?? 0) + 1);
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
                  await poll();
                } catch (problem) {
                  setError(problem);
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
                .sort((a, b) => (a.ranking ?? 0) - (b.ranking ?? 0))
                .map((candidate) => (
                  <Candidate
                    key={candidate.id}
                    candidate={candidate}
                    tiedWith={tallies.get(candidate.ranking!) ?? 1}
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
                <Candidate key={candidate.id} candidate={candidate} tiedWith={1} />
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

import { useEffect, useState } from "react";
import { call } from "../api/client";
import { ErrorBox, Hero, Kv, Loading, Pill } from "../components/ui";

/* What is serving, and what it will not answer.
 *
 * This page is as honest as the API allows, and no more. It says plainly that the digests cover
 * predictor code rather than weights, that 7 of 15 predictors ship no model card, and that no
 * model identity is attached to a run -- because a model page that implies reproducibility the
 * backend cannot deliver is worse than no model page. The caveat text is the API's own; this
 * renders it rather than rewording it. */

interface ModelEntry {
  name: string;
  version: string;
  predictor_sha256: string | null;
  model_card: Record<string, unknown> | null;
  model_card_present: boolean;
}

interface Manifest {
  models: ModelEntry[];
  weights_source: string | null;
  caveat: string;
}

interface Health {
  ok: boolean;
  settings_missing: string[];
  note: string;
}

function Card({ entry }: { entry: ModelEntry }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="axis">
      <div className="axis-head">
        <b>{entry.name}</b>
        <div className="chips">
          <span className="chip">{entry.version}</span>
          {entry.model_card_present ? (
            <Pill tone="good">documented</Pill>
          ) : (
            <Pill tone="warn">no model card</Pill>
          )}
        </div>
      </div>
      <dl className="kv">
        <Kv k="predictor.py sha256" v={entry.predictor_sha256 ?? "no value"} />
      </dl>
      {entry.model_card_present ? (
        <>
          <div className="btns">
            <button className="btn outline" style={{ padding: "5px 9px" }} onClick={() => setOpen(!open)}>
              {open ? "Hide model card" : "Show model card"}
            </button>
          </div>
          {open ? (
            <pre className="mono" style={{ whiteSpace: "pre-wrap", marginTop: 10 }}>
              {JSON.stringify(entry.model_card, null, 2)}
            </pre>
          ) : null}
        </>
      ) : (
        <p className="why">
          This predictor ships no model card, so its training data, applicability domain and
          calibration are not documented here. Not checked is not the same as passed.
        </p>
      )}
    </div>
  );
}

export function ModelsAndHealth() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    call<Manifest>("/api/peptide/api/v1/models")
      .then(({ body }) => setManifest(body))
      .catch(setError);
    call<Health>("/api/peptide/healthz")
      .then(({ body }) => setHealth(body))
      .catch(() => setHealth(null));
  }, []);

  if (!manifest && !error) return <Loading>Asking the job API what it is running…</Loading>;

  const documented = manifest?.models.filter((one) => one.model_card_present).length ?? 0;
  const total = manifest?.models.length ?? 0;

  return (
    <>
      <Hero
        title="Models & health"
        sub="What this deployment is running, and what it does not tell you."
        badge={health?.ok ? <Pill tone="good">API answering</Pill> : <Pill tone="bad">API unreachable</Pill>}
      />
      <ErrorBox error={error} />

      <div className="card">
        <h2>Read this first</h2>
        <div className="status">
          <b>No run carries a model identity.</b> The job API returns no version or fingerprint
          with a run&rsquo;s results, so it is not possible to say from a stored result which models
          produced it. This page describes what is serving <b>now</b>. If a predictor is updated,
          past results become unattributable.
        </div>
        <p className="footer-note" style={{ marginTop: 12 }}>
          {manifest?.caveat}
        </p>
      </div>

      <div className="card">
        <h2>Service</h2>
        <div className="metric-grid">
          <div className="metric">
            <small>Predictors</small>
            <b>{total}</b>
          </div>
          <div className="metric">
            <small>With a model card</small>
            <b>
              {documented} of {total}
            </b>
          </div>
          <div className="metric">
            <small>Settings missing</small>
            <b>{health ? health.settings_missing.length : "unknown"}</b>
          </div>
        </div>
        <dl className="kv" style={{ marginTop: 14 }}>
          <Kv k="Weights synced from" v={manifest?.weights_source ?? "not configured"} />
        </dl>
        {health && health.settings_missing.length > 0 ? (
          <div className="status bad" style={{ marginTop: 12 }}>
            The job API is missing {health.settings_missing.length} required setting
            {health.settings_missing.length === 1 ? "" : "s"}:{" "}
            <span className="mono">{health.settings_missing.join(", ")}</span>. Submitting a run
            will fail with 503 until these are set.
          </div>
        ) : null}
        {health ? <p className="footer-note">{health.note}</p> : null}
      </div>

      <div className="card">
        <h2>What this platform will not answer</h2>
        <ul className="unordered">
          <li>
            Pathogens other than <span className="mono">Escherichia_coli</span>,{" "}
            <span className="mono">Staphylococcus_aureus</span> and{" "}
            <span className="mono">Pseudomonas_aeruginosa</span>. A pathogen outside that list is
            skipped, not scored — it is not an error and produces no warning in the results.
          </li>
          <li>
            Peptides shorter than 6 or longer than 50 residues. The request is refused rather than
            scored out of domain.
          </li>
          <li>
            Wound contexts and desired functions outside the API&rsquo;s own vocabularies. The
            example briefs in <span className="mono">data/briefs/</span> use many terms it rejects,
            so they are not offered in the form.
          </li>
          <li>
            Whether a candidate works. Every number here is a model&rsquo;s estimate on a
            computationally generated sequence. None of it substitutes for wet-lab validation.
          </li>
        </ul>
      </div>

      <div className="card">
        <h2>Predictors ({total})</h2>
        {manifest?.models.map((entry) => (
          <Card key={entry.name} entry={entry} />
        ))}
      </div>
    </>
  );
}

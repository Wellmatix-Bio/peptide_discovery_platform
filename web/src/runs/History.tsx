import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { hideHistoryEntry, history, serverRunState } from "../api/peptide";
import type { HistoryPage } from "../api/types";
import { ErrorBox, Hero, Loading, Pill, words } from "../components/ui";

/* Every run this account has submitted.
 *
 * The rows come from the accounts service, which recorded them by observing responses -- so each
 * row is a run the API actually accepted, and no endpoint can invent one. They also go STALE: the
 * service does not poll on anyone's behalf, so a row is as current as the last time its owner
 * looked. The page prints the service's own staleness note rather than paraphrasing it, and shows
 * observed_at on every row. */

const PAGE = 25;

export function History() {
  const navigate = useNavigate();
  const [page, setPage] = useState<HistoryPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setBusy(true);
    history({ limit: PAGE, offset })
      .then(({ body }) => setPage(body))
      .catch(setError)
      .finally(() => setBusy(false));
  }, [offset]);

  useEffect(load, [load]);

  if (!page && !error) return <Loading>Reading your runs…</Loading>;

  return (
    <>
      <Hero
        title="Run history"
        sub={page ? `${page.total} run${page.total === 1 ? "" : "s"} on this account` : undefined}
        badge={<Link className="btn primary" to="/">Start a run</Link>}
      />
      <ErrorBox error={error} />

      {page && page.total === 0 ? (
        <div className="card">
          <h2>Nothing here yet</h2>
          <p>
            Runs appear here once you submit one. <Link to="/">Start a run</Link>.
          </p>
          <p className="help">{page.note}</p>
        </div>
      ) : null}

      {page && page.total > 0 ? (
        <div className="card">
          <table className="runs">
            <thead>
              <tr>
                <th>Name</th>
                <th>State</th>
                <th>Stage</th>
                <th>Submitted</th>
                <th>Last seen</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {page.entries.map((entry) => {
                const state = serverRunState(entry);
                const tone =
                  state === "succeeded"
                    ? "good"
                    : state === "running" || state === "submitted"
                      ? "muted"
                      : "bad";
                const open = () =>
                  entry.resource_id &&
                  navigate(`/runs/${encodeURIComponent(entry.resource_id)}`);
                return (
                  <tr key={entry.id} className={entry.resource_id ? "clickable" : undefined}>
                    <td onClick={open}>
                      <b>{entry.run_name || "unnamed run"}</b>
                      <br />
                      <span className="help">{entry.summary}</span>
                    </td>
                    <td onClick={open}>
                      <Pill tone={tone}>{state === "abandoned" ? "no result" : state}</Pill>
                    </td>
                    <td onClick={open}>{entry.stage ? words(entry.stage) : "—"}</td>
                    <td onClick={open} className="num">
                      {entry.created_at}
                    </td>
                    <td onClick={open} className="num">
                      {entry.observed_at}
                    </td>
                    <td>
                      <button
                        className="btn outline"
                        style={{ padding: "5px 9px" }}
                        onClick={async () => {
                          if (
                            !window.confirm(
                              "Hide this run from your history?\n\nThis removes only your history entry. The run's files in Cloud Storage are not deleted — this app cannot delete them.",
                            )
                          ) {
                            return;
                          }
                          try {
                            await hideHistoryEntry(entry.id);
                            load();
                          } catch (problem) {
                            setError(problem);
                          }
                        }}
                      >
                        Hide
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="btns">
            <button
              className="btn secondary"
              disabled={offset === 0 || busy}
              onClick={() => setOffset(Math.max(0, offset - PAGE))}
            >
              Newer
            </button>
            <button
              className="btn secondary"
              disabled={busy || offset + PAGE >= page.total}
              onClick={() => setOffset(offset + PAGE)}
            >
              Older
            </button>
          </div>

          <p className="footer-note">{page.note}</p>
          <p className="footer-note">
            <b>Last seen</b> is when this service last saw a response about the run pass through for
            your account. {page.staleness}
          </p>
        </div>
      ) : null}
    </>
  );
}

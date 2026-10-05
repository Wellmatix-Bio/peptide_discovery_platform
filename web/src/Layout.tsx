import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { call } from "./api/client";
import { useAuth } from "./auth/AuthContext";
import { lastRun } from "./workspace";

/* The source project's shell: top bar, dark side navigation, main column.
 *
 * The top bar shows what is serving, from GET /api/peptide/api/v1/models. Note what it does NOT
 * show: a single "model fingerprint". This platform runs many predictors, not one, and the API can
 * only digest their code and not their weights (docs/BASELINE.md), so one reassuring hash in the
 * corner would be a claim the backend cannot support. The count links to the model page, where
 * the caveat is stated in full. */
export function Layout() {
  const { user, signOut } = useAuth();
  const [serving, setServing] = useState<string>("connecting…");
  const [loggedOutNote, setLoggedOutNote] = useState<string | null>(null);
  const location = useLocation();
  const openRun = lastRun.get();

  useEffect(() => {
    call<{ models: unknown[] }>("/api/peptide/api/v1/models")
      .then(({ body }) => setServing(`${body.models.length} predictors`))
      .catch(() => setServing("job API unreachable"));
  }, []);

  useEffect(() => {
    // Block-bodied: an arrow body returns the value of scrollTo, which React treats as a
    // cleanup function and then throws on.
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  const nav = (to: string, label: string, enabled = true) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        "nav" + (isActive ? " active" : "") + (enabled ? "" : " disabled")
      }
      end
    >
      {label}
    </NavLink>
  );

  return (
    <>
      <div className="top">
        <div className="brand">
          WELLMATIX <span>Peptide Discovery</span>
        </div>
        <div className="tag">Computational shortlist &bull; wet-lab validation required</div>
        <Link
          to="/models"
          className="tag"
          style={{ marginLeft: "auto", background: "#f2f6f8", color: "#5c7382" }}
          title="What is serving. No single fingerprint: the API reports one entry per predictor, and weights are not digested."
        >
          {serving}
        </Link>
        <div className="user">
          <Link to="/account">{user?.email}</Link>
          <button
            className="btn outline"
            style={{ padding: "6px 10px" }}
            onClick={async () => setLoggedOutNote(await signOut())}
          >
            Sign out
          </button>
        </div>
      </div>
      <div className="shell">
        <aside>
          <p className="small">Runs</p>
          {nav("/", "Start a run")}
          {nav("/runs", "Run history")}
          {nav(openRun ? `/runs/${encodeURIComponent(openRun)}` : "/runs", "Current run", !!openRun)}
          <p className="small" style={{ marginTop: 22 }}>
            This deployment
          </p>
          {nav("/models", "Models & health")}
          <p className="small" style={{ marginTop: 22 }}>
            Account
          </p>
          {nav("/account", "Account")}
        </aside>
        <div className="main">
          {loggedOutNote ? (
            <div className="status" style={{ marginBottom: 16 }}>
              {loggedOutNote}
            </div>
          ) : null}
          <Outlet />
        </div>
      </div>
    </>
  );
}

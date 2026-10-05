import { useState } from "react";
import { call } from "../api/client";
import type { Session } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ErrorBox, Hero, Kv } from "../components/ui";

/* The account page. Two things it must get right:
   - A wrong current password answers 403, NOT 401, so changing a password with a typo does not
     sign the user out mid-task. The client's 401 rule is what makes that distinction matter.
   - A successful change bumps password_version server-side, which revokes every existing token
     including this tab's. The response carries a replacement, which is stored immediately. */

export function Account() {
  const { user, signIn, signOut } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [done, setDone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [signOutNote, setSignOutNote] = useState<string | null>(null);

  const change = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setDone(null);
    setBusy(true);
    try {
      const { body } = await call<Session>("/auth/change-password", {
        method: "POST",
        body: { current_password: current, new_password: next },
      });
      /* Store the replacement token before anything else: the old one is already dead. */
      signIn(body);
      setCurrent("");
      setNext("");
      setDone(
        "Password changed. Every other signed-in session for this account has been revoked; this" +
          " tab was given a replacement token.",
      );
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Hero title="Account" sub={user?.email} />

      <div className="grid">
        <div>
          <div className="card">
            <h2>Change password</h2>
            <form onSubmit={change}>
              <div className="field">
                <label htmlFor="current">Current password</label>
                <input
                  id="current"
                  type="password"
                  value={current}
                  onChange={(event) => setCurrent(event.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="next">New password</label>
                <input
                  id="next"
                  type="password"
                  value={next}
                  onChange={(event) => setNext(event.target.value)}
                  autoComplete="new-password"
                  required
                  minLength={12}
                />
                <p className="help">At least 12 characters.</p>
              </div>
              <ErrorBox error={error} />
              {done ? <div className="status good">{done}</div> : null}
              <div className="btns">
                <button className="btn primary" type="submit" disabled={busy}>
                  {busy ? "Changing…" : "Change password"}
                </button>
              </div>
              <p className="help">
                Changing your password is the only way to revoke sessions. A wrong current password
                is refused without signing you out.
              </p>
            </form>
          </div>
        </div>

        <div>
          <div className="card">
            <h2>This account</h2>
            <dl className="kv">
              <Kv k="Email" v={user?.email ?? "—"} plain />
              <Kv k="Account id" v={String(user?.id ?? "—")} />
              <Kv k="Created" v={user?.created_at ?? "—"} plain />
            </dl>
          </div>

          <div className="card">
            <h2>What signing out does</h2>
            <p>
              It discards the token stored in this browser. It does <b>not</b> revoke anything on
              the server: sessions are stateless signed tokens, so any other copy of your token
              stays valid until it expires.
            </p>
            <p className="help">
              To revoke every session, change your password. That bumps a counter embedded in every
              token, which is the only real revocation this service has.
            </p>
            <div className="btns">
              <button
                className="btn outline"
                onClick={async () => setSignOutNote(await signOut())}
              >
                Sign out of this browser
              </button>
            </div>
            {signOutNote ? <p className="help">{signOutNote}</p> : null}
          </div>

          <div className="card">
            <h2>Lost your recovery code?</h2>
            <p className="help">
              A code is shown once, when it is issued, and stored only as a hash — nobody can look
              yours up. Changing your password above issues a new one.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { call } from "../api/client";
import type { Captcha, Registration, Session } from "../api/types";
import { ErrorBox } from "../components/ui";
import { useAuth } from "./AuthContext";

/* Sign in, register, and recovery. Three rules these pages exist to honour:
   - ONE message for any failed sign-in. The server sends one; this renders it and adds nothing.
   - The captcha is an <img src="data:image/svg+xml,…">, never injected markup. The SVG comes from
     the server and is rendered as an image so it cannot execute anything.
   - A recovery code is shown ONCE. Continue stays disabled until it has been copied or
     downloaded, and leaving the page warns. */

function Shell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="brand">
          WELLMATIX <span>Peptide Discovery</span>
        </div>
        <div className="card">
          <h2>{title}</h2>
          {children}
        </div>
      </div>
    </div>
  );
}

/** The captcha challenge, refetched after any error so a spent nonce is never retried. */
function useCaptcha() {
  const [captcha, setCaptcha] = useState<Captcha | null>(null);
  const [answer, setAnswer] = useState("");
  const refresh = useCallback(() => {
    setAnswer("");
    call<Captcha>("/auth/captcha", { session: false })
      .then(({ body }) => setCaptcha(body))
      .catch(() => setCaptcha(null));
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);
  return { captcha, answer, setAnswer, refresh };
}

function CaptchaField({
  captcha,
  answer,
  setAnswer,
  refresh,
}: ReturnType<typeof useCaptcha> & Record<string, unknown>) {
  return (
    <div className="field">
      <label htmlFor="captcha">Type the characters shown</label>
      <div className="captcha">
        {captcha ? (
          <img
            src={`data:image/svg+xml,${encodeURIComponent(captcha.svg)}`}
            alt="captcha challenge"
            height={44}
          />
        ) : (
          <span className="help">challenge unavailable</span>
        )}
        <button type="button" className="btn outline" onClick={refresh}>
          New image
        </button>
      </div>
      <input
        id="captcha"
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
        autoComplete="off"
        required
      />
      <p className="help">
        This stops casual bots and nothing more &mdash; the characters are text in the image and a
        script can read them.
      </p>
    </div>
  );
}

export function SignIn() {
  const { signIn, expired } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      /* session: false -- a 401 here is a refused credential, not a dead session, and must not
         trigger the app's sign-out path. */
      const { body } = await call<Session>("/auth/login", {
        method: "POST",
        body: { email, password },
        session: false,
      });
      signIn(body);
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell title="Sign in">
      {expired ? (
        <div className="status" style={{ marginBottom: 14 }}>
          Your session ended &mdash; it expired, or your password was changed. Sign in again.
        </div>
      ) : null}
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="username"
            required
          />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
        </div>
        <ErrorBox error={error} />
        <div className="btns">
          <button className="btn primary" disabled={busy} type="submit">
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <Link className="btn secondary" to="/register">
            Create an account
          </Link>
          <Link className="btn outline" to="/recover">
            Use a recovery code
          </Link>
        </div>
      </form>
    </Shell>
  );
}

export function Register() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const captcha = useCaptcha();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    if (!captcha.captcha) {
      setError(new Error("No captcha challenge was loaded. Press “New image” and try again."));
      return;
    }
    setBusy(true);
    try {
      const { body } = await call<Registration>("/auth/register", {
        method: "POST",
        body: {
          email,
          password,
          captcha_nonce: captcha.captcha.nonce,
          captcha_answer: captcha.answer,
        },
        session: false,
      });
      /* Sign in FIRST, then navigate. Drawing the recovery screen from this component instead
         would show it before the session exists, and a reload would land on the sign-in page
         with the code lost. */
      signIn(body);
      navigate("/register/recovery-code", {
        replace: true,
        state: { recoveryCode: body.recovery_code },
      });
    } catch (problem) {
      setError(problem);
      /* Any failure spends the nonce, right or wrong: a fresh challenge is required. */
      captcha.refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell title="Create an account">
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="username"
            required
          />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="new-password"
            required
            minLength={12}
          />
          <p className="help">At least 12 characters. A passphrase is fine and easier to keep.</p>
        </div>
        <CaptchaField {...captcha} />
        <ErrorBox error={error} />
        <div className="status" style={{ marginBottom: 12 }}>
          Each run you submit starts a GPU job that costs real money to run. Please only submit
          runs you mean to.
        </div>
        <div className="btns">
          <button className="btn primary" disabled={busy} type="submit">
            {busy ? "Creating…" : "Create account"}
          </button>
          <Link className="btn outline" to="/">
            Back to sign in
          </Link>
        </div>
      </form>
    </Shell>
  );
}

/** Shown once, after registering or recovering. Continue is disabled until the code has been
 *  copied or downloaded, and leaving warns, because there is no second chance to see it. */
export function RecoveryCodeShown() {
  const navigate = useNavigate();
  const state = (window.history.state?.usr ?? {}) as { recoveryCode?: string };
  const code = state.recoveryCode;
  const [kept, setKept] = useState(false);

  useEffect(() => {
    if (kept || !code) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [kept, code]);

  if (!code) {
    return (
      <Shell title="Recovery code">
        <p>
          There is no code to show. A recovery code is displayed once, at the moment it is issued.
          If you have lost yours, change your password from the{" "}
          <Link to="/account">account page</Link> &mdash; that issues a new one.
        </p>
      </Shell>
    );
  }

  const download = () => {
    const blob = new Blob(
      [
        `Wellmatix Peptide Discovery recovery code\n\n${code}\n\n` +
          `Issued ${new Date().toISOString()}.\nSingle use. A new code is issued when this one is used.\n`,
      ],
      { type: "text/plain" },
    );
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "wellmatix-peptide-recovery-code.txt";
    link.click();
    URL.revokeObjectURL(link.href);
    setKept(true);
  };

  return (
    <Shell title="Save your recovery code">
      <p>
        This is the only way back into your account if you forget your password. It is shown{" "}
        <b>once</b> and cannot be retrieved later.
      </p>
      <div className="code">{code}</div>
      <p className="help">
        Single use. Using it signs you in, sets a new password and issues a replacement code.
      </p>
      <div className="btns">
        <button
          className="btn secondary"
          type="button"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(code);
            } catch {
              /* clipboard blocked: the download is still offered, and the code is on screen */
            }
            setKept(true);
          }}
        >
          Copy
        </button>
        <button className="btn secondary" type="button" onClick={download}>
          Download
        </button>
        <button
          className="btn primary"
          type="button"
          disabled={!kept}
          title={kept ? undefined : "Copy or download the code first"}
          onClick={() => navigate("/", { replace: true })}
        >
          Continue
        </button>
      </div>
      {kept ? null : <p className="help">Copy or download the code to continue.</p>}
    </Shell>
  );
}

export function Recover() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const { body } = await call<Registration>("/auth/recover", {
        method: "POST",
        body: { email, recovery_code: code, new_password: password },
        session: false,
      });
      signIn(body);
      navigate("/register/recovery-code", {
        replace: true,
        state: { recoveryCode: body.recovery_code },
      });
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell title="Use a recovery code">
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="username"
            required
          />
        </div>
        <div className="field">
          <label htmlFor="code">Recovery code</label>
          <input
            id="code"
            value={code}
            onChange={(event) => setCode(event.target.value)}
            placeholder="XXXX-XXXX-XXXX-XXXX"
            autoComplete="off"
            required
          />
        </div>
        <div className="field">
          <label htmlFor="new-password">New password</label>
          <input
            id="new-password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="new-password"
            required
            minLength={12}
          />
        </div>
        <ErrorBox error={error} />
        <p className="help">
          Using the code signs you in and issues a replacement code, which is shown once on the
          next screen.
        </p>
        <div className="btns">
          <button className="btn primary" disabled={busy} type="submit">
            {busy ? "Checking…" : "Sign in with this code"}
          </button>
          <Link className="btn outline" to="/">
            Back to sign in
          </Link>
        </div>
      </form>
    </Shell>
  );
}

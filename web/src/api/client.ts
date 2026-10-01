/* The one place the app talks HTTP. Every path is same-origin (/auth/*, /api/peptide/*), served
   by nginx in production and by Vite's dev proxy in development, so there is no base URL and no
   CORS anywhere.

   A 401 on a SIGNED-IN call means the session is dead (expired, or revoked by a password change),
   and the app signs out. A 401 from login/recover is just a refused credential and must not. That
   is why change-password answers 403 for a wrong current password: a 401 there would sign the user
   out in the middle of what they were doing. */

const SESSION_KEY = "wmx.peptide.session";

export type Stored = { token: string; user: { id: number; email: string; created_at: string } };

export function loadSession(): Stored | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as Stored) : null;
  } catch {
    return null;
  }
}

export function saveSession(session: Stored | null): void {
  try {
    if (session) localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    /* storage unavailable (private mode): the session lives only in memory for this tab */
  }
}

let sessionExpired: () => void = () => {};
export function onSessionExpired(handler: () => void): void {
  sessionExpired = handler;
}

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

/* The dashboard's reading of an error body, kept: FastAPI puts a string, an object with its own
   `detail`, or a validation list under `detail`. */
export function describe(status: number, body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "detail" in detail && typeof (detail as { detail: unknown }).detail === "string") {
    return (detail as { detail: string }).detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((one) => {
        const item = one as { loc?: unknown[]; msg?: string };
        const where = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
        return where ? `${where}: ${item.msg}` : String(item.msg);
      })
      .join("\n");
  }
  if (detail) return JSON.stringify(detail, null, 2);
  return fallback || `HTTP ${status}`;
}

type Options = {
  method?: string;
  body?: unknown;
  /* Statuses that are answers rather than failures, returned instead of thrown. */
  accept?: number[];
  /* False for the public auth routes, whose 401 is a refusal and not a dead session. */
  session?: boolean;
  signal?: AbortSignal;
};

export async function call<T>(path: string, options: Options = {}): Promise<{ status: number; body: T }> {
  const { method = "GET", body, accept = [], session = true, signal } = options;
  const headers: Record<string, string> = { Accept: "application/json" };
  const stored = session ? loadSession() : null;
  if (stored) headers.Authorization = `Bearer ${stored.token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  } catch (problem) {
    throw new ApiError(0, `The service could not be reached (${(problem as Error).message}).`, null);
  }
  let parsed: unknown = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null;
  }
  if (response.ok || accept.includes(response.status)) {
    return { status: response.status, body: parsed as T };
  }
  /* Only the ACCOUNTS service's own 401 means the session is dead. A 401 under /api/ was produced
     by the upstream and passed through verbatim by the proxy, so it says nothing about this
     session -- signing out on it would throw the user out of the app over someone else's refusal.
     This is not hypothetical: with PEPTIDE_UPSTREAM pointing at the wrong service (port 8080 is a
     common default, and another app was listening on it), the upstream answered 401 and every
     signed-in user was bounced to the sign-in page reading "your session ended". */
  if (response.status === 401 && session && stored && !path.startsWith("/api/")) {
    sessionExpired();
  }
  throw new ApiError(
    response.status,
    path.startsWith("/api/") && (response.status === 401 || response.status === 403)
      ? `${describe(response.status, parsed, response.statusText)}\n\n` +
          "The job API answered " +
          response.status +
          ", but it has no authentication of its own — so this almost always means the proxy is" +
          " pointed at the wrong service. Check PEPTIDE_UPSTREAM."
      : describe(response.status, parsed, response.statusText),
    parsed,
  );
}

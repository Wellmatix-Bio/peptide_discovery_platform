import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { call, loadSession, onSessionExpired, saveSession, type Stored } from "../api/client";
import type { Session, User } from "../api/types";
import { lastRun } from "../workspace";

/* The session lives in localStorage so a new tab is signed in too. It is a bearer token: anything
   that can run script on this origin can read it, which is why the app renders every server string
   as text and never as HTML. */

type Auth = {
  user: User | null;
  ready: boolean;
  expired: boolean;
  signIn: (session: Session) => void;
  signOut: () => Promise<string | null>;
};

const AuthContext = createContext<Auth | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useState<Stored | null>(() => loadSession());
  const [ready, setReady] = useState(false);
  const [expired, setExpired] = useState(false);

  const signIn = useCallback((session: Session) => {
    /* A different account may be signing in to this tab; the remembered run belongs to whoever
       was here before. See workspace.ts. */
    lastRun.clear();
    const next = { token: session.token, user: session.user };
    saveSession(next);
    setStored(next);
    setExpired(false);
  }, []);

  /* Client-side, and honest about it: the server answers {"logged_out": false} and says why. */
  const signOut = useCallback(async () => {
    let why: string | null = null;
    try {
      const { body } = await call<{ logged_out: boolean; why: string }>("/auth/logout", { method: "POST", session: false });
      why = body.why;
    } catch {
      why = null;
    }
    lastRun.clear();
    saveSession(null);
    setStored(null);
    return why;
  }, []);

  useEffect(() => {
    onSessionExpired(() => {
      lastRun.clear();
      saveSession(null);
      setStored(null);
      setExpired(true);
    });
  }, []);

  /* Confirm a remembered session is still alive before trusting it. */
  useEffect(() => {
    if (!stored) {
      setReady(true);
      return;
    }
    call<User>("/auth/me")
      .then(({ body }) => setStored((current) => (current ? { ...current, user: body } : current)))
      .catch(() => undefined)
      .finally(() => setReady(true));
    // Only on first load: later changes to `stored` come from signIn/signOut, which are authoritative.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const value = useMemo<Auth>(
    () => ({ user: stored?.user ?? null, ready, expired, signIn, signOut }),
    [stored, ready, expired, signIn, signOut],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): Auth {
  const auth = useContext(AuthContext);
  if (!auth) throw new Error("useAuth outside AuthProvider");
  return auth;
}

/* The status rules in call(), which decide whether a user stays signed in.
 *
 * The 401 split is the one that bit in practice. The client used to sign out on ANY 401, so when
 * PEPTIDE_UPSTREAM pointed at the wrong service (something else was listening on the job API's
 * default port 8080) that service's 401 was passed through by the proxy and every signed-in user
 * was thrown to the sign-in page reading "your session ended". An upstream's refusal says nothing
 * about this session.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, call, onSessionExpired, saveSession } from "../api/client";

const SESSION = { token: "t0ken", user: { id: 1, email: "a@b.c", created_at: "2026-01-01" } };

function respondWith(status: number, body: unknown = { detail: "no" }) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })),
  );
}

let expired: number;

beforeEach(() => {
  expired = 0;
  onSessionExpired(() => {
    expired += 1;
  });
  saveSession(SESSION);
});

afterEach(() => {
  saveSession(null);
  vi.unstubAllGlobals();
  onSessionExpired(() => {});
});

describe("a 401 from the accounts service", () => {
  it("signs the user out", async () => {
    respondWith(401);
    await expect(call("/auth/history")).rejects.toBeInstanceOf(ApiError);
    expect(expired).toBe(1);
  });
});

describe("a 401 passed through from the job API", () => {
  it("does NOT sign the user out", async () => {
    respondWith(401);
    await expect(call("/api/peptide/api/v1/models")).rejects.toBeInstanceOf(ApiError);
    expect(expired).toBe(0);
  });

  it("explains that the upstream has no authentication, so the proxy is misconfigured", async () => {
    respondWith(401);
    await call("/api/peptide/api/v1/models").catch((problem: ApiError) => {
      expect(problem.message).toContain("PEPTIDE_UPSTREAM");
    });
    expect.assertions(1);
  });

  it("says the same for a 403", async () => {
    respondWith(403);
    await call("/api/peptide/api/v1/jobs/create", { method: "POST", body: {} }).catch(
      (problem: ApiError) => {
        expect(problem.message).toContain("PEPTIDE_UPSTREAM");
      },
    );
    expect(expired).toBe(0);
  });
});

describe("statuses that are answers, not failures", () => {
  it("returns an accepted status instead of throwing", async () => {
    respondWith(404, { detail: "not written yet" });
    const { status } = await call("/api/peptide/api/v1/jobs/x/results", { accept: [404] });
    expect(status).toBe(404);
    expect(expired).toBe(0);
  });
});

describe("a 401 from a public auth route", () => {
  it("is a refused credential and does not sign out", async () => {
    respondWith(401);
    await expect(call("/auth/login", { method: "POST", body: {}, session: false })).rejects.toThrow();
    expect(expired).toBe(0);
  });
});

describe("403 and 429 anywhere", () => {
  it.each([403, 429])("%s never signs the user out", async (status) => {
    respondWith(status);
    await expect(call("/auth/change-password", { method: "POST", body: {} })).rejects.toThrow();
    expect(expired).toBe(0);
  });
});

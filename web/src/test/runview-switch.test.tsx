/* Switching directly from one run to another.
 *
 * Reported by review: a late response from the run you navigated AWAY from can land after the new
 * run's, and write its state over the page you are now looking at. RunView's poll() called
 * setStatus/setResults unconditionally; the effect's `live` flag was only consulted after poll()
 * returned, and only to decide whether to schedule the next tick. jobStatus() accepts an
 * AbortSignal, which was never passed.
 *
 * The same navigation also kept the previous run's state, because the route parameter changes
 * without remounting the component.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunView } from "../runs/RunView";
import { saveSession } from "../api/client";

const SLOW = "projects/p/locations/us-central1/customJobs/111";
const FAST = "projects/p/locations/us-central1/customJobs/222";

function statusFor(jobId: string, stage: string) {
  return { job_id: jobId, status: "running", stage, vertex_state: "JOB_STATE_RUNNING", error: null };
}

/** The first run's status resolves LATE, after the second run's has already arrived. */
function serveWithLateFirstResponse() {
  let seen = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const slow = url.includes("111");
      seen += 1;
      const body = slow ? statusFor(SLOW, "s04_candidate_generation") : statusFor(FAST, "s08_safety_developability");
      if (slow) await new Promise((resolve) => setTimeout(resolve, 120));
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
  return () => seen;
}

function show(jobId: string) {
  return render(
    <MemoryRouter initialEntries={[`/runs/${encodeURIComponent(jobId)}`]}>
      <Routes>
        <Route path="/runs/:jobId" element={<RunView />} />
      </Routes>
    </MemoryRouter>,
  );
}

/** Navigating BETWEEN runs: the route parameter changes and RunView is NOT remounted, so the
 *  cleanup's abort is not the thing protecting us -- the staleness check is. This is the case
 *  the review described, and the one the abort alone does not cover. */
function Switcher() {
  const navigate = useNavigate();
  return (
    <button type="button" onClick={() => navigate(`/runs/${encodeURIComponent(FAST)}`)}>
      go to the other run
    </button>
  );
}

function showSwitchable(jobId: string) {
  return render(
    <MemoryRouter initialEntries={[`/runs/${encodeURIComponent(jobId)}`]}>
      <Switcher />
      <Routes>
        <Route path="/runs/:jobId" element={<RunView />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => saveSession({ token: "t", user: { id: 1, email: "a@b.c", created_at: "2026-01-01" } }));
afterEach(() => {
  cleanup();
  saveSession(null);
  vi.unstubAllGlobals();
});

describe("switching from one run straight to another", () => {
  it("does not let the abandoned run's late response overwrite the new one", async () => {
    serveWithLateFirstResponse();
    const first = show(SLOW);
    // Navigate away before the slow response lands.
    first.unmount();
    show(FAST);

    await waitFor(() =>
      expect(screen.getAllByText(/safety developability/i).length).toBeGreaterThan(0),
    );
    // Give the abandoned request time to resolve and do damage if it can.
    await new Promise((resolve) => setTimeout(resolve, 250));

    // The abandoned run's stage must never appear on the page for the run now showing.
    expect(screen.queryByText(/candidate generation/i)).not.toBeInTheDocument();
    expect(screen.queryByText(SLOW)).not.toBeInTheDocument();
  });

  it("shows the run named in the URL, not the one before it", async () => {
    serveWithLateFirstResponse();
    show(FAST);
    await waitFor(() => expect(screen.getByText(FAST)).toBeInTheDocument());
    expect(screen.queryByText(SLOW)).not.toBeInTheDocument();
  });
});


describe("switching while the previous run's request is still in flight", () => {
  it("ignores the abandoned run's response even though the component was never unmounted", async () => {
    serveWithLateFirstResponse();
    showSwitchable(SLOW);
    // Switch before the slow response lands. The route parameter changes; RunView stays mounted,
    // so its cleanup abort is not what protects the page here.
    screen.getByRole("button", { name: /go to the other run/i }).click();

    await waitFor(() =>
      expect(screen.getAllByText(/safety developability/i).length).toBeGreaterThan(0),
    );
    await new Promise((resolve) => setTimeout(resolve, 250));

    expect(screen.queryByText(/candidate generation/i)).not.toBeInTheDocument();
    expect(screen.queryByText(SLOW)).not.toBeInTheDocument();
  });
});


describe("switching away from a run that has already loaded", () => {
  it("does not leave the previous run's data on screen under the new run's heading", async () => {
    /* The route parameter changes without remounting, so every piece of state survives unless it
       is cleared. Without the reset, run A's stage and id stay visible while run B loads --
       briefly showing one run's data labelled as another's. */
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("111")) {
          return new Response(JSON.stringify(statusFor(SLOW, "s04_candidate_generation")), {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        }
        // The second run never answers, so anything on screen can only be the first run's.
        await new Promise((resolve) => setTimeout(resolve, 5000));
        return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
      }),
    );

    showSwitchable(SLOW);
    await waitFor(() =>
      expect(screen.getAllByText(/candidate generation/i).length).toBeGreaterThan(0),
    );

    screen.getByRole("button", { name: /go to the other run/i }).click();
    await waitFor(() => expect(screen.getByText(FAST)).toBeInTheDocument());

    expect(screen.queryByText(/candidate generation/i)).not.toBeInTheDocument();
    expect(screen.queryByText(SLOW)).not.toBeInTheDocument();
  });
});

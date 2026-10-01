/* The run view, against responses CAPTURED FROM THE RUNNING API.
 *
 * src/test/fixtures/README.md says how they were produced and what capturing them revealed. The
 * four `derived/` files are built from a captured one by a stated transformation, for three cases
 * no real run has produced yet; they are kept separate so nobody mistakes one for evidence.
 *
 * What is actually being guarded here is §7: that the page does not overstate what the API said.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunView } from "../runs/RunView";
import { saveSession } from "../api/client";

import realResults from "./fixtures/results-9223232270029029376.json";
import realStatus from "./fixtures/status-9223232270029029376.json";
import smallResults from "./fixtures/results-156676979574177792.json";
import withheld from "./fixtures/derived/results-with-withheld-and-unranked.json";
import tied from "./fixtures/derived/results-with-a-tie.json";
import workerDied from "./fixtures/derived/status-worker-died.json";
import succeededNoResult from "./fixtures/derived/status-succeeded-no-result.json";
import stillRunning from "./fixtures/derived/status-running.json";

const JOB = realStatus.job_id;

function serve(status: unknown, results: unknown, resultsCode = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = url.includes("/results") ? results : status;
      const code = url.includes("/results") ? resultsCode : 200;
      return new Response(JSON.stringify(body), {
        status: code,
        headers: { "content-type": "application/json" },
      });
    }),
  );
}

function show() {
  return render(
    <MemoryRouter initialEntries={[`/runs/${encodeURIComponent(JOB)}`]}>
      <Routes>
        <Route path="/runs/:jobId" element={<RunView />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  saveSession({ token: "t", user: { id: 1, email: "a@b.c", created_at: "2026-01-01" } });
});

afterEach(() => {
  // Not automatic: this project's vitest config does not set `globals`, so Testing Library's
  // auto-cleanup never registers and renders would pile up across tests.
  cleanup();
  saveSession(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("a real succeeded run", () => {
  it("shows the API's own counts, not counts re-derived from the list", async () => {
    serve(realStatus, realResults);
    show();
    // The fixture has 33 candidates, all ranked, and the API says so itself.
    expect(realResults.ranked_candidates).toBe(33);
    await waitFor(() => expect(screen.getByText("Shortlist")).toBeInTheDocument());
    const shortlist = screen.getByText("Shortlist").closest(".card") as HTMLElement;
    // n_final and ranked_candidates are both 33 in this run, so each tile is read by its label
    // rather than by the value -- which also proves the value sits under the right label.
    const tile = (label: string) =>
      within(shortlist).getByText(label).closest(".metric")!.querySelector("b")!.textContent;
    expect(tile("Final candidates")).toBe(String(realResults.n_final));
    expect(tile("Ranked")).toBe(String(realResults.ranked_candidates));
    expect(tile("Insufficient evidence")).toBe(
      String(realResults.insufficient_evidence_candidates),
    );
  });

  it("renders the real peptide sequences the API returned", async () => {
    serve(realStatus, realResults);
    show();
    const first = realResults.candidates[0]!;
    await waitFor(() => expect(screen.getByText(first.sequence!)).toBeInTheDocument());
    expect(first.sequence).toMatch(/^[ACDEFGHIKLMNPQRSTVWY]+$/);
  });

  it("shows both status and vertex_state, and never collapses them into one word", async () => {
    serve(realStatus, realResults);
    show();
    await waitFor(() => expect(screen.getByText(/status \(from the worker\)/)).toBeInTheDocument());
    expect(screen.getByText(/vertex_state \(live from Vertex\)/)).toBeInTheDocument();
    expect(screen.getByText(realStatus.vertex_state)).toBeInTheDocument();
  });

  it("does not claim a model identity, because the API attaches none to a run", async () => {
    serve(realStatus, realResults);
    show();
    await waitFor(() => expect(screen.getByText("Provenance")).toBeInTheDocument());
    expect(screen.getByText(/returns no model identity with a run/i)).toBeInTheDocument();
  });

  it("works for a different real run too", async () => {
    serve(
      { ...realStatus, job_id: smallResults.job_id },
      smallResults,
    );
    show();
    await waitFor(() => expect(screen.getByText("Shortlist")).toBeInTheDocument());
    expect(screen.getByText(smallResults.candidates[0]!.sequence!)).toBeInTheDocument();
  });
});

describe("a run Vertex failed", () => {
  it("is shown as failed, with the API's own error text rather than a guess", async () => {
    serve(workerDied, null, 404);
    show();
    await waitFor(() => expect(screen.getByText(workerDied.error!)).toBeInTheDocument());
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.queryByText(/^running$/)).not.toBeInTheDocument();
  });

  it("stops offering cancel, since there is nothing left to cancel", async () => {
    serve(workerDied, null, 404);
    show();
    await waitFor(() => expect(screen.getByText("failed")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /cancel this run/i })).not.toBeInTheDocument();
  });
});

describe("a run Vertex SUCCEEDED but whose worker wrote no result", () => {
  /* The case the "abandoned" state exists for, and the most misleading thing this view could
     get wrong. The API reports status "pending" FOREVER here and nothing ever contradicts it
     (docs/BASELINE.md), so a page trusting `status` alone shows a dead run as still working.
     A Vertex FAILURE is the easier case -- Vertex says so and carries an error. */
  it("is not presented as still in progress", async () => {
    serve(succeededNoResult, null, 404);
    show();
    await waitFor(() =>
      expect(screen.getByText(/Vertex AI reports this job has finished/i)).toBeInTheDocument(),
    );
    expect(screen.getByText("no result written")).toBeInTheDocument();
    expect(screen.queryByText(/^running$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^submitted$/)).not.toBeInTheDocument();
  });

  it("still shows the raw pending status, so the API is not misquoted either", async () => {
    serve(succeededNoResult, null, 404);
    show();
    await waitFor(() => expect(screen.getByText("no result written")).toBeInTheDocument());
    // status and stage are BOTH "pending" here, so each is read by its own label.
    const valueFor = (label: RegExp) =>
      screen.getByText(label).nextElementSibling!.textContent;
    expect(valueFor(/status \(from the worker\)/)).toBe("pending");
    expect(valueFor(/vertex_state \(live from Vertex\)/)).toBe("JOB_STATE_SUCCEEDED");
  });

  it("does not offer cancel", async () => {
    serve(succeededNoResult, null, 404);
    show();
    await waitFor(() => expect(screen.getByText("no result written")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /cancel this run/i })).not.toBeInTheDocument();
  });
});

describe("a running run", () => {
  it("names the stage the worker last recorded, and offers cancel", async () => {
    serve(stillRunning, null, 404);
    show();
    // The stage name appears in both the page title and the detail list; one is enough.
    await waitFor(() =>
      expect(screen.getAllByText(/functional models/i).length).toBeGreaterThan(0),
    );
    expect(screen.getByRole("button", { name: /cancel this run/i })).toBeInTheDocument();
    expect(screen.getByText(/safe to close this tab/i)).toBeInTheDocument();
  });
});

describe("withheld values", () => {
  it('renders "No value", never a dash and never zero', async () => {
    serve(realStatus, withheld);
    show();
    await waitFor(() => expect(screen.getByText("Shortlist")).toBeInTheDocument());
    // candidate[0] has its sequence withheld; a dash or a 0 would both read as a real answer.
    const noValues = screen.getAllByText("No value");
    expect(noValues.length).toBeGreaterThan(0);
    expect(screen.queryByText("—")).not.toBeInTheDocument();
  });

  it("lists an unranked candidate without giving it a position", async () => {
    serve(realStatus, withheld);
    show();
    await waitFor(() =>
      expect(screen.getByText(/Returned without a rank/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Deliberately unordered/i)).toBeInTheDocument();
    expect(screen.getByText("unranked")).toBeInTheDocument();
  });
});

describe("ties", () => {
  it("shows a shared position as a tie rather than inventing an order", async () => {
    serve(realStatus, tied);
    show();
    await waitFor(() => expect(screen.getByText("Shortlist")).toBeInTheDocument());
    expect(screen.getAllByText(/tied/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/2 candidates share this position/i).length).toBeGreaterThan(0);
  });
});

describe("the fixtures themselves", () => {
  it("are captured responses, and the derived ones say so", () => {
    expect(realResults).not.toHaveProperty("_derived_from");
    expect(withheld).toHaveProperty("_derived_from");
    expect(workerDied._derived_from.changed).toMatch(/pending/);
  });

  it("name no real infrastructure", () => {
    const text = JSON.stringify([realResults, realStatus, smallResults]);
    expect(text).toContain("example-project");
    expect(text).not.toMatch(/iam\.gserviceaccount|docker\.pkg\.dev/);
  });
});

/* The run view against the OTHER TWO pipelines' response shapes.
 *
 * These envelopes are SYNTHESIZED FROM web/openapi/peptide.json, not captured from a running API,
 * and that is a real difference from the fixtures in src/test/fixtures/ -- nothing here is
 * evidence of what a metabolic or cosmetic run actually returns, only of what the API says it
 * will. They are built inline rather than written to fixtures/ so no one can mistake them for
 * captures. The snapshot they are drawn from is CI-enforced against api.py, so a field renamed
 * server-side breaks the generated types these objects are typed against.
 *
 * What is guarded here is the defect these tests were written for: GET /results returns one of
 * three shapes, the app read every one of them as the wound-healing shape, and a metabolic run
 * therefore rendered a page of "No value" with its real numbers unread in the payload. No
 * compile error, no runtime error, nothing in the console -- so only a test that renders a
 * metabolic envelope and looks for a metabolic number can catch its return.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RunView } from "../runs/RunView";
import { saveSession } from "../api/client";
import type {
  CosmeticJobResultsResponse,
  JobResultsResponse,
  MetabolicJobResultsResponse,
} from "../api/types";

import realStatus from "./fixtures/status-9223232270029029376.json";

const JOB = realStatus.job_id;

const UNRELIABLE_NOTE =
  "GIPR potency is reported but NOT usable for ranking: leave-one-cluster-out R2 is -0.108." +
  " The values still count toward the gates and the ranking score. Treat them as unscreened," +
  " not as evidence.";

const metabolic: MetabolicJobResultsResponse = {
  pipeline: "metabolic",
  job_id: JOB,
  run_id: "456",
  status: "success",
  n_final: 1,
  ranked_candidates: 1,
  insufficient_evidence_candidates: 0,
  component_stats: {},
  candidates: [
    {
      id: "cand-m-1",
      sequence: "HSQGTFTSDYSKYLD",
      final_score_rank: 1,
      final_score: 0.82,
      evidence_coverage: 1,
      missing_modules: [],
      pec50: { GLP1R: 9.4, GCGR: 8.1, GIPR: 7.25 },
      ec50_nm: { GLP1R: 0.4 },
      binding_pkd: { GLP1R: 8.9 },
      agonism_probability: { GLP1R: 0.91 },
      potency_unreliable_receptors: ["GIPR"],
      potency_unreliable_note: UNRELIABLE_NOTE,
      metabolic_activity_score: 0.77,
      half_life_minutes: 143.5,
      molecular_weight: 1800.2,
      net_charge: -1.1,
    },
  ],
};

const cosmetic: CosmeticJobResultsResponse = {
  pipeline: "cosmetic",
  job_id: JOB,
  run_id: "789",
  status: "success",
  n_final: 1,
  ranked_candidates: 1,
  insufficient_evidence_candidates: 0,
  component_stats: {},
  candidates: [
    {
      id: "cand-c-1",
      sequence: "KTTKS",
      final_score_rank: 1,
      final_score: 0.61,
      primary_function: "collagen_synthesis",
      component_scores: {
        collagen_synthesis: { score: 0.73, screened: true },
        antioxidant: { score: null, screened: false },
      },
      cytotoxicity: {},
      functional_scores: { collagen_synthesis: {} },
      not_screened: ["antioxidant"],
      physchem_flags: [],
      safety_flags: [],
      synthesis_flags: {},
      irritation_tier: "low",
      skin_sensitizer: false,
      skin_sensitization_probability: 0.12,
    },
  ],
};

/** The wound-healing envelope WITHOUT a `pipeline` field, which is what a deployed API that
 *  predates the tag returns -- every captured fixture in this repo is one. */
const untagged = {
  job_id: JOB,
  run_id: "123",
  status: "success",
  n_final: 1,
  ranked_candidates: 1,
  insufficient_evidence_candidates: 0,
  component_stats: {},
  candidates: [
    {
      id: "cand-w-1",
      sequence: "GLFDIVKKVVGALGSL",
      final_score_rank: 1,
      amp_probability: 0.88,
      log_mic_um: {},
      pmbic: {},
    },
  ],
} as unknown as JobResultsResponse;

function serve(results: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) =>
      new Response(JSON.stringify(url.includes("/results") ? results : realStatus), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    ),
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

/** The per-candidate body is behind a "More" button, as it is for wound healing. */
async function expand(sequence: string) {
  await waitFor(() => expect(screen.getByText(sequence)).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: "More" }));
}

beforeEach(() => {
  saveSession({ token: "t", user: { id: 1, email: "a@b.c", created_at: "2026-01-01" } });
});

afterEach(() => {
  cleanup();
  saveSession(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("a metabolic run", () => {
  it("renders the receptor potency the API returned, instead of a page of No value", async () => {
    serve(metabolic);
    show();
    await expand("HSQGTFTSDYSKYLD");
    expect(screen.getByText("Receptor potency")).toBeInTheDocument();
    expect(screen.getByText("pEC50 9.40")).toBeInTheDocument();
    expect(screen.getByText("0.40 nM")).toBeInTheDocument();
    expect(screen.getByText("143.5")).toBeInTheDocument();
  });

  it("shows every receptor the API mentioned, not only the ones in one of its maps", async () => {
    serve(metabolic);
    show();
    await expand("HSQGTFTSDYSKYLD");
    // GCGR appears in pec50 alone; a table built from ec50_nm or binding_pkd would drop it.
    // getAllByText, not getByText: GIPR is deliberately named twice, in its row and in the
    // header flag, and a single-match query would fail for the receptor that matters most.
    for (const receptor of ["GLP1R", "GCGR", "GIPR"]) {
      expect(screen.getAllByText(receptor, { exact: false }).length).toBeGreaterThan(0);
    }
  });

  it("flags GIPR as not predictive WITHOUT hiding its value", async () => {
    serve(metabolic);
    show();
    await waitFor(() =>
      expect(screen.getByText("GIPR potency not predictive")).toBeInTheDocument(),
    );
    // Flagged in the header, before anything is expanded: a reader who never opens the panel
    // still sees it. And the number is still reported -- the contract is "flagged", not "withheld".
    await expand("HSQGTFTSDYSKYLD");
    expect(screen.getByText("pEC50 7.25")).toBeInTheDocument();
    expect(screen.getByText(UNRELIABLE_NOTE)).toBeInTheDocument();
  });

  it("does not render the wound-healing panels, whose fields a metabolic run has none of", async () => {
    serve(metabolic);
    show();
    await expand("HSQGTFTSDYSKYLD");
    expect(screen.queryByText("Biofilm pMBIC by pathogen")).not.toBeInTheDocument();
    expect(screen.queryByText("MIC by pathogen (log10 µM)")).not.toBeInTheDocument();
    expect(screen.queryByText("Antimicrobial probability")).not.toBeInTheDocument();
  });
});

describe("a cosmetic run", () => {
  it("renders its own components and says which screens did not run", async () => {
    serve(cosmetic);
    show();
    await waitFor(() => expect(screen.getByText("KTTKS")).toBeInTheDocument());
    // "not screened" is the API saying the screen DID NOT RUN. It is a flag in the header, not a
    // zero folded into the scores, because a screen that did not run is not one that passed.
    expect(screen.getByText("antioxidant not screened")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "More" }));
    expect(screen.getByText("Skin safety")).toBeInTheDocument();
    expect(screen.getByText("0.730")).toBeInTheDocument();
  });
});

describe("a response with no pipeline field", () => {
  it("is read as wound healing, which is what the API's own default says it is", async () => {
    serve(untagged);
    show();
    await expand("GLFDIVKKVVGALGSL");
    expect(screen.getByText("MIC by pathogen (log10 µM)")).toBeInTheDocument();
    expect(screen.queryByText("Receptor potency")).not.toBeInTheDocument();
  });
});

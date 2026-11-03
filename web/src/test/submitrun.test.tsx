/* The submit form, for both pipelines it can submit.
 *
 * What is guarded here is the thing that cannot be caught by typechecking: WHICH BYTES GO TO THE
 * API. The two pipelines take different stage names -- a wound-healing brief sent under
 * `pipeline: "metabolic"` is a 422 -- so these tests read the actual request body out of the
 * stubbed fetch rather than asserting on what the form looks like.
 *
 * The form had no test at all before this. It is the one screen in the app that spends money.
 */
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SubmitRun } from "../runs/SubmitRun";
import { saveSession } from "../api/client";
import { METABOLIC_EXAMPLES } from "../runs/examples";

/** Every POST the form made, as parsed bodies. */
let sent: Record<string, unknown>[] = [];

function stubApi() {
  sent = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === "POST") sent.push(JSON.parse(String(init.body)));
      return new Response(
        JSON.stringify({ job_id: "projects/p/locations/l/customJobs/1", result_path: "gs://x" }),
        { status: 202, headers: { "content-type": "application/json" } },
      );
    }),
  );
}

function show() {
  return render(
    <MemoryRouter>
      <SubmitRun />
    </MemoryRouter>,
  );
}

const pickPipeline = (label: "Wound healing" | "Metabolic" | "Cosmetic") =>
  fireEvent.click(screen.getByRole("button", { name: label }));

const submit = () =>
  fireEvent.click(screen.getByRole("button", { name: /Submit this run/ }));

/** The receptor pills are labels wrapping a hidden checkbox, so they are clicked by their text. */
const chooseReceptor = (receptor: string) => {
  const chooser = screen.getByText("Target receptors").closest(".field") as HTMLElement;
  fireEvent.click(within(chooser).getByText(receptor));
};

beforeEach(() => {
  saveSession({ token: "t", user: { id: 1, email: "a@b.c", created_at: "2026-01-01" } });
  stubApi();
});

afterEach(() => {
  cleanup();
  saveSession(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("the pipeline picker", () => {
  it("starts on wound healing, so the existing behaviour is the default", () => {
    show();
    expect(screen.getByText("The wound")).toBeInTheDocument();
    expect(screen.queryByText("Target receptors")).not.toBeInTheDocument();
  });

  it("swaps in the metabolic brief, whose fields are not the wound-healing ones", () => {
    show();
    pickPipeline("Metabolic");
    expect(screen.getByText("Target receptors")).toBeInTheDocument();
    // A metabolic brief has none of these, and offering them would invite a 422.
    expect(screen.queryByText("Wound context")).not.toBeInTheDocument();
    expect(screen.queryByText("Target pathogens")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Dosing interval/)).not.toBeInTheDocument();
  });

  it("keeps each pipeline's answers when switching away and back", () => {
    show();
    pickPipeline("Metabolic");
    chooseReceptor("GLP1R");
    pickPipeline("Wound healing");
    pickPipeline("Metabolic");
    const chooser = screen.getByText(/Target receptors/).closest(".field") as HTMLElement;
    expect(chooser.querySelector(".choice.on")).toBeTruthy();
  });
});

describe("submitting a metabolic run", () => {
  it("refuses to send one with no receptor, because the API requires at least one", async () => {
    show();
    pickPipeline("Metabolic");
    submit();
    await waitFor(() =>
      expect(screen.getByText(/at least one target receptor/i)).toBeInTheDocument(),
    );
    expect(sent).toHaveLength(0);
  });

  it("sends the metabolic tag and the metabolic stage names, not the wound-healing ones", async () => {
    show();
    pickPipeline("Metabolic");
    chooseReceptor("GLP1R");
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    const body = sent[0] as {
      pipeline: string;
      stages: Record<string, { brief?: { target_receptors?: string[] } }>;
    };
    expect(body.pipeline).toBe("metabolic");
    expect(body.stages.s01_metabolic_therapeutic_product_brief!.brief!.target_receptors).toEqual([
      "GLP1R",
    ]);
    expect(Object.keys(body.stages)).toContain("s06_metabolic_functional_screening");
    // The wound-healing names are not merely unused; under this tag the API rejects them.
    expect(Object.keys(body.stages)).not.toContain("s01_therapeutic_product_brief");
    expect(Object.keys(body.stages)).not.toContain("s06_functional_models");
  });

  it("warns that GIPR potency is not predictive, and still allows the run", async () => {
    show();
    pickPipeline("Metabolic");
    chooseReceptor("GIPR");
    // Before any submission: the warning is for a decision, not a post-mortem.
    expect(screen.getByText(/GIPR potency is not predictive/)).toBeInTheDocument();
    expect(screen.getByText(/still count toward the gates/)).toBeInTheDocument();
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    expect((sent[0] as { pipeline: string }).pipeline).toBe("metabolic");
  });

  it("says nothing about GIPR when GIPR was not chosen", () => {
    show();
    pickPipeline("Metabolic");
    chooseReceptor("GLP1R");
    expect(screen.queryByText(/not predictive/)).not.toBeInTheDocument();
  });
});

describe("submitting a cosmetic run", () => {
  it("sends the cosmetic tag, its nested brief and its own stage names", async () => {
    show();
    pickPipeline("Cosmetic");
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    const body = sent[0] as {
      pipeline: string;
      stages: Record<string, { brief?: Record<string, unknown> }>;
    };
    expect(body.pipeline).toBe("cosmetic");
    const brief = body.stages.s01_cosmetic_therapeutic_product_brief!.brief as {
      application: { area: string };
      functions: { primary: string };
      peptide: { min_length: number };
    };
    // NESTED, not flat, and spelled out rather than left to the server's defaults.
    expect(brief.application.area).toBe("face");
    expect(brief.functions.primary).toBe("collagen_ecm");
    expect(brief.peptide.min_length).toBe(6);
    expect(Object.keys(body.stages)).toContain("s08_cosmetic_safety_developability");
    expect(Object.keys(body.stages)).not.toContain("s08_safety_developability");
  });

  it("submits with no target depth, because null is a request and not a blank", async () => {
    show();
    pickPipeline("Cosmetic");
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    const brief = (sent[0] as { stages: Record<string, { brief: { target_depth: unknown } }> })
      .stages.s01_cosmetic_therapeutic_product_brief!.brief;
    expect(brief.target_depth).toBeNull();
  });

  it("refuses more secondary functions than the API accepts", async () => {
    show();
    pickPipeline("Cosmetic");
    const field = screen.getByText(/Secondary functions/).closest(".field") as HTMLElement;
    for (const value of ["collagen ecm", "antioxidant", "pigmentation"]) {
      fireEvent.click(within(field).getByText(value));
    }
    submit();
    await waitFor(() =>
      expect(screen.getByText(/at most 2 secondary functions/i)).toBeInTheDocument(),
    );
    expect(sent).toHaveLength(0);
  });

  it("does not offer a candidate count, which this pipeline has no field for", () => {
    show();
    pickPipeline("Cosmetic");
    expect(screen.queryByLabelText(/Candidates to generate/)).not.toBeInTheDocument();
    expect(screen.getByText(/genetic algorithm sized by population/)).toBeInTheDocument();
  });
});

describe("submitting a wound-healing run still works", () => {
  it("sends the wound-healing tag and its own stage names", async () => {
    show();
    for (const [legend, value] of [
      ["Wound context", "acute"],
      ["Desired functions", "antimicrobial"],
      ["Target pathogens", "Escherichia coli"],
    ] as const) {
      const field = screen.getByText(legend).closest(".field") as HTMLElement;
      fireEvent.click(within(field).getByText(value));
    }
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    const body = sent[0] as { pipeline: string; stages: Record<string, unknown> };
    expect(body.pipeline).toBe("wound_healing");
    expect(Object.keys(body.stages)).toContain("s01_therapeutic_product_brief");
  });
});

describe("the JSON tab takes its pipeline from the text, not from the picker", () => {
  const metabolicJson = JSON.stringify({
    request_id: "pasted",
    pipeline: "metabolic",
    stages: {
      s01_metabolic_therapeutic_product_brief: {
        brief: METABOLIC_EXAMPLES[0]!.brief,
      },
    },
  });

  function paste(text: string) {
    fireEvent.click(screen.getByRole("button", { name: "Advanced: JSON" }));
    fireEvent.change(screen.getByLabelText("request JSON"), { target: { value: text } });
  }

  it("accepts a pasted metabolic request while the picker shows wound healing", async () => {
    show();
    paste(metabolicJson);
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    expect((sent[0] as { pipeline: string }).pipeline).toBe("metabolic");
  });

  it("checks the pasted metabolic brief rather than waving it through", async () => {
    show();
    paste(
      JSON.stringify({
        pipeline: "metabolic",
        stages: { s01_metabolic_therapeutic_product_brief: { brief: { target_receptors: [], min_length: 20, max_length: 40 } } },
      }),
    );
    submit();
    await waitFor(() =>
      expect(screen.getByText(/at least one target receptor/i)).toBeInTheDocument(),
    );
    expect(sent).toHaveLength(0);
  });

  it("accepts a pasted cosmetic request, whose brief may legitimately be absent", async () => {
    show();
    // Unlike the other two, every cosmetic field has a server-side default, so `{}` is a real
    // request and an absent brief is not an error.
    paste(JSON.stringify({ pipeline: "cosmetic", stages: {} }));
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    expect((sent[0] as { pipeline: string }).pipeline).toBe("cosmetic");
  });

  it("refuses a pipeline it has no checks for instead of starting a billed job", async () => {
    show();
    paste(JSON.stringify({ pipeline: "dermal", stages: {} }));
    submit();
    await waitFor(() => expect(screen.getByText(/cannot submit a "dermal"/)).toBeInTheDocument());
    expect(sent).toHaveLength(0);
  });

  it("reads an untagged request as wound healing, the API's own default", async () => {
    show();
    paste(
      JSON.stringify({
        stages: {
          s01_therapeutic_product_brief: {
            brief: {
              wound_context: ["acute"],
              desired_functions: ["antimicrobial"],
              pathogens: ["Escherichia_coli"],
              min_length: 6,
              max_length: 20,
              dosing_interval_hours: 24,
            },
          },
        },
      }),
    );
    submit();
    await waitFor(() => expect(sent).toHaveLength(1));
    expect((sent[0] as { pipeline: string }).pipeline).toBe("wound_healing");
  });
});

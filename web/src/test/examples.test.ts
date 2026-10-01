/* The examples the form offers must be valid requests.
 *
 * Checked against web/openapi/peptide.json -- the snapshot taken from api.py itself -- not
 * against a list repeated here. The whole reason this file exists is that the repository's own
 * 32 example briefs are all invalid (docs/BRIEF_VALIDITY.md); shipping examples that 422 would be
 * the same mistake with fewer files.
 */
import { describe, expect, it } from "vitest";
import snapshot from "../../openapi/peptide.json";
import { EMPTY_BRIEF, EXAMPLES } from "../runs/examples";

type Schema = {
  components: {
    schemas: {
      BriefFields: {
        required: string[];
        properties: Record<
          string,
          { items?: { enum?: string[] }; enum?: string[]; minimum?: number; maximum?: number }
        >;
      };
    };
  };
};

const brief = (snapshot as unknown as Schema).components.schemas.BriefFields;
const enumOf = (field: string): string[] => {
  const property = brief.properties[field];
  const values = property?.items?.enum ?? property?.enum;
  if (!values) throw new Error(`${field} has no enum in the snapshot`);
  return values;
};

describe("the snapshot is the source of truth", () => {
  it("still describes BriefFields with the fields the form builds", () => {
    expect(brief.required.sort()).toEqual(
      [
        "desired_functions",
        "dosing_interval_hours",
        "max_length",
        "min_length",
        "pathogens",
        "wound_context",
      ].sort(),
    );
  });

  it("has at least one example to check, so this file cannot pass vacuously", () => {
    expect(EXAMPLES.length).toBeGreaterThanOrEqual(3);
  });
});

describe.each(EXAMPLES.map((example) => [example.id, example] as const))(
  "example %s",
  (_id, example) => {
    it.each(["wound_context", "desired_functions", "pathogens"])(
      "uses only %s values the API accepts",
      (field) => {
        const allowed = enumOf(field);
        const used = example.brief[field as "wound_context"] as string[];
        expect(used.length).toBeGreaterThan(0);
        for (const value of used) expect(allowed).toContain(value);
      },
    );

    it("keeps the peptide length inside the API's bound, min <= max", () => {
      const { minimum, maximum } = brief.properties.min_length!;
      expect(example.brief.min_length).toBeGreaterThanOrEqual(minimum!);
      expect(example.brief.max_length).toBeLessThanOrEqual(maximum!);
      expect(example.brief.min_length).toBeLessThanOrEqual(example.brief.max_length);
    });

    it("keeps the dosing interval inside the API's bound", () => {
      const { minimum, maximum } = brief.properties.dosing_interval_hours!;
      expect(example.brief.dosing_interval_hours).toBeGreaterThanOrEqual(minimum!);
      expect(example.brief.dosing_interval_hours).toBeLessThanOrEqual(maximum!);
    });

    it("has a label and a note, since both are shown to the user", () => {
      expect(example.label.length).toBeGreaterThan(0);
      expect(example.note.length).toBeGreaterThan(0);
    });
  },
);

describe("the form's starting state", () => {
  it("selects nothing the user did not choose", () => {
    expect(EMPTY_BRIEF.wound_context).toEqual([]);
    expect(EMPTY_BRIEF.desired_functions).toEqual([]);
    expect(EMPTY_BRIEF.pathogens).toEqual([]);
  });

  it("starts the numeric fields inside the API's bounds", () => {
    expect(EMPTY_BRIEF.min_length).toBeGreaterThanOrEqual(brief.properties.min_length!.minimum!);
    expect(EMPTY_BRIEF.max_length).toBeLessThanOrEqual(brief.properties.max_length!.maximum!);
    expect(EMPTY_BRIEF.dosing_interval_hours).toBeGreaterThanOrEqual(
      brief.properties.dosing_interval_hours!.minimum!,
    );
  });
});

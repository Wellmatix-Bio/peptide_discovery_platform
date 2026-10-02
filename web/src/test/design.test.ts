/* The copied design system must stay a copy.
 *
 * Compares against the ORIGINAL FILE on disk rather than a list of rules repeated here, because a
 * remembered list drifts exactly as silently as the thing it is meant to guard. If the source
 * project is not available the test SKIPS with a reason rather than passing quietly -- a green
 * tick that checked nothing is worse than a skip.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const SOURCE = "/home/ajit/Documents/wmxccs/web/src/styles.css";
const COPY = resolve(dirname(fileURLToPath(import.meta.url)), "../styles.css");

const available = existsSync(SOURCE);

/** Every rule as selector -> body, from the single-line minified block and the added blocks. */
function rules(css: string): Map<string, string> {
  const found = new Map<string, string>();
  const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");
  // Media queries are matched whole, so their inner rules are not mistaken for top-level ones.
  const media = withoutComments.match(/@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g) ?? [];
  for (const block of media) found.set(block.slice(0, block.indexOf("{")).trim(), block);
  const flat = withoutComments.replace(/@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, "");
  for (const match of flat.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    found.set(match[1]!.trim(), match[2]!.trim());
  }
  return found;
}

describe.skipIf(!available)("the design system is copied, not reinterpreted", () => {
  const source = available ? rules(readFileSync(SOURCE, "utf8")) : new Map();
  const copy = available ? rules(readFileSync(COPY, "utf8")) : new Map();

  it("found rules in both files, so this cannot pass vacuously", () => {
    expect(source.size).toBeGreaterThan(40);
    expect(copy.size).toBeGreaterThan(40);
  });

  it("keeps the :root tokens byte-identical", () => {
    expect(copy.get(":root")).toBe(source.get(":root"));
  });

  it.each([
    "body",
    ".top",
    ".brand",
    ".tag",
    ".shell",
    "aside",
    ".nav",
    ".main",
    ".hero",
    ".badge",
    ".card",
    ".field label",
    ".btn",
    ".primary",
    ".secondary",
    ".outline",
    ".metric",
    ".candidate",
    ".chip",
    ".status",
    ".tabs",
    ".tab",
    ".help",
    ".mono",
    ".err",
    "table.runs",
    ".kv",
  ])("keeps %s exactly as the source has it", (selector) => {
    expect(source.has(selector)).toBe(true);
    expect(copy.get(selector)).toBe(source.get(selector));
  });

  it("keeps the mobile breakpoint", () => {
    const key = [...copy.keys()].find((one) => one.startsWith("@media") && one.includes("900px"));
    expect(key).toBeDefined();
  });

  it("does not carry over classes this app has no use for", () => {
    // .band and .klass are the source's glycan-specific boxes. An unused class invites someone to
    // reuse it for something it does not mean (see the spec's own .top collision).
    expect(copy.has(".band")).toBe(false);
    expect(copy.has(".klass")).toBe(false);
  });
});

describe.skipIf(available)("design source unavailable", () => {
  it("reports that it could not check", () => {
    console.warn(`design.test.ts: ${SOURCE} is not present, so the copy was not verified`);
    expect(available).toBe(false);
  });
});


describe("the additions correct what the source's form rules do to checkboxes", () => {
  const css = readFileSync(COPY, "utf8");

  it("still inherits the source's .field input rule, which is why the override is needed", () => {
    // The verbatim block styles every input in a .field as a text box: full width, padding, a
    // border and a white background. Applied to a checkbox that renders a full-width box
    // floating above its own label, which is what the Stages list and the multi-selects looked
    // like before. If this rule ever stops existing, the override below is dead weight.
    expect(css).toContain(".field input,.field select,.field textarea{width:100%");
  });

  it("resets width, padding, border and background for checkboxes and radios", () => {
    const override = css.match(
      /\.field input\[type="checkbox"\],\.field input\[type="radio"\]\{([^}]*)\}/,
    );
    expect(override, "the checkbox override is missing").not.toBeNull();
    const body = override![1]!;
    expect(body).toContain("width:auto");
    expect(body).toContain("padding:0");
    expect(body).toContain("border:0");
    expect(body).toContain("background:none");
  });

  it("hides the native control in the selection components without removing it", () => {
    // Visually hidden, not display:none -- it has to stay focusable and announced.
    for (const selector of [".choice input{", ".toggle input{"]) {
      const rule = css.slice(css.indexOf(selector)).slice(0, 200);
      expect(rule).toContain("opacity:0");
      expect(rule).not.toContain("display:none");
      expect(rule).not.toContain("visibility:hidden");
    }
  });

  it("gives both selection components a visible focus ring", () => {
    expect(css).toContain(".choice input:focus-visible+.mark");
    expect(css).toContain(".toggle input:focus-visible+.switch");
  });
});

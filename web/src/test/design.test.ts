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

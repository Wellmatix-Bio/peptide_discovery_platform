/* Write web/src/test/design-manifest.json from the reference project's styles.css.
 *
 * WHY A MANIFEST. design.test.ts compares the copied design system against the ORIGINAL FILE on
 * disk, which exists on one developer's machine. Everywhere else -- including CI -- 31 of its 36
 * tests skipped, so the guard that keeps the copy faithful ran nowhere that mattered.
 *
 * The manifest is the source's selector -> body map, committed. It cannot detect the SOURCE
 * changing (nothing in CI can see the source), but it does detect the COPY drifting, which is
 * what the guard is for and what a contributor can actually cause.
 *
 *   node web/scripts/design_manifest.mjs [path-to-source-styles.css]
 *
 * Re-run it when the reference design system changes, and commit the result.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE =
  process.argv[2] ??
  process.env.WMXCCS_STYLES ??
  "/home/ajit/Documents/wmxccs/web/src/styles.css";
const OUT = resolve(HERE, "../src/test/design-manifest.json");

if (!existsSync(SOURCE)) {
  console.error(`Source design system not found: ${SOURCE}`);
  console.error("Pass the path as an argument or set WMXCCS_STYLES.");
  process.exit(2);
}

/** Must stay identical to rules() in design.test.ts, which imports this file's output. */
function rules(css) {
  const found = new Map();
  const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const media = withoutComments.match(/@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g) ?? [];
  for (const block of media) found.set(block.slice(0, block.indexOf("{")).trim(), block);
  const flat = withoutComments.replace(/@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, "");
  for (const match of flat.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    found.set(match[1].trim(), match[2].trim());
  }
  return found;
}

const map = rules(readFileSync(SOURCE, "utf8"));
if (map.size < 40) {
  console.error(`Only ${map.size} rules parsed from ${SOURCE}; refusing to write a thin manifest.`);
  process.exit(2);
}
writeFileSync(OUT, `${JSON.stringify(Object.fromEntries(map), null, 2)}\n`);
console.log(`wrote ${OUT} (${map.size} rules from ${SOURCE})`);

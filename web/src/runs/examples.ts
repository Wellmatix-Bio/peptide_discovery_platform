/* Starting points for the brief form.
 *
 * These are NOT the repository's data/briefs/*.json. None of those 32 files is a valid request:
 * they use 35 wound_context values, 20 desired_functions and 12 pathogens the API rejects, so
 * offering them would hand the user a 422 (docs/BRIEF_VALIDITY.md). Rather than silently
 * stripping the parts the API refuses -- which changes the question a brief is asking -- this
 * app ships its own examples, built only from vocabulary the API accepts.
 *
 * Every value below is checked against web/openapi/peptide.json by
 * src/test/examples.test.ts, so an enum narrowed in api.py breaks the test rather than the user's
 * submission. The types come from the generated schema, so a REMOVED enum value is a compile
 * error here.
 */
import type {
  BriefFields,
  CosmeticBrief,
  DesiredFunction,
  MetabolicBrief,
  Pathogen,
  TargetReceptor,
  WoundContext,
} from "../api/types";

export interface Example {
  id: string;
  label: string;
  /** Why someone would pick this one. Shown under the label. */
  note: string;
  brief: BriefFields;
}

const wound = (...values: WoundContext[]): WoundContext[] => values;
const functions = (...values: DesiredFunction[]): DesiredFunction[] => values;
const pathogens = (...values: Pathogen[]): Pathogen[] => values;

export const EXAMPLES: Example[] = [
  {
    id: "infected-diabetic-ulcer",
    label: "Infected diabetic foot ulcer",
    note:
      "The platform's main case: a chronic, poorly perfused, infected wound needing both" +
      " antimicrobial action and active healing.",
    brief: {
      wound_context: wound("chronic", "infected", "diabetic", "high_glucose", "ischemic"),
      desired_functions: functions(
        "antimicrobial",
        "keratinocyte_migration",
        "angiogenesis",
        "anti_inflammatory",
      ),
      pathogens: pathogens("Staphylococcus_aureus", "Pseudomonas_aeruginosa"),
      min_length: 6,
      max_length: 30,
      dosing_interval_hours: 48,
    },
  },
  {
    id: "biofilm-positive-ulcer",
    label: "Biofilm-positive chronic ulcer",
    note:
      "Uses the biofilm_positive context. Note that “antibiofilm” is not an accepted" +
      " desired function, so the brief asks for antimicrobial action instead — the biofilm" +
      " model still scores every candidate. “high_exudate” was dropped from the accepted" +
      " contexts: it affects dressing choice rather than any model the pipeline runs.",
    brief: {
      wound_context: wound("chronic", "infected", "biofilm_positive"),
      desired_functions: functions("antimicrobial", "anti_inflammatory", "immunomodulation"),
      pathogens: pathogens("Pseudomonas_aeruginosa", "Staphylococcus_aureus"),
      min_length: 10,
      max_length: 40,
      dosing_interval_hours: 24,
    },
  },
  {
    id: "clean-surgical",
    label: "Clean surgical incision",
    note:
      "No infection to treat: the goal is closure and controlled remodelling. The" +
      " \u201cclean\u201d context the name refers to is no longer an accepted value, so the" +
      " brief says acute and surgical instead.",
    brief: {
      wound_context: wound("acute", "surgical"),
      desired_functions: functions(
        "cell_proliferation/migration",
        "collagen_synthesis",
        "collagen_remodeling",
      ),
      pathogens: pathogens("Staphylococcus_aureus"),
      min_length: 6,
      max_length: 20,
      dosing_interval_hours: 24,
    },
  },
  {
    id: "burn",
    label: "Second-degree burn",
    note: "A large acute wound at high infection risk, needing re-vascularisation.",
    brief: {
      wound_context: wound("acute", "burn", "infected"),
      desired_functions: functions(
        "antimicrobial",
        "keratinocyte_migration",
        "angiogenesis",
        "anti_inflammatory",
      ),
      pathogens: pathogens("Pseudomonas_aeruginosa", "Staphylococcus_aureus"),
      min_length: 8,
      max_length: 30,
      dosing_interval_hours: 12,
    },
  },
  {
    id: "minimal",
    label: "Minimal brief",
    note:
      "The smallest thing the API accepts — one context, one function, one pathogen. Useful" +
      " for a first run, since every other stage takes its defaults.",
    brief: {
      wound_context: wound("acute"),
      desired_functions: functions("antimicrobial"),
      pathogens: pathogens("Escherichia_coli"),
      min_length: 6,
      max_length: 20,
      dosing_interval_hours: 24,
    },
  },
];

/** The form's own starting state: the minimal example, so nothing is pre-selected that the user
 *  did not choose beyond what the API requires. */
export const EMPTY_BRIEF: BriefFields = {
  wound_context: [],
  desired_functions: [],
  pathogens: [],
  min_length: 6,
  max_length: 30,
  dosing_interval_hours: 24,
};

/* ---- The metabolic pipeline ----
 *
 * A DIFFERENT brief schema, not the wound-healing one with extra fields: `target_receptors` is
 * required and is its only enum, and there is no dosing interval. Its other three fields
 * (desired_functions, pathogens, wound_context) are plain string arrays with no enum, so nothing
 * here sets them -- an unset optional field is always valid, while a guessed value is a guess
 * about what the ranking stage does with it. The JSON tab reaches them.
 *
 * Checked against MetabolicBrief in the snapshot by src/test/examples.test.ts, the same way.
 */
export interface MetabolicExample {
  id: string;
  label: string;
  note: string;
  brief: MetabolicBrief;
}

const receptors = (...values: TargetReceptor[]): TargetReceptor[] => values;

export const METABOLIC_EXAMPLES: MetabolicExample[] = [
  {
    id: "glp1r-agonist",
    label: "GLP-1 receptor agonist",
    note:
      "One receptor, the best-supported of the three: GLP1R has 271 training labels against" +
      " GIPR's 55, and it is the only one whose potency model ranks candidates usefully.",
    brief: {
      target_receptors: receptors("GLP1R"),
      min_length: 20,
      max_length: 40,
    },
  },
  {
    id: "dual-glp1r-gcgr",
    label: "Dual GLP-1 / glucagon agonist",
    note:
      "The pipeline's own default pair. Both receptors have potency models whose ranking carries" +
      " information, which is why these two are the default and GIPR is not.",
    brief: {
      target_receptors: receptors("GLP1R", "GCGR"),
      min_length: 25,
      max_length: 45,
    },
  },
  {
    id: "triple-agonist",
    label: "Triple agonist, including GIPR",
    note:
      "Includes GIPR deliberately. Its potency values are returned and are flagged as not" +
      " predictive — leave-one-cluster-out R² is −0.108, worse than always guessing the training" +
      " mean — and they still count toward the gates and the ranking score. Pick this only if you" +
      " want the GIPR numbers for their own sake.",
    brief: {
      target_receptors: receptors("GLP1R", "GCGR", "GIPR"),
      min_length: 25,
      max_length: 45,
    },
  },
  {
    id: "metabolic-minimal",
    label: "Minimal metabolic brief",
    note: "The smallest thing the API accepts: one receptor and a length range. Every stage takes its defaults.",
    brief: {
      target_receptors: receptors("GLP1R"),
      min_length: 20,
      max_length: 35,
    },
  },
];

/** Nothing pre-selected beyond the length range, as with EMPTY_BRIEF. `target_receptors` starts
 *  empty even though the API requires at least one, so the form asks rather than assumes -- which
 *  receptor a run targets is the whole question a metabolic brief answers. */
export const EMPTY_METABOLIC_BRIEF: MetabolicBrief = {
  target_receptors: [],
  min_length: 20,
  max_length: 40,
};

/* ---- The cosmetic pipeline ----
 *
 * A NESTED brief, and one where the API requires nothing: every field has a server-side default,
 * so `{}` is a valid cosmetic brief. The examples still spell every field out, because a default
 * is not an answer -- someone reading back what they submitted should see what was asked for, not
 * have to look up what the server filled in.
 *
 * Checked against CosmeticBrief and its four nested objects by src/test/examples.test.ts.
 */
export interface CosmeticExample {
  id: string;
  label: string;
  note: string;
  brief: CosmeticBrief;
}

export const COSMETIC_EXAMPLES: CosmeticExample[] = [
  {
    id: "anti-wrinkle-face",
    label: "Anti-wrinkle face serum",
    note:
      "The pipeline's main cosmetic case: collagen and ECM synthesis for facial skin, delivered" +
      " to the dermis where the fibroblasts are.",
    brief: {
      application: { area: "face", sensitive_skin: false },
      functions: { primary: "collagen_ecm", secondary: ["antioxidant"] },
      peptide: { min_length: 6, max_length: 12 },
      product: { ph: 5.5 },
      target_depth: "dermis",
    },
  },
  {
    id: "brightening-pigmentation",
    label: "Brightening / pigmentation",
    note:
      "Targets pigmentation first. The epidermis is where melanocytes sit, so a deeper target" +
      " would be asking the ranking stage to reward delivery past the cells that matter.",
    brief: {
      application: { area: "face", sensitive_skin: false },
      functions: { primary: "pigmentation", secondary: ["antioxidant"] },
      peptide: { min_length: 6, max_length: 10 },
      product: { ph: 5.0 },
      target_depth: "epidermis",
    },
  },
  {
    id: "sensitive-eye-area",
    label: "Sensitive eye-area cream",
    note:
      "Sets sensitive_skin, which tightens the irritation gate in stage 8, and keeps the peptide" +
      " short. The eye area is its own application area in the API's vocabulary.",
    brief: {
      application: { area: "eye_area", sensitive_skin: true },
      functions: { primary: "antioxidant", secondary: [] },
      peptide: { min_length: 6, max_length: 8 },
      product: { ph: 5.5 },
      target_depth: "epidermis",
    },
  },
  {
    id: "cosmetic-minimal",
    label: "Minimal cosmetic brief",
    note:
      "One function and nothing else chosen. Every other field here is the API's own default," +
      " written out rather than omitted so the submitted brief says what it asked for.",
    brief: {
      application: { area: "face", sensitive_skin: false },
      functions: { primary: "collagen_ecm", secondary: [] },
      peptide: { min_length: 6, max_length: 12 },
      product: { ph: 5.5 },
      target_depth: null,
    },
  },
];

/** The cosmetic form's starting state: the API's own defaults, spelled out, with no target depth
 *  chosen. `target_depth` is nullable server-side and a null means "do not weight delivery", which
 *  is a different request from any of the three depths -- so it starts null rather than guessing. */
/* Required<>, not CosmeticBrief: the API makes every field optional because every field has
   a default, but THIS object is the form's starting point and must be complete, so the
   nested setters in SubmitRun have something whole to spread from. */
export const EMPTY_COSMETIC_BRIEF: Required<CosmeticBrief> = {
  application: { area: "face", sensitive_skin: false },
  functions: { primary: "collagen_ecm", secondary: [] },
  peptide: { min_length: 6, max_length: 12 },
  product: { ph: 5.5 },
  target_depth: null,
};

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
import type { BriefFields, DesiredFunction, Pathogen, WoundContext } from "../api/types";

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
      " model still scores every candidate.",
    brief: {
      wound_context: wound("chronic", "infected", "biofilm_positive", "high_exudate"),
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
    note: "No infection to treat: the goal is closure and controlled remodelling.",
    brief: {
      wound_context: wound("acute", "clean", "surgical"),
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
      wound_context: wound("acute", "burn", "high_exudate", "infected"),
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

/* The form's options and bounds, read at runtime from web/openapi/peptide.json.
 *
 * Deliberately not a hand-written list. §6.3: build each form from the API's validators. If
 * api.py gains a wound_context value, it appears in the form as soon as the snapshot is
 * regenerated; if it loses one, the generated TYPES break the code that referenced it. A list
 * retyped here would drift silently and produce 422s the user cannot act on.
 */
import snapshot from "../../openapi/peptide.json";

type Property = {
  items?: { enum?: string[] };
  enum?: string[];
  minimum?: number;
  maximum?: number;
  default?: unknown;
  anyOf?: Property[];
};

type Snapshot = {
  components: { schemas: Record<string, { required?: string[]; properties?: Record<string, Property> }> };
};

const schemas = (snapshot as unknown as Snapshot).components.schemas;

function property(schema: string, field: string): Property {
  const found = schemas[schema]?.properties?.[field];
  if (!found) throw new Error(`${schema}.${field} is not in the OpenAPI snapshot`);
  return found;
}

/** Enum values for a field, whether it is a scalar enum or an array of one. */
export function options(schema: string, field: string): string[] {
  const p = property(schema, field);
  const values = p.items?.enum ?? p.enum ?? p.anyOf?.flatMap((one) => one.enum ?? [])
  if (!values || values.length === 0) {
    throw new Error(`${schema}.${field} has no enum in the OpenAPI snapshot`);
  }
  return values;
}

export interface Bound {
  min: number;
  max: number;
}

export function bound(schema: string, field: string): Bound {
  const p = property(schema, field);
  if (p.minimum === undefined || p.maximum === undefined) {
    throw new Error(`${schema}.${field} has no bounds in the OpenAPI snapshot`);
  }
  return { min: p.minimum, max: p.maximum };
}

export function defaultOf<T>(schema: string, field: string, fallback: T): T {
  const value = property(schema, field).default;
  return (value === undefined ? fallback : value) as T;
}

export const WOUND_CONTEXTS = options("BriefFields", "wound_context");
export const DESIRED_FUNCTIONS = options("BriefFields", "desired_functions");
export const PATHOGENS = options("BriefFields", "pathogens");
export const CYTOTOXICITY_CELL_TYPES = options("Stage8Request", "cytotoxicity_cell_type");
export const GENERATION_TAGS = options("Stage4Request", "tags");

export const LENGTH_BOUND = bound("BriefFields", "min_length");
export const DOSING_BOUND = bound("BriefFields", "dosing_interval_hours");

/** "high_glucose" -> "high glucose". Underscores are the API's spelling, not a person's. */
export const humanise = (value: string): string => value.replace(/_/g, " ");

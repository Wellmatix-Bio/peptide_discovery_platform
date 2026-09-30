/* Aliases for the generated schemas, so pages import a name rather than a lookup path.
   peptide.gen.ts is GENERATED from web/openapi/peptide.json by `npm run api:types`; never edit
   it, and never hand-write a type that duplicates one of its fields. CI regenerates it and fails
   on any diff, so a field renamed in api.py surfaces here as a compile error. */
import type { components } from "./peptide.gen";

export type CreateJobRequest = components["schemas"]["CreateJobRequest"];
export type CreateJobResponse = components["schemas"]["CreateJobResponse"];
export type JobStatusResponse = components["schemas"]["JobStatusResponse"];
export type JobResultsResponse = components["schemas"]["JobResultsResponse"];
export type CandidateResponse = components["schemas"]["CandidateResponse"];
export type ComponentStats = components["schemas"]["ComponentStats"];
export type BriefFields = components["schemas"]["BriefFields"];
export type E2ERequest = components["schemas"]["E2ERequest"];
export type Stage1Request = components["schemas"]["Stage1Request"];
export type Stage4Request = components["schemas"]["Stage4Request"];
export type Stage5Request = components["schemas"]["Stage5Request"];
export type Stage6Request = components["schemas"]["Stage6Request"];
export type Stage7Request = components["schemas"]["Stage7Request"];
export type Stage8Request = components["schemas"]["Stage8Request"];
export type Stage9Request = components["schemas"]["Stage9Request"];

export type WoundContext = BriefFields["wound_context"][number];
export type DesiredFunction = BriefFields["desired_functions"][number];
export type Pathogen = BriefFields["pathogens"][number];
export type CytotoxicityCellType = NonNullable<Stage8Request["cytotoxicity_cell_type"]>;

/* --- The accounts service publishes no OpenAPI, so its few shapes are hand-written.
   src/test/accounts-types.test.ts checks them against real responses from the running service. */

export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface Session {
  token: string;
  expires_in: number;
  user: User;
}

export interface Registration extends Session {
  recovery_code: string;
}

export interface RecoveredSession extends Session {
  recovery_code: string;
}

export interface Captcha {
  nonce: string;
  svg: string;
  expires_in: number;
}

/** A history row. `status`, `stage` and `vertex_state` are as of `observed_at`, NOT as of the
 *  read: the accounts service records what it saw pass through and does not poll on anyone's
 *  behalf. Never render them without saying when they were seen. */
export interface HistoryEntry {
  id: number;
  service: string;
  kind: string;
  resource_id: string | null;
  run_name: string | null;
  status_code: number;
  summary: string;
  status: string | null;
  stage: string | null;
  vertex_state: string | null;
  has_envelope: boolean;
  envelope_note: string | null;
  created_at: string;
  observed_at: string;
}

export interface HistoryPage {
  total: number;
  returned: number;
  limit: number;
  offset: number;
  entries: HistoryEntry[];
  note: string;
  staleness: string;
}

export interface HistoryDetail extends Omit<HistoryEntry, "has_envelope"> {
  envelope: JobResultsResponse | null;
  staleness: string;
}

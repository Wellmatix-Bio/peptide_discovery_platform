/* The job API's four endpoints, and the two facts about them that every view depends on.
   Nothing here computes a value the API did not return. */
import { call } from "./client";
import type {
  CreateJobRequest,
  CreateJobResponse,
  HistoryDetail,
  HistoryPage,
  JobResultsResponse,
  JobStatusResponse,
} from "./types";

const BASE = "/api/peptide/api/v1/jobs";

/** Vertex states that mean the job will not progress further. Mirrors
 *  services/accounts/accounts/history.py's TERMINAL_VERTEX_STATES. */
export const TERMINAL_VERTEX_STATES = [
  "JOB_STATE_SUCCEEDED",
  "JOB_STATE_FAILED",
  "JOB_STATE_CANCELLED",
  "JOB_STATE_EXPIRED",
] as const;

export const RUN_STATES = [
  "submitted",
  "running",
  "succeeded",
  "failed",
  "cancelled",
  "abandoned",
] as const;

export type RunState =
  | "submitted"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "abandoned";

/** What the run actually is, from BOTH fields the API returns.
 *
 *  The API derives `status` only from the worker's results.json, so a worker that died without
 *  writing it reports "pending" forever while Vertex says JOB_STATE_FAILED (docs/BASELINE.md).
 *  "abandoned" is that case, and it exists so no view can render a dead run as still working.
 *  This is a classification of two returned fields, not a computed result. */
/** The API now returns `run_state` itself, reconciled server-side. Prefer it: this function is
 *  the fallback for a response from an older API that does not carry the field, and the two are
 *  held to agree by web/src/test/runstate.test.ts. Two implementations of one rule is a thing to
 *  remove, not to keep -- this one goes when no deployed API predates the field. */
export function serverRunState(body: {
  run_state?: string | null;
  status?: string | null;
  vertex_state?: string | null;
}): RunState {
  const given = body.run_state;
  if (given && (RUN_STATES as readonly string[]).includes(given)) return given as RunState;
  return runState(body.status ?? null, body.vertex_state ?? null);
}

export function runState(status: string | null, vertexState: string | null): RunState {
  /* The order matters and mirrors run_state() in src/backend/api_e2e/api.py exactly.
   *
   * An earlier version checked `terminal && (status === null || status === "pending")` for
   * `abandoned`, and `status === "running"` below it. A worker that reported progress and was
   * then killed leaves status at "running" forever, so that pair sent it to `running` -- the UI
   * showing a finished job as still working, which is the defect run_state was added to end.
   * The terminal check now has no status condition at all. */
  const terminal = vertexState !== null && (TERMINAL_VERTEX_STATES as readonly string[]).includes(vertexState);
  if (status === "success") return "succeeded";
  if (status === "failed" || status === "fail") return "failed";
  if (vertexState === "JOB_STATE_CANCELLED") return "cancelled";
  if (vertexState === "JOB_STATE_FAILED" || vertexState === "JOB_STATE_EXPIRED") return "failed";
  if (terminal) return "abandoned";
  if (status === "running") return "running";
  return "submitted";
}

export function isFinished(state: RunState): boolean {
  return state === "succeeded" || state === "failed" || state === "cancelled" || state === "abandoned";
}

/** What to tell the reader, in the API's terms and no stronger. */
export const STATE_EXPLANATION: Record<RunState, string> = {
  submitted:
    "Accepted by Vertex AI. The worker has not written any progress yet, so there is nothing to" +
    " report beyond that.",
  running: "The worker has reported progress. The stage shown is the last one it recorded.",
  succeeded: "The worker finished and wrote its results.",
  failed: "The run did not finish. The reason below is the one the API reported.",
  cancelled: "Cancelled. Results were not produced.",
  abandoned:
    "Vertex AI reports this job has finished, but the worker never wrote a result. The API" +
    " therefore still reports its status as “pending”; it is not still working. This" +
    " usually means the worker stopped before it could write anything — out of memory, a" +
    " CUDA failure, or a preempted machine.",
};

export function createJob(request: CreateJobRequest) {
  return call<CreateJobResponse>(`${BASE}/create`, { method: "POST", body: request });
}

export function jobStatus(jobId: string, signal?: AbortSignal) {
  return call<JobStatusResponse>(`${BASE}/${jobId}/status`, { signal });
}

/** Results are only present once the worker wrote them; a 404 here is an answer ("not yet"),
 *  not a failure, so it is accepted rather than thrown. */
export function jobResults(jobId: string, signal?: AbortSignal) {
  return call<JobResultsResponse>(`${BASE}/${jobId}/results`, { accept: [404], signal });
}

export function cancelJob(jobId: string) {
  return call<unknown>(`${BASE}/${jobId}/cancel`, { method: "POST" });
}

export function history(params: { limit?: number; offset?: number } = {}) {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  const suffix = query.toString();
  return call<HistoryPage>(`/auth/history${suffix ? `?${suffix}` : ""}`);
}

export function historyEntry(id: number) {
  return call<HistoryDetail>(`/auth/history/${id}`);
}

/** Removes only this account's history row. The run's artifacts in Cloud Storage are untouched;
 *  callers must say "hide", never "delete". */
export function hideHistoryEntry(id: number) {
  return call<{ hidden: number; artifacts_deleted: boolean; why: string }>(`/auth/history/${id}`, {
    method: "DELETE",
  });
}

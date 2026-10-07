/* The server and the client must classify a run identically.
 *
 * The API now returns `run_state`, reconciled from the worker's self-report and the live Vertex
 * state. The client keeps `runState()` as a fallback for a response from an older API, which
 * means the rule exists twice -- and two implementations of one rule drift, silently, in the
 * direction nobody is testing.
 *
 * These tests hold them together by enumerating every (worker status, vertex state) pair the
 * API can produce and asserting the client reaches the same verdict. The expectations are
 * written out rather than derived, so a change on either side has to be made deliberately here.
 *
 * When no deployed API predates `run_state`, delete runState(), serverRunState's fallback, and
 * most of this file.
 */
import { describe, expect, it } from "vitest";

import { RUN_STATES, runState, serverRunState } from "../api/peptide";

/** Every combination the API can emit: the worker writes one of these, Vertex reports one of
 *  those, and the third column is what BOTH implementations must answer. */
const CASES: Array<[string | null, string, string]> = [
  // Nothing from the worker yet.
  [null, "JOB_STATE_QUEUED", "submitted"],
  [null, "JOB_STATE_PENDING", "submitted"],
  [null, "JOB_STATE_RUNNING", "submitted"],
  [null, "JOB_STATE_SUCCEEDED", "abandoned"],
  [null, "JOB_STATE_FAILED", "failed"],
  [null, "JOB_STATE_CANCELLED", "cancelled"],
  [null, "JOB_STATE_EXPIRED", "failed"],
  // The worker reported progress and then may or may not have survived.
  ["running", "JOB_STATE_RUNNING", "running"],
  ["running", "JOB_STATE_SUCCEEDED", "abandoned"],
  ["running", "JOB_STATE_FAILED", "failed"],
  ["running", "JOB_STATE_CANCELLED", "cancelled"],
  ["running", "JOB_STATE_EXPIRED", "failed"],
  // The worker finished. Its own word is first-hand and wins.
  ["success", "JOB_STATE_SUCCEEDED", "succeeded"],
  ["success", "JOB_STATE_RUNNING", "succeeded"],
  ["failed", "JOB_STATE_FAILED", "failed"],
  ["failed", "JOB_STATE_SUCCEEDED", "failed"],
];

describe("run state, classified the same way on both sides", () => {
  it.each(CASES)("worker %s + vertex %s is %s", (status, vertexState, expected) => {
    expect(runState(status, vertexState)).toBe(expected);
  });

  it.each(CASES)(
    "the server's answer is taken as-is: %s + %s",
    (status, vertexState, expected) => {
      expect(serverRunState({ run_state: expected, status, vertex_state: vertexState })).toBe(
        expected,
      );
    },
  );

  it("falls back to the client's rule when the API does not carry run_state", () => {
    for (const [status, vertexState, expected] of CASES) {
      expect(serverRunState({ status, vertex_state: vertexState })).toBe(expected);
    }
  });

  it("ignores a run_state the client does not recognise, rather than rendering it", () => {
    /* A value outside the vocabulary would reach the UI's state lookups and produce an undefined
       explanation. Falling back is the conservative answer. */
    expect(
      serverRunState({ run_state: "half-done", status: "running", vertex_state: "JOB_STATE_FAILED" }),
    ).toBe("failed");
  });

  it("every state in the vocabulary is covered by these cases", () => {
    const reached = new Set(CASES.map(([, , expected]) => expected));
    expect([...reached].sort()).toEqual([...RUN_STATES].sort());
  });

  it("a dead worker is never reported as still running", () => {
    /* The defect this field was added for. A worker killed mid-stage leaves `status` reading
       "running" forever; neither implementation may repeat that. */
    for (const terminal of ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]) {
      expect(runState("running", terminal)).not.toBe("running");
      expect(serverRunState({ status: "running", vertex_state: terminal })).not.toBe("running");
    }
  });
});

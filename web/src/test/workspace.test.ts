/* The remembered "current run" must not survive a change of account in the same tab.
 *
 * It is per TAB (sessionStorage), not per account. Before this was fixed, a second account
 * signing in to the same tab inherited the first one's "Current run" link and could read the
 * previous account's job id out of the href. The proxy refuses to open it, so no run data
 * leaked, but the id should not have been readable. Found by signing in as a second account
 * during the walkthrough (docs/WEB_WALKTHROUGH.md).
 */
import { afterEach, describe, expect, it } from "vitest";
import { lastRun } from "../workspace";

const JOB = "projects/p/locations/us-central1/customJobs/123";

afterEach(() => lastRun.clear());

describe("the remembered run", () => {
  it("round-trips within one tab", () => {
    lastRun.set(JOB);
    expect(lastRun.get()).toBe(JOB);
  });

  it("is cleared outright, not merely overwritten", () => {
    lastRun.set(JOB);
    lastRun.clear();
    expect(lastRun.get()).toBeNull();
  });

  it("survives storage being unavailable, because it is only a convenience", () => {
    const original = Object.getOwnPropertyDescriptor(window, "sessionStorage");
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get() {
        throw new Error("blocked, as in a private window");
      },
    });
    expect(() => lastRun.set(JOB)).not.toThrow();
    expect(lastRun.get()).toBeNull();
    expect(() => lastRun.clear()).not.toThrow();
    if (original) Object.defineProperty(window, "sessionStorage", original);
  });
});

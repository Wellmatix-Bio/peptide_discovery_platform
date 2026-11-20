import { describe, expect, it } from "vitest";
import { num, pct } from "../components/ui";

describe("num", () => {
  it("rounds ordinary values to the requested decimals", () => {
    expect(num(9.4, 2)).toBe("9.40");
    expect(num(0, 2)).toBe("0.00");
    expect(num(null)).toBeNull();
  });

  it("never rounds a nonzero value down to zero", () => {
    expect(num(0.0004, 2)).toBe("4e-4");
    expect(num(-0.0004, 2)).toBe("-4e-4");
    expect(num(1.234e-5, 3)).toBe("1.234e-5");
    expect(num(3.2e-12, 2)).toBe("3.2e-12");
  });

  it("keeps plain decimals when they are no longer than scientific", () => {
    expect(num(0.004999, 2)).toBe("0.004999");
  });
});

describe("pct", () => {
  it("does not show a nonzero probability as 0.00%", () => {
    expect(pct(0.000001)).toBe("1e-4%");
    expect(pct(0.5)).toBe("50.00%");
  });
});

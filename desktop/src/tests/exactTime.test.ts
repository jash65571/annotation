import { describe, expect, it } from "vitest";

import {
  formatExactSeconds,
  manuscriptDisplay,
  parseExactSeconds,
} from "../lib/exactTime";

describe("parseExactSeconds", () => {
  it("parses simple fractions to display floats", () => {
    expect(parseExactSeconds("45/2")).toBe(22.5);
    expect(parseExactSeconds("0/1")).toBe(0);
    expect(parseExactSeconds("1/4")).toBe(0.25);
  });

  it("parses bare integers as whole seconds", () => {
    expect(parseExactSeconds("7")).toBe(7);
    expect(parseExactSeconds("-3")).toBe(-3);
  });

  it("parses repeating fractions to the nearest float", () => {
    expect(parseExactSeconds("1/3")).toBeCloseTo(0.3333333333, 9);
  });

  it("returns NaN for malformed input", () => {
    expect(parseExactSeconds("")).toBeNaN();
    expect(parseExactSeconds("abc")).toBeNaN();
    expect(parseExactSeconds("1/0")).toBeNaN();
    expect(parseExactSeconds("1.5/2")).toBeNaN();
    expect(parseExactSeconds("1/-2")).toBeNaN();
  });
});

describe("formatExactSeconds", () => {
  it("renders exact terminating fractions", () => {
    expect(formatExactSeconds("45/2")).toBe("22.500000");
    expect(formatExactSeconds("45/2", 1)).toBe("22.5");
    expect(formatExactSeconds("0/1")).toBe("0.000000");
  });

  it("rounds repeating fractions correctly at 6 dp", () => {
    expect(formatExactSeconds("1/3")).toBe("0.333333");
    expect(formatExactSeconds("2/3")).toBe("0.666667");
    expect(formatExactSeconds("1/7")).toBe("0.142857");
  });

  it("rounds half-up at the boundary", () => {
    // 1/20 = 0.05 exactly -> half-up at 1 dp gives 0.1
    expect(formatExactSeconds("1/20", 1)).toBe("0.1");
    // 1/8 = 0.125 exactly -> half-up at 2 dp gives 0.13
    expect(formatExactSeconds("1/8", 2)).toBe("0.13");
    // 3/2 = 1.5 -> half-up at 0 dp gives 2
    expect(formatExactSeconds("3/2", 0)).toBe("2");
    // just below the half boundary rounds down
    expect(formatExactSeconds("49/1000", 1)).toBe("0.0");
    // just above rounds up
    expect(formatExactSeconds("51/1000", 1)).toBe("0.1");
  });

  it("handles values a float parse would misrender", () => {
    // 0.1 + 0.2 style traps do not exist: pure integer math throughout.
    expect(formatExactSeconds("3/10", 1)).toBe("0.3");
    expect(formatExactSeconds("1/1000000", 6)).toBe("0.000001");
    expect(formatExactSeconds("1/2000000", 6)).toBe("0.000001"); // 0.0000005 half-up
    expect(formatExactSeconds("1/3000000", 6)).toBe("0.000000");
  });

  it("carries rounding across the integer boundary", () => {
    expect(formatExactSeconds("1999/1000", 2)).toBe("2.00");
    expect(formatExactSeconds("999999/1000000", 3)).toBe("1.000");
  });

  it("rounds negatives half away from zero", () => {
    expect(formatExactSeconds("-1/20", 1)).toBe("-0.1");
    expect(formatExactSeconds("-1/3", 6)).toBe("-0.333333");
    expect(formatExactSeconds("-45/2", 1)).toBe("-22.5");
  });

  it("supports dp = 0 (whole seconds)", () => {
    expect(formatExactSeconds("7/2", 0)).toBe("4");
    expect(formatExactSeconds("10/3", 0)).toBe("3");
  });

  it("handles large exact PTS-style values without precision loss", () => {
    // 9007199254740993/1 is not representable as a float (2^53 + 1).
    expect(formatExactSeconds("9007199254740993/1", 0)).toBe("9007199254740993");
  });

  it("returns malformed input unchanged (display-only fallback)", () => {
    expect(formatExactSeconds("not-a-fraction")).toBe("not-a-fraction");
    expect(formatExactSeconds("1/0")).toBe("1/0");
  });
});

describe("manuscriptDisplay", () => {
  it("projects to 0.1s with ROUND_HALF_UP", () => {
    expect(manuscriptDisplay("45/2")).toBe("22.5s");
    expect(manuscriptDisplay("1/3")).toBe("0.3s");
    expect(manuscriptDisplay("2/3")).toBe("0.7s");
    // 0.05 exactly is a half boundary -> up to 0.1
    expect(manuscriptDisplay("1/20")).toBe("0.1s");
    // 4.25 -> 4.3 (half-up), 4.24 -> 4.2
    expect(manuscriptDisplay("17/4")).toBe("4.3s");
    expect(manuscriptDisplay("106/25")).toBe("4.2s");
    expect(manuscriptDisplay("0/1")).toBe("0.0s");
  });

  it("returns malformed input unchanged", () => {
    expect(manuscriptDisplay("??")).toBe("??");
  });
});

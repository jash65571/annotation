import { describe, expect, it } from "vitest";

import {
  contentRect,
  displayToSource,
  sourceToDisplayRect,
} from "../features/anchors/coords";

const SOURCE = { width: 1920, height: 1080 };

describe("contentRect", () => {
  it("fills the display exactly when aspect ratios match", () => {
    expect(contentRect({ width: 960, height: 540 }, SOURCE)).toEqual({
      x: 0,
      y: 0,
      width: 960,
      height: 540,
    });
  });

  it("returns null for degenerate sizes", () => {
    expect(contentRect({ width: 0, height: 540 }, SOURCE)).toBeNull();
    expect(contentRect({ width: 960, height: 540 }, { width: 0, height: 0 })).toBeNull();
    expect(contentRect({ width: -5, height: 540 }, SOURCE)).toBeNull();
  });
});

describe("displayToSource — no letterbox (exact fit)", () => {
  const display = { width: 1920, height: 1080 };

  it("maps 1:1 when display equals source", () => {
    expect(displayToSource({ x: 10, y: 20 }, display, SOURCE)).toEqual({ x: 10, y: 20 });
    expect(displayToSource({ x: 0, y: 0 }, display, SOURCE)).toEqual({ x: 0, y: 0 });
  });

  it("scales proportionally at half size", () => {
    const half = { width: 960, height: 540 };
    expect(displayToSource({ x: 480, y: 270 }, half, SOURCE)).toEqual({ x: 960, y: 540 });
    expect(displayToSource({ x: 1, y: 1 }, half, SOURCE)).toEqual({ x: 2, y: 2 });
  });

  it("clamps the far edges to the last valid pixel", () => {
    expect(displayToSource({ x: 1920, y: 1080 }, display, SOURCE)).toEqual({
      x: 1919,
      y: 1079,
    });
  });
});

describe("displayToSource — horizontal letterbox (wide container)", () => {
  // 2000x1080 container, 1920x1080 source: scale 1, bars 40px on each side.
  const display = { width: 2000, height: 1080 };

  it("returns null for points in the left/right bars", () => {
    expect(displayToSource({ x: 10, y: 500 }, display, SOURCE)).toBeNull();
    expect(displayToSource({ x: 39.9, y: 500 }, display, SOURCE)).toBeNull();
    expect(displayToSource({ x: 1990, y: 500 }, display, SOURCE)).toBeNull();
  });

  it("maps points inside the content, offset by the bar width", () => {
    expect(displayToSource({ x: 40, y: 0 }, display, SOURCE)).toEqual({ x: 0, y: 0 });
    expect(displayToSource({ x: 140, y: 200 }, display, SOURCE)).toEqual({ x: 100, y: 200 });
  });

  it("treats the exact content edge as inside (clamped)", () => {
    expect(displayToSource({ x: 1960, y: 1080 }, display, SOURCE)).toEqual({
      x: 1919,
      y: 1079,
    });
  });
});

describe("displayToSource — vertical letterbox (tall container)", () => {
  // 1000x2000 container, 1000x1000 source: scale 1, bars 500px top/bottom.
  const square = { width: 1000, height: 1000 };
  const display = { width: 1000, height: 2000 };

  it("returns null for points in the top/bottom bars", () => {
    expect(displayToSource({ x: 500, y: 100 }, display, square)).toBeNull();
    expect(displayToSource({ x: 500, y: 499.5 }, display, square)).toBeNull();
    expect(displayToSource({ x: 500, y: 1600 }, display, square)).toBeNull();
  });

  it("maps points inside the content, offset by the bar height", () => {
    expect(displayToSource({ x: 0, y: 500 }, display, square)).toEqual({ x: 0, y: 0 });
    expect(displayToSource({ x: 250, y: 1200 }, display, square)).toEqual({ x: 250, y: 700 });
  });
});

describe("displayToSource — scaled letterbox", () => {
  // 800x600 container, 1920x1080 source: scale = 800/1920, content 800x450,
  // vertical bars 75px top and bottom.
  const display = { width: 800, height: 600 };

  it("computes the scaled content rect", () => {
    const content = contentRect(display, SOURCE);
    expect(content).not.toBeNull();
    expect(content?.x).toBe(0);
    expect(content?.y).toBe(75);
    expect(content?.width).toBe(800);
    expect(content?.height).toBe(450);
  });

  it("maps through the scale into integer source pixels", () => {
    expect(displayToSource({ x: 400, y: 300 }, display, SOURCE)).toEqual({ x: 960, y: 540 });
    expect(displayToSource({ x: 0, y: 75 }, display, SOURCE)).toEqual({ x: 0, y: 0 });
  });

  it("rejects the bars and clamps the far corner", () => {
    expect(displayToSource({ x: 400, y: 74 }, display, SOURCE)).toBeNull();
    expect(displayToSource({ x: 400, y: 526 }, display, SOURCE)).toBeNull();
    expect(displayToSource({ x: 800, y: 525 }, display, SOURCE)).toEqual({
      x: 1919,
      y: 1079,
    });
  });

  it("always returns integers", () => {
    const mapped = displayToSource({ x: 123.4, y: 234.5 }, display, SOURCE);
    expect(mapped).not.toBeNull();
    expect(Number.isInteger(mapped?.x)).toBe(true);
    expect(Number.isInteger(mapped?.y)).toBe(true);
  });
});

describe("sourceToDisplayRect", () => {
  it("is identity at exact fit", () => {
    const rect = { x: 100, y: 50, width: 200, height: 80 };
    expect(sourceToDisplayRect(rect, { width: 1920, height: 1080 }, SOURCE)).toEqual(rect);
  });

  it("applies scale and letterbox offset", () => {
    const display = { width: 800, height: 600 }; // scale 800/1920, offsetY 75
    const rect = { x: 1920 / 2, y: 0, width: 960, height: 1080 };
    const mapped = sourceToDisplayRect(rect, display, SOURCE);
    expect(mapped).not.toBeNull();
    expect(mapped?.x).toBeCloseTo(400, 6);
    expect(mapped?.y).toBeCloseTo(75, 6);
    expect(mapped?.width).toBeCloseTo(400, 6);
    expect(mapped?.height).toBeCloseTo(450, 6);
  });

  it("returns null for degenerate geometry", () => {
    const rect = { x: 0, y: 0, width: 10, height: 10 };
    expect(sourceToDisplayRect(rect, { width: 0, height: 0 }, SOURCE)).toBeNull();
  });
});

describe("round-trip consistency", () => {
  it("display -> source -> display lands within one source pixel", () => {
    const display = { width: 800, height: 600 };
    const scale = 800 / 1920;
    const points = [
      { x: 3, y: 80 },
      { x: 400, y: 300 },
      { x: 799, y: 524 },
      { x: 123.7, y: 401.2 },
    ];
    for (const px of points) {
      const src = displayToSource(px, display, SOURCE);
      expect(src).not.toBeNull();
      if (src === null) continue;
      const back = sourceToDisplayRect(
        { x: src.x, y: src.y, width: 1, height: 1 },
        display,
        SOURCE,
      );
      expect(back).not.toBeNull();
      if (back === null) continue;
      // The original point must fall within the mapped 1px source cell
      // (allow a hair of float slack at clamped edges).
      expect(px.x).toBeGreaterThanOrEqual(back.x - 1e-6 - scale);
      expect(px.x).toBeLessThanOrEqual(back.x + back.width + scale + 1e-6);
      expect(px.y).toBeGreaterThanOrEqual(back.y - 1e-6 - scale);
      expect(px.y).toBeLessThanOrEqual(back.y + back.height + scale + 1e-6);
    }
  });

  it("source -> display -> source is exact for integer source points", () => {
    const display = { width: 960, height: 540 }; // scale 0.5 exact
    const sources = [
      { x: 0, y: 0 },
      { x: 960, y: 540 },
      { x: 1918, y: 1078 },
    ];
    for (const s of sources) {
      const rect = sourceToDisplayRect(
        { x: s.x, y: s.y, width: 1, height: 1 },
        display,
        SOURCE,
      );
      expect(rect).not.toBeNull();
      if (rect === null) continue;
      const back = displayToSource(
        { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 },
        display,
        SOURCE,
      );
      expect(back).toEqual(s);
    }
  });
});

/**
 * Pure coordinate math for the visual-anchor editor.
 *
 * A frame image is shown with `object-fit: contain` inside a display element,
 * so the rendered content is centered with letterbox bars on one axis. These
 * helpers map between display-element pixels and SOURCE video pixels. Source
 * coordinates are the only ones ever persisted; display coordinates are
 * ephemeral view math.
 */

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The rectangle (in display coordinates) actually covered by the source
 * content when it is object-fit:contain'd into `display`. Null when either
 * size is degenerate. */
export function contentRect(display: Size, source: Size): Rect | null {
  if (
    !(display.width > 0) ||
    !(display.height > 0) ||
    !(source.width > 0) ||
    !(source.height > 0)
  ) {
    return null;
  }
  const scale = Math.min(display.width / source.width, display.height / source.height);
  const width = source.width * scale;
  const height = source.height * scale;
  return {
    x: (display.width - width) / 2,
    y: (display.height - height) / 2,
    width,
    height,
  };
}

/**
 * Map a point in display-element pixels to SOURCE pixel coordinates
 * (integers, clamped to [0, source-1]). Returns null when the point falls in
 * the letterbox bars or the geometry is degenerate.
 */
export function displayToSource(px: Point, display: Size, source: Size): Point | null {
  const content = contentRect(display, source);
  if (content === null) return null;

  if (
    px.x < content.x ||
    px.y < content.y ||
    px.x > content.x + content.width ||
    px.y > content.y + content.height
  ) {
    return null;
  }

  const scale = content.width / source.width;
  const rawX = (px.x - content.x) / scale;
  const rawY = (px.y - content.y) / scale;
  const clamp = (v: number, max: number) => Math.min(Math.max(Math.floor(v), 0), max - 1);
  return {
    x: clamp(rawX, source.width),
    y: clamp(rawY, source.height),
  };
}

/**
 * Map a rectangle in SOURCE pixels to display-element pixels for drawing
 * existing anchor boxes. Returns null when the geometry is degenerate.
 */
export function sourceToDisplayRect(rect: Rect, display: Size, source: Size): Rect | null {
  const content = contentRect(display, source);
  if (content === null) return null;
  const scale = content.width / source.width;
  return {
    x: content.x + rect.x * scale,
    y: content.y + rect.y * scale,
    width: rect.width * scale,
    height: rect.height * scale,
  };
}

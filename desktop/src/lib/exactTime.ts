/**
 * Exact rational time helpers. The engine stores time as exact "num/den"
 * second strings; the UI only ever *projects* those for display. Decimal
 * projection is done with BigInt math (round half-up, away from zero) so the
 * displayed digits are the true rounding of the exact value — never a
 * parseFloat round-trip. Exact strings are display inputs only; the UI never
 * stores a float back as truth.
 */

interface ExactFraction {
  num: bigint;
  den: bigint;
}

const FRACTION_RE = /^(-?\d+)\s*\/\s*(\d+)$/;
const INTEGER_RE = /^(-?\d+)$/;

function parseFraction(exact: string): ExactFraction | null {
  const trimmed = exact.trim();
  const frac = FRACTION_RE.exec(trimmed);
  if (frac) {
    const numText = frac[1];
    const denText = frac[2];
    if (numText === undefined || denText === undefined) return null;
    const den = BigInt(denText);
    if (den === 0n) return null;
    return { num: BigInt(numText), den };
  }
  const whole = INTEGER_RE.exec(trimmed);
  if (whole) {
    const numText = whole[1];
    if (numText === undefined) return null;
    return { num: BigInt(numText), den: 1n };
  }
  return null;
}

/**
 * Float projection of an exact "num/den" seconds string, for layout math
 * (percent positions, seek targets) only — never for stored facts.
 * Returns NaN when the input is not a valid exact string.
 */
export function parseExactSeconds(exact: string): number {
  const frac = parseFraction(exact);
  if (frac === null) return NaN;
  return Number(frac.num) / Number(frac.den);
}

/**
 * Decimal rendering of an exact "num/den" seconds string with `dp` decimal
 * places, computed by BigInt long division with ROUND_HALF_UP (half away from
 * zero). Falls back to returning the raw input unchanged when it is not a
 * valid exact string (display-only, total function).
 */
export function formatExactSeconds(exact: string, dp = 6): string {
  const frac = parseFraction(exact);
  if (frac === null || dp < 0 || !Number.isInteger(dp)) return exact;

  const negative = frac.num < 0n;
  const magnitude = negative ? -frac.num : frac.num;
  const scale = 10n ** BigInt(dp);
  const scaled = magnitude * scale;
  let quotient = scaled / frac.den;
  const remainder = scaled % frac.den;
  if (remainder * 2n >= frac.den) quotient += 1n;

  const digits = quotient.toString().padStart(dp + 1, "0");
  const intPart = digits.slice(0, digits.length - dp);
  const fracPart = dp > 0 ? `.${digits.slice(digits.length - dp)}` : "";
  const sign = negative && quotient !== 0n ? "-" : "";
  return `${sign}${intPart}${fracPart}`;
}

/**
 * The Manuscript display projection: exact seconds rounded to 0.1s with
 * ROUND_HALF_UP, e.g. "45/2" -> "22.5s", "1/20" (0.05) -> "0.1s".
 */
export function manuscriptDisplay(exact: string): string {
  const frac = parseFraction(exact);
  if (frac === null) return exact;
  return `${formatExactSeconds(exact, 1)}s`;
}

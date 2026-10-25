// Shared race maths for the UI: tyre colours, strategy edits and replay interpolation.

import type { Compound, LapEvent, Stint } from "./api";

export const COMPOUNDS: Compound[] = ["SOFT", "MEDIUM", "HARD"];

export const COMPOUND_COLOR: Record<Compound, string> = {
  SOFT: "var(--soft)",
  MEDIUM: "var(--medium)",
  HARD: "var(--hard)",
};

export const HIGHLIGHT = ["var(--focus)", "var(--rival)", "var(--third)"] as const;

export function fmtGap(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "–";
  if (s === 0) return "leader";
  return `+${s.toFixed(1)} s`;
}

export function fmtDelta(s: number | null | undefined, unit = " s"): string {
  if (s == null || !Number.isFinite(s)) return "–";
  return `${s > 0 ? "+" : s < 0 ? "−" : "±"}${Math.abs(s).toFixed(1)}${unit}`;
}

export const pct = (p: number) => `${Math.round(p * 100)}%`;

export function describe(stints: Stint[]): string {
  const letters = stints.map((s) => s.compound[0]).join("-");
  const stops = stints.slice(0, -1).map((s) => s.end_lap);
  return stops.length ? `${letters} (stops ${stops.join(", ")})` : `${letters} (no stop)`;
}

/** Status of each lap (1-based index into the returned array, index 0 unused). */
export function lapStatus(events: LapEvent[], totalLaps: number): (LapEvent["kind"] | null)[] {
  const status: (LapEvent["kind"] | null)[] = Array(totalLaps + 1).fill(null);
  for (const e of events) if (e.lap <= totalLaps) status[e.lap] = e.kind;
  return status;
}

/** Contiguous neutralised periods, for shading charts. */
export function eventSpans(events: LapEvent[]): { from: number; to: number; kind: LapEvent["kind"] }[] {
  const sorted = [...events].sort((a, b) => a.lap - b.lap);
  const spans: { from: number; to: number; kind: LapEvent["kind"] }[] = [];
  for (const e of sorted) {
    const last = spans[spans.length - 1];
    if (last && last.kind === e.kind && e.lap === last.to + 1) last.to = e.lap;
    else spans.push({ from: e.lap, to: e.lap, kind: e.kind });
  }
  return spans;
}

/** Gap to the leader at continuous race time ``t`` (laps), from per-lap gaps. */
export function gapAt(gaps: (number | null)[], t: number, firstLap = 1): number | null {
  const i = t - firstLap;
  const lo = Math.floor(i);
  const a = gaps[lo];
  const b = gaps[lo + 1];
  if (a == null) return b ?? null;
  if (b == null) return a;
  return a + (b - a) * (i - lo);
}

/** Point on the outline at a distance fraction in [0, 1), interpolated between samples. */
export function pointAt(points: [number, number][], fraction: number[], f: number): [number, number] {
  const x = ((f % 1) + 1) % 1;
  let lo = 0;
  let hi = fraction.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (fraction[mid] <= x) lo = mid;
    else hi = mid - 1;
  }
  const next = (lo + 1) % points.length;
  const span = (next === 0 ? 1 : fraction[next]) - fraction[lo];
  const w = span > 0 ? (x - fraction[lo]) / span : 0;
  return [
    points[lo][0] + (points[next][0] - points[lo][0]) * w,
    points[lo][1] + (points[next][1] - points[lo][1]) * w,
  ];
}

// ---- strategy editing -------------------------------------------------------------------

/** Index of the stint running on ``lap``. */
export function stintAt(stints: Stint[], lap: number): number {
  const i = stints.findIndex((s, k) => k === stints.length - 1 || lap <= s.end_lap);
  return i < 0 ? stints.length - 1 : i;
}

/** Move stop ``i`` (end of stint i) to ``lap``, keeping stints at least one lap long. */
export function moveStop(stints: Stint[], i: number, lap: number, totalLaps: number, lockedUntil: number): Stint[] {
  const prev = i === 0 ? 0 : stints[i - 1].end_lap;
  const next = i + 1 < stints.length - 1 ? stints[i + 1].end_lap : totalLaps;
  const clamped = Math.max(prev + 1, lockedUntil + 1, Math.min(lap, next - 1));
  return stints.map((s, k) => (k === i ? { ...s, end_lap: clamped } : s));
}

export function setCompound(stints: Stint[], i: number, compound: Compound): Stint[] {
  return stints.map((s, k) => (k === i ? { ...s, compound } : s));
}

/** Add a stop in the middle of the longest unlocked stint. */
export function addStop(stints: Stint[], totalLaps: number, lockedUntil: number): Stint[] {
  let best = -1;
  let bestLen = 0;
  stints.forEach((s, k) => {
    const start = k === 0 ? 1 : stints[k - 1].end_lap + 1;
    const end = k === stints.length - 1 ? totalLaps : s.end_lap;
    const from = Math.max(start, lockedUntil + 1);
    if (end - from + 1 > bestLen) {
      best = k;
      bestLen = end - from + 1;
    }
  });
  if (best < 0 || bestLen < 4) return stints;
  const start = Math.max(best === 0 ? 1 : stints[best - 1].end_lap + 1, lockedUntil + 1);
  const end = best === stints.length - 1 ? totalLaps : stints[best].end_lap;
  const split = Math.floor((start + end) / 2);
  const out = [...stints];
  const tail: Stint = { compound: other(stints[best].compound), end_lap: end, age_at_start: 0 };
  out.splice(best, 1, { ...stints[best], end_lap: split }, tail);
  return out;
}

export function removeStop(stints: Stint[], i: number, totalLaps: number): Stint[] {
  if (stints.length < 2 || i >= stints.length - 1) return stints;
  const out = [...stints];
  const merged = { ...out[i], end_lap: i + 1 === out.length - 1 ? totalLaps : out[i + 1].end_lap };
  out.splice(i, 2, merged);
  return out;
}

function other(c: Compound): Compound {
  return c === "HARD" ? "MEDIUM" : "HARD";
}

/** Problems that would make the API reject the strategy (shown before submitting). */
export function validate(stints: Stint[], totalLaps: number): string[] {
  const issues: string[] = [];
  if (new Set(stints.map((s) => s.compound)).size < 2) issues.push("Use at least two different compounds (dry-race rule).");
  stints.slice(0, -1).forEach((s, k) => {
    if (s.end_lap >= totalLaps) issues.push(`Stop ${k + 1} must be before the last lap.`);
  });
  return issues;
}

export function sameStrategy(a: Stint[], b: Stint[]): boolean {
  return (
    a.length === b.length &&
    a.every((s, k) => s.compound === b[k].compound && (k === a.length - 1 || s.end_lap === b[k].end_lap))
  );
}

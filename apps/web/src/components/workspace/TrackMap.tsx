"use client";

import { useMemo, useState } from "react";
import type { Track } from "@/lib/api";

export interface CarDot {
  driver: string;
  x: number;
  y: number;
  color: string | null;   // highlight colour, or null for the rest of the field
  position: number | null;
  gap: string;
}

/** 2D circuit outline with cars placed by their progress through the lap. */
export function TrackMap({ track, cars, simulated }: { track: Track; cars: CarDot[]; simulated: boolean }) {
  const [hover, setHover] = useState<string | null>(null);
  const path = useMemo(
    () => track.points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(4)},${(-y).toFixed(4)}`).join("") + "Z",
    [track.points],
  );
  const [sx, sy] = track.points[0];
  const ordered = [...cars].sort((a, b) => Number(a.color !== null) - Number(b.color !== null));
  const hovered = cars.find((c) => c.driver === hover);
  return (
    <div className="relative">
      <svg viewBox="-1.2 -1.2 2.4 2.4" className="mx-auto block aspect-square w-full max-w-[520px]"
        role="img" aria-label="Circuit map with car positions">
        <path d={path} fill="none" stroke="var(--line-strong)" strokeWidth={0.07} strokeLinejoin="round" />
        <path d={path} fill="none" stroke="var(--surface-2)" strokeWidth={0.045} strokeLinejoin="round" />
        <line x1={sx - 0.06} y1={-sy} x2={sx + 0.06} y2={-sy} stroke="var(--ink-2)" strokeWidth={0.012} />
        {ordered.map((c) => {
          const big = c.color !== null;
          return (
            <g key={c.driver} transform={`translate(${c.x},${-c.y})`}
              onMouseEnter={() => setHover(c.driver)} onMouseLeave={() => setHover(null)}>
              <circle r={0.09} fill="transparent" />
              <circle r={big ? 0.045 : 0.03} fill={c.color ?? "var(--ink-3)"}
                stroke={simulated ? "var(--ink)" : "var(--surface-1)"} strokeWidth={0.008}
                strokeDasharray={simulated ? "0.015 0.01" : undefined} />
              {(big || hover === c.driver) && (
                <text x={0.06} y={0.02} fontSize={0.075} fill="var(--ink)" fontWeight={600}>{c.driver}</text>
              )}
            </g>
          );
        })}
      </svg>
      {hovered && (
        <div className="pointer-events-none absolute left-2 top-2 rounded-md border border-line bg-surface-2 px-2 py-1 text-xs">
          <span className="font-semibold text-ink">{hovered.driver}</span>
          <span className="tabular ml-2 text-ink-2">P{hovered.position ?? "–"} · {hovered.gap}</span>
        </div>
      )}
      <p className="mt-1 text-center text-[11px] text-ink-3">
        Positions on track are interpolated from lap timing{simulated ? " (simulated median)" : ""}.
      </p>
    </div>
  );
}

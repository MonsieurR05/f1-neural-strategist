"use client";

import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { useState } from "react";
import { useWidth } from "@/lib/hooks";
import { eventSpans, fmtGap } from "@/lib/race";
import type { LapEvent } from "@/lib/api";

export interface Series {
  driver: string;
  color: string;
  gaps: (number | null)[];      // index 0 = lap 1
  dashed?: boolean;             // simulated
  label?: string;
}

const H = 300;
const M = { top: 12, right: 56, bottom: 28, left: 44 };

/** Gap to the leader by lap. Highlighted drivers in colour, the field in grey, SC/VSC shaded. */
export function GapChart({ field, series, totalLaps, events, currentLap, onLap }: {
  field: Record<string, (number | null)[]>;
  series: Series[];
  totalLaps: number;
  events: LapEvent[];
  currentLap: number;
  onLap?: (lap: number) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hoverLap, setHoverLap] = useState<number | null>(null);
  const highlighted = series.flatMap((s) => s.gaps).filter((g): g is number => g != null);
  const cap = Math.min(Math.max(30, ...highlighted) * 1.1, 120);
  const x = scaleLinear().domain([1, totalLaps]).range([M.left, width - M.right]);
  const y = scaleLinear().domain([0, cap]).range([M.top, H - M.bottom]);
  const path = (gaps: (number | null)[]) =>
    line<number | null>()
      .defined((g) => g != null)
      .x((_, i) => x(i + 1))
      .y((g) => y(Math.min(g as number, cap)))(gaps) ?? "";

  const lap = hoverLap ?? null;
  return (
    <div ref={ref} className="relative">
      <svg width={width} height={H} role="img" aria-label="Gap to the race leader by lap"
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const l = Math.round(x.invert(e.clientX - box.left));
          setHoverLap(l >= 1 && l <= totalLaps ? l : null);
        }}
        onMouseLeave={() => setHoverLap(null)}
        onClick={() => lap && onLap?.(lap)}>
        <defs>
          <clipPath id="plot"><rect x={M.left} y={M.top} width={width - M.left - M.right} height={H - M.top - M.bottom} /></clipPath>
        </defs>
        {eventSpans(events).map((s) => (
          <g key={`${s.kind}${s.from}`}>
            <rect x={x(s.from - 0.5)} width={x(s.to + 0.5) - x(s.from - 0.5)} y={M.top} height={H - M.top - M.bottom}
              fill="var(--sc)" opacity={s.kind === "VSC" ? 0.07 : 0.13} />
            <text x={x(s.from - 0.5) + 3} y={M.top + 11} fontSize={10} fill="var(--ink-2)">{s.kind}</text>
          </g>
        ))}
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={M.left - 6} y={y(t) + 3} fontSize={10} textAnchor="end" fill="var(--ink-3)" className="tabular">{t}s</text>
          </g>
        ))}
        {x.ticks(Math.min(10, totalLaps)).map((t) => (
          <text key={t} x={x(t)} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--ink-3)" className="tabular">{t}</text>
        ))}
        <text x={width - M.right} y={H - 8} fontSize={10} textAnchor="end" fill="var(--ink-3)" dx={50}>lap</text>
        <g clipPath="url(#plot)">
          {Object.entries(field).map(([d, gaps]) => (
            <path key={d} d={path(gaps)} fill="none" stroke="var(--ink-3)" strokeOpacity={0.35} strokeWidth={1} />
          ))}
          {series.map((s) => (
            <path key={s.driver + (s.dashed ? "-sim" : "")} d={path(s.gaps)} fill="none" stroke={s.color}
              strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} />
          ))}
        </g>
        {series.filter((s) => !s.dashed).map((s) => {
          const last = [...s.gaps].reverse().findIndex((g) => g != null);
          if (last < 0) return null;
          const i = s.gaps.length - 1 - last;
          return (
            <text key={`l-${s.driver}`} x={x(i + 1) + 6} y={y(Math.min(s.gaps[i] as number, cap)) + 3}
              fontSize={11} fontWeight={600} fill="var(--ink)">{s.driver}</text>
          );
        })}
        <line x1={x(currentLap)} x2={x(currentLap)} y1={M.top} y2={H - M.bottom} stroke="var(--ink-2)" strokeDasharray="2 3" />
        {lap && <line x1={x(lap)} x2={x(lap)} y1={M.top} y2={H - M.bottom} stroke="var(--ink)" strokeOpacity={0.5} />}
      </svg>
      {lap && (
        <div className="pointer-events-none absolute top-2 rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs shadow"
          style={{ left: Math.min(x(lap) + 10, width - 170) }}>
          <div className="mb-1 font-semibold text-ink">Lap {lap}</div>
          {series.map((s) => (
            <div key={s.driver + (s.dashed ? "s" : "")} className="tabular flex items-center gap-2 text-ink-2">
              <span className="inline-block h-0.5 w-3" style={{ background: s.color, opacity: s.dashed ? 0.7 : 1 }} />
              <span className="text-ink">{s.label ?? s.driver}</span>
              <span className="ml-auto">{fmtGap(s.gaps[lap - 1])}</span>
            </div>
          ))}
          <div className="mt-1 text-[10px] text-ink-3">Click to jump the replay here</div>
        </div>
      )}
    </div>
  );
}

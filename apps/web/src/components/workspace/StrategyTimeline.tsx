"use client";

import { useRef, useState } from "react";
import type { Stint } from "@/lib/api";
import { COMPOUND_COLOR, COMPOUNDS, moveStop, removeStop, setCompound } from "@/lib/race";

/** Stints as a horizontal bar over the race distance. Editable: drag stops, click a stint
 *  to change its compound, double-click a stop to remove it. Stints run before ``lockedUntil``
 *  (and the tyres on the car at that point) can't be changed. */
export function StrategyTimeline({ stints, totalLaps, onChange, lockedUntil = 0, compact = false }: {
  stints: Stint[];
  totalLaps: number;
  onChange?: (s: Stint[]) => void;
  lockedUntil?: number;
  compact?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const editable = !!onChange;
  const h = compact ? "h-4" : "h-8";
  const lapFromX = (clientX: number) => {
    const box = ref.current!.getBoundingClientRect();
    return Math.round(((clientX - box.left) / box.width) * totalLaps);
  };
  const starts = stints.map((_, k) => (k === 0 ? 1 : stints[k - 1].end_lap + 1));
  const ends = stints.map((s, k) => (k === stints.length - 1 ? totalLaps : s.end_lap));

  return (
    <div className="select-none">
      <div ref={ref} className={`relative ${h} w-full overflow-hidden rounded`}
        onPointerMove={(e) => drag !== null && onChange?.(moveStop(stints, drag, lapFromX(e.clientX), totalLaps, lockedUntil))}
        onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
        {stints.map((s, k) => {
          const left = ((starts[k] - 1) / totalLaps) * 100;
          const width = ((ends[k] - starts[k] + 1) / totalLaps) * 100;
          // a stint's compound can only change if it hasn't started yet
          const changeable = editable && starts[k] > lockedUntil;
          return (
            <button key={k} type="button" disabled={!changeable}
              title={`${s.compound.toLowerCase()} · laps ${starts[k]}–${ends[k]}${changeable ? " · click to change compound" : editable ? " · already fitted" : ""}`}
              onClick={() => changeable && onChange?.(setCompound(stints, k, COMPOUNDS[(COMPOUNDS.indexOf(s.compound) + 1) % 3]))}
              className={`absolute top-0 flex h-full items-center justify-center border-r-2 border-surface-1 text-[11px] font-bold ${changeable ? "cursor-pointer hover:brightness-110" : "cursor-default"}`}
              style={{ left: `${left}%`, width: `${width}%`, background: COMPOUND_COLOR[s.compound], color: "#121211" }}>
              {!compact && width > 5 && <span>{s.compound[0]}{width > 14 ? ` ${ends[k] - starts[k] + 1}` : ""}</span>}
            </button>
          );
        })}
        {editable && stints.slice(0, -1).map((s, k) => s.end_lap > lockedUntil && (
          <div key={`stop${k}`} role="slider" aria-label={`Stop ${k + 1}`} aria-valuenow={s.end_lap}
            aria-valuemin={1} aria-valuemax={totalLaps} tabIndex={0}
            onPointerDown={(e) => { (e.target as HTMLElement).setPointerCapture(e.pointerId); setDrag(k); }}
            onDoubleClick={() => onChange?.(removeStop(stints, k, totalLaps))}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight")
                onChange?.(moveStop(stints, k, s.end_lap + (e.key === "ArrowLeft" ? -1 : 1), totalLaps, lockedUntil));
              if (e.key === "Delete" || e.key === "Backspace") onChange?.(removeStop(stints, k, totalLaps));
            }}
            title={`Stop on lap ${s.end_lap}: drag or use ←/→; double-click to remove`}
            className="absolute top-0 z-10 h-full w-3 -translate-x-1/2 cursor-ew-resize focus:outline-none"
            style={{ left: `${(s.end_lap / totalLaps) * 100}%` }}>
            <div className={`mx-auto h-full w-1 rounded ${drag === k ? "bg-focus" : "bg-ink"}`} />
          </div>
        ))}
      </div>
      {!compact && (
        <div className="tabular mt-1 flex justify-between text-[10px] text-ink-3">
          <span>lap 1</span>
          {stints.slice(0, -1).map((s, k) => (
            <span key={k}>stop {k + 1}: lap {s.end_lap}</span>
          ))}
          <span>lap {totalLaps}</span>
        </div>
      )}
    </div>
  );
}

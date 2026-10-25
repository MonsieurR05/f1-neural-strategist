"use client";

import type { RaceDetail } from "@/lib/api";
import { StrategyTimeline } from "./StrategyTimeline";

/** What actually happened: road order at the line, grid, strategies. Also the table view of
 *  the replay charts. */
export function Classification({ race, focus, onFocus }: {
  race: RaceDetail;
  focus: string;
  onFocus: (d: string) => void;
}) {
  const L = race.race.total_laps;
  return (
    <table className="tabular w-full text-sm">
      <thead>
        <tr className="text-left text-[11px] uppercase tracking-wide text-ink-3">
          <th className="py-1 font-medium">Pos</th>
          <th className="py-1 font-medium">Driver</th>
          <th className="py-1 font-medium">Grid</th>
          <th className="w-1/2 py-1 font-medium">Strategy</th>
        </tr>
      </thead>
      <tbody>
        {race.drivers.map((d) => (
          <tr key={d.driver} onClick={() => onFocus(d.driver)}
            className={`cursor-pointer border-t border-line hover:bg-surface-2 ${d.driver === focus ? "bg-surface-2" : ""}`}>
            <td className="py-1.5 text-ink-2">{d.finished ? d.finish_position : "DNF"}</td>
            <td className="py-1.5">
              <div className="font-semibold text-ink">{d.driver}</div>
              <div className="text-[11px] text-ink-3">{d.team}</div>
            </td>
            <td className="py-1.5 text-ink-2">{d.grid_position ?? "–"}</td>
            <td className="py-1.5">
              <StrategyTimeline stints={d.strategy} totalLaps={L} compact />
              <div className="mt-0.5 text-[11px] text-ink-3">{d.strategy_label}</div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

"use client";

import { useState } from "react";
import { api, ApiError, type EventsMode, type FinishRow, type RaceDetail, type SimulateResponse, type Stint } from "@/lib/api";
import { useElapsed } from "@/lib/hooks";
import { addStop, describe, fmtDelta, pct, sameStrategy, validate } from "@/lib/race";
import { Button, ErrorNote, ProvenanceBadge } from "@/components/ui";
import { StrategyTimeline } from "./StrategyTimeline";

export interface SimPair {
  driver: string;
  stints: Stint[];
  baseline: SimulateResponse;
  edited: SimulateResponse;
}

const RUNS = 300;

export function StrategyBuilder({ race, driver, draft, onDraft, result, onResult }: {
  race: RaceDetail;
  driver: string;
  draft: Stint[];
  onDraft: (s: Stint[]) => void;
  result: SimPair | null;
  onResult: (r: SimPair) => void;
}) {
  const L = race.race.total_laps;
  const history = race.drivers.find((d) => d.driver === driver)!;
  const [events, setEvents] = useState<EventsMode>("actual");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const elapsed = useElapsed(busy);
  const issues = validate(draft, L);
  const unchanged = sameStrategy(draft, history.strategy);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const common = { race_id: race.race.race_id, runs: RUNS, seed: 0, events };
      // same seed for both: identical random draws, so the difference is the strategy
      const [baseline, edited] = await Promise.all([
        api.simulate(common),
        api.simulate(unchanged ? common : { ...common, strategies: { [driver]: draft } }),
      ]);
      onResult({ driver, stints: draft, baseline, edited });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Simulation failed.");
    } finally {
      setBusy(false);
    }
  }

  const mine = result?.driver === driver ? result : null;
  return (
    <div className="space-y-5">
      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-3">As raced</h3>
          <span className="tabular text-xs text-ink-2">{history.strategy_label} · finished P{history.finish_position}</span>
        </div>
        <StrategyTimeline stints={history.strategy} totalLaps={L} compact />
      </div>

      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-3">Your strategy</h3>
          <span className="tabular text-xs text-ink-2">{describe(draft)}</span>
        </div>
        <StrategyTimeline stints={draft} totalLaps={L} onChange={onDraft} lockedUntil={1} />
        <p className="mt-1.5 text-[11px] text-ink-3">
          Drag a stop (or focus it and use ←/→), click a stint to change its tyres, double-click a stop to remove it.
          The starting tyres were fitted before lap 1 and can&apos;t change.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <Button variant="ghost" onClick={() => onDraft(addStop(draft, L, 1))}>Add stop</Button>
          <Button variant="ghost" disabled={unchanged} onClick={() => onDraft(history.strategy)}>Reset to as raced</Button>
        </div>
        {issues.length > 0 && <ul className="mt-2 list-disc pl-5 text-xs text-warn">{issues.map((i) => <li key={i}>{i}</li>)}</ul>}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <fieldset className="flex overflow-hidden rounded-md border border-line-strong text-xs">
          <legend className="sr-only">Safety cars</legend>
          {(["actual", "sampled"] as const).map((m) => (
            <label key={m} className={`cursor-pointer px-2.5 py-1.5 ${events === m ? "bg-surface-2 text-ink" : "text-ink-3"}`}>
              <input type="radio" className="sr-only" name="events" checked={events === m} onChange={() => setEvents(m)} />
              {m === "actual" ? "Real safety cars" : "Random safety cars"}
            </label>
          ))}
        </fieldset>
        <Button onClick={run} disabled={busy || issues.length > 0}>
          {busy ? `Simulating… ${elapsed.toFixed(0)} s` : `Simulate (${RUNS} races each)`}
        </Button>
      </div>
      {error && <ErrorNote message={error} />}
      {mine && <Comparison pair={mine} actualPos={history.finish_position} />}
    </div>
  );
}

function Comparison({ pair, actualPos }: { pair: SimPair; actualPos: number }) {
  const base = pair.baseline.finish.find((r) => r.driver === pair.driver)!;
  const mine = pair.edited.finish.find((r) => r.driver === pair.driver)!;
  const rows: [string, (r: FinishRow) => string, number][] = [
    ["Expected position", (r) => r.mean_pos.toFixed(2), mine.mean_pos - base.mean_pos],
    ["80% range", (r) => `P${Math.round(r.p10_pos)}–P${Math.round(r.p90_pos)}`, NaN],
    ["Win", (r) => pct(r.p_win), mine.p_win - base.p_win],
    ["Podium", (r) => pct(r.p_podium), mine.p_podium - base.p_podium],
    ["Points", (r) => pct(r.p_points), mine.p_points - base.p_points],
    ["Time lost in traffic", (r) => `${r.mean_held_s.toFixed(1)} s`, mine.mean_held_s - base.mean_held_s],
  ];
  const better = mine.mean_pos < base.mean_pos - 0.05;
  const worse = mine.mean_pos > base.mean_pos + 0.05;
  return (
    <div className="rounded-md border border-dashed border-focus/60 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">{pair.driver}: as raced vs yours</h3>
        <ProvenanceBadge kind="simulated" />
      </div>
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-ink-3">
            <th className="py-1 font-medium" />
            <th className="py-1 font-medium">As raced</th>
            <th className="py-1 font-medium">Yours</th>
            <th className="py-1 text-right font-medium">Change</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, f, d]) => (
            <tr key={label} className="border-t border-line">
              <td className="py-1.5 text-ink-2">{label}</td>
              <td className="py-1.5 text-ink">{f(base)}</td>
              <td className="py-1.5 text-ink">{f(mine)}</td>
              <td className="py-1.5 text-right text-ink-2">
                {Number.isNaN(d) ? "" : label === "Expected position" || label === "Time lost in traffic"
                  ? fmtDelta(d, label === "Expected position" ? "" : " s") : fmtDelta(d * 100, " pts")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-2">
        {better ? "Your strategy finishes ahead on average." : worse ? "Your strategy finishes behind on average." : "No meaningful difference on average."}{" "}
        Both columns are simulated with the same random draws, so the difference comes from the strategy. The real result was P{actualPos}.
      </p>
      {pair.edited.notes.map((n) => <p key={n} className="mt-1 text-[11px] text-ink-3">{n}</p>)}
    </div>
  );
}

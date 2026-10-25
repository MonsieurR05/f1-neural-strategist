"use client";

import { useState } from "react";
import { api, ApiError, type EventsMode, type OptimiseResponse, type RaceDetail, type Stint, type StrategyResult } from "@/lib/api";
import { useElapsed } from "@/lib/hooks";
import { fmtDelta, pct } from "@/lib/race";
import { Button, ErrorNote, ProvenanceBadge } from "@/components/ui";
import { StrategyTimeline } from "./StrategyTimeline";

export function OptimiserPanel({ race, driver, onLoad }: {
  race: RaceDetail;
  driver: string;
  onLoad: (stints: Stint[]) => void;
}) {
  const L = race.race.total_laps;
  const [startLap, setStartLap] = useState(1);
  const [events, setEvents] = useState<EventsMode>("sampled");
  const [maxStops, setMaxStops] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OptimiseResponse | null>(null);
  const elapsed = useElapsed(busy);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.optimise({ race_id: race.race.race_id, driver, start_lap: startLap, events, max_stops: maxStops, top: 8 }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Optimisation failed.");
    } finally {
      setBusy(false);
    }
  }

  const shown = result?.driver === driver ? result : null;
  const best = shown?.strategies[0];
  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-2">
        Enumerates every legal strategy for <span className="font-semibold text-ink">{driver}</span> from the chosen lap,
        screens them all in the simulator, then re-runs the best with 300 races each and compares them with what was raced.
      </p>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <label className="space-y-1">
          <span className="block text-[11px] uppercase tracking-wide text-ink-3">From lap</span>
          <input type="number" min={1} max={L - 8} value={startLap}
            onChange={(e) => setStartLap(Math.max(1, Math.min(L - 8, Number(e.target.value) || 1)))}
            className="tabular w-full rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-ink" />
        </label>
        <label className="space-y-1">
          <span className="block text-[11px] uppercase tracking-wide text-ink-3">Max stops</span>
          <select value={maxStops} onChange={(e) => setMaxStops(Number(e.target.value))}
            className="w-full rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-ink">
            {[1, 2, 3].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-[11px] uppercase tracking-wide text-ink-3">Safety cars</span>
          <select value={events} onChange={(e) => setEvents(e.target.value as EventsMode)}
            className="w-full rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-ink">
            <option value="sampled">Unknown (random)</option>
            <option value="actual">Hindsight (real)</option>
          </select>
        </label>
      </div>
      <Button onClick={run} disabled={busy}>
        {busy ? `Optimising… ${elapsed.toFixed(0)} s (usually under a minute)` : "Find the best strategy"}
      </Button>
      {error && <ErrorNote message={error} />}

      {shown && best && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-ink-3">
            <span className="tabular">{shown.candidates_evaluated} strategies screened · as raced: {shown.as_raced}</span>
            <ProvenanceBadge kind="simulated" />
          </div>
          <ol className="space-y-2">
            {shown.strategies.map((s) => (
              <Candidate key={s.strategy} s={s} best={best} totalLaps={L} lockedUntil={startLap}
                onLoad={() => onLoad(s.stints)} />
            ))}
          </ol>
          {shown.notes.map((n) => <p key={n} className="text-xs text-ink-2">{n}</p>)}
          <details className="rounded-md border border-line p-2.5 text-xs text-ink-2">
            <summary className="cursor-pointer font-medium text-ink">Model caveats</summary>
            <ul className="mt-2 list-disc space-y-1 pl-4">{shown.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          </details>
        </div>
      )}
    </div>
  );
}

function Candidate({ s, best, totalLaps, lockedUntil, onLoad }: {
  s: StrategyResult; best: StrategyResult; totalLaps: number; lockedUntil: number; onLoad: () => void;
}) {
  const budget: [string, number][] = [
    ["Pits", s.pit_stops_s - best.pit_stops_s],
    ["Tyre wear", s.tyre_wear_s - best.tyre_wear_s],
    ["Compound pace", s.compound_pace_s - best.compound_pace_s],
    ["Traffic", s.traffic_s - best.traffic_s],
  ];
  return (
    <li className={`rounded-md border p-3 ${s.as_raced ? "border-ink-3" : "border-line"} ${s.rank === 1 ? "bg-surface-2" : ""}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="tabular text-xs text-ink-3">#{s.rank}</span>
          <span className="tabular text-sm font-semibold text-ink">{s.strategy}</span>
          {s.as_raced && <span className="rounded bg-surface-2 px-1.5 text-[10px] text-ink-2">as raced</span>}
          {s.extrapolated && <span title="Tyres run longer than ever observed at this circuit" className="rounded bg-warn/20 px-1.5 text-[10px] text-warn">extrapolated</span>}
        </div>
        {!s.as_raced && <button type="button" onClick={onLoad} className="text-xs text-focus hover:underline">Load into builder</button>}
      </div>
      <StrategyTimeline stints={s.stints} totalLaps={totalLaps} lockedUntil={lockedUntil} compact />
      <div className="tabular mt-2 grid grid-cols-4 gap-2 text-xs">
        <div><div className="text-ink-3">Exp. pos</div><div className="text-ink">{s.expected_pos.toFixed(2)}</div></div>
        <div><div className="text-ink-3">Podium</div><div className="text-ink">{pct(s.p_podium)}</div></div>
        <div><div className="text-ink-3">vs as raced</div><div className="text-ink">{fmtDelta(s.time_vs_as_raced_s)}</div></div>
        <div><div className="text-ink-3">Beats it</div><div className="text-ink">{s.p_ahead_of_as_raced == null ? "–" : pct(s.p_ahead_of_as_raced)}</div></div>
      </div>
      {s.rank > 1 && (
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink-3">
          {budget.filter(([, d]) => Math.abs(d) >= 0.5).map(([k, d]) => (
            <span key={k} className="tabular">{k} <span className={d > 0 ? "text-warn" : "text-ink-2"}>{fmtDelta(d)}</span></span>
          ))}
        </div>
      )}
      <p className="mt-1.5 text-xs text-ink-2">{s.explanation}</p>
    </li>
  );
}

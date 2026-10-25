"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RaceDetail, Stint, Track } from "@/lib/api";
import { fmtGap, gapAt, HIGHLIGHT, lapStatus, pointAt } from "@/lib/race";
import { Panel, ProvenanceBadge } from "@/components/ui";
import { Classification } from "./Classification";
import { GapChart, type Series } from "./GapChart";
import { OptimiserPanel } from "./OptimiserPanel";
import { StrategyBuilder, type SimPair } from "./StrategyBuilder";
import { TrackMap, type CarDot } from "./TrackMap";

type Tab = "strategy" | "optimise" | "race";

export function RaceWorkspace({ race, track }: { race: RaceDetail; track: Track | null }) {
  const L = race.race.total_laps;
  const order = race.drivers.map((d) => d.driver);
  const [focus, setFocus] = useState(order[0]);
  const [rival, setRival] = useState<string | null>(order[1] ?? null);
  const [tab, setTab] = useState<Tab>("strategy");
  const [drafts, setDrafts] = useState<Record<string, Stint[]>>({});
  const [sim, setSim] = useState<SimPair | null>(null);
  const [view, setView] = useState<"historical" | "simulated">("historical");
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // playback: speed is laps per second
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = (now: number) => {
      const dt = last.current == null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setT((prev) => {
        const next = Math.min(L, prev + dt * speed);
        if (next >= L) setPlaying(false);
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      last.current = null;
    };
  }, [playing, speed, L]);

  const simulated = view === "simulated" && sim !== null;
  const gaps: Record<string, (number | null)[]> = useMemo(() => {
    if (!simulated) return race.gap_to_leader_by_lap;
    return Object.fromEntries(Object.entries(sim!.edited.laps).map(([d, b]) => [d, b.median_gap_s]));
  }, [simulated, sim, race.gap_to_leader_by_lap]);
  const positions = (d: string, lap: number): number | null =>
    simulated ? (sim!.edited.laps[d]?.median_pos[lap - 1] ?? null) : (race.positions_by_lap[d]?.[lap - 1] ?? null);

  const highlight = [focus, rival].filter((d): d is string => !!d);
  const colorOf = (d: string) => {
    const i = highlight.indexOf(d);
    return i >= 0 ? HIGHLIGHT[i] : null;
  };
  const cars: CarDot[] = useMemo(() => {
    if (!track) return [];
    const phase = t - Math.floor(t);
    return order.flatMap((d) => {
      const g = gapAt(gaps[d] ?? [], t);
      if (g == null) return [];
      const [x, y] = pointAt(track.points, track.fraction, phase - g / race.nominal_lap_s);
      return [{ driver: d, x, y, color: colorOf(d), position: positions(d, Math.max(1, Math.floor(t))), gap: fmtGap(g) }];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t, gaps, track, focus, rival]);

  const status = lapStatus(race.events, L);
  const currentLap = Math.min(L, Math.floor(t) + (t < L ? 1 : 0));
  const series: Series[] = highlight.map((d, i) => ({ driver: d, color: HIGHLIGHT[i], gaps: race.gap_to_leader_by_lap[d] ?? [] }));
  if (sim && sim.driver === focus) {
    series.push({ driver: focus, color: HIGHLIGHT[0], dashed: true, label: `${focus} (yours, simulated)`,
      gaps: sim.edited.laps[focus]?.median_gap_s ?? [] });
  }
  const draft = drafts[focus] ?? race.drivers.find((d) => d.driver === focus)!.strategy;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link href="/" className="text-xs text-ink-3 hover:text-focus">← All races</Link>
          <h1 className="text-xl font-semibold text-ink">{race.race.event_name} {race.race.year}</h1>
          <p className="tabular text-sm text-ink-2">{L} laps · {race.race.circuit}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <DriverSelect label="Focus" value={focus} options={order} color={HIGHLIGHT[0]}
            onChange={(d) => { setFocus(d); if (d === rival) setRival(null); }} />
          <DriverSelect label="Compare" value={rival ?? ""} options={order.filter((d) => d !== focus)} color={HIGHLIGHT[1]}
            onChange={(d) => setRival(d || null)} allowNone />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="space-y-4">
          <Panel title="Race replay" actions={
            <div className="flex items-center gap-2">
              {sim && (
                <div className="flex overflow-hidden rounded-md border border-line-strong text-xs">
                  {(["historical", "simulated"] as const).map((v) => (
                    <button key={v} type="button" onClick={() => setView(v)}
                      className={`px-2 py-1 ${view === v ? "bg-surface-2 text-ink" : "text-ink-3"}`}>
                      {v === "historical" ? "Historical" : `Simulated (${sim.driver} yours)`}
                    </button>
                  ))}
                </div>
              )}
              <ProvenanceBadge kind={simulated ? "simulated" : "historical"} />
            </div>
          }>
            {track ? <TrackMap track={track} cars={cars} simulated={simulated} />
              : <p className="text-sm text-ink-3">No circuit outline for {race.race.circuit} yet; the gap chart below shows the race.</p>}
            <div className="mt-3 flex items-center gap-3">
              <button type="button" onClick={() => { if (t >= L) setT(1); setPlaying((p) => !p); }}
                aria-label={playing ? "Pause" : "Play"}
                className="h-8 w-8 rounded-full bg-focus text-sm font-bold text-white">{playing ? "❚❚" : "▶"}</button>
              <input type="range" min={1} max={L} step={0.01} value={t} aria-label="Race lap"
                onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} className="flex-1 accent-[var(--focus)]" />
              <span className="tabular w-20 text-right text-sm text-ink">Lap {currentLap}/{L}</span>
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed"
                className="rounded border border-line-strong bg-surface-2 px-1 py-0.5 text-xs text-ink-2">
                {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s} lap/s</option>)}
              </select>
            </div>
            {status[currentLap] && (
              <div className="mt-2 inline-block rounded bg-medium/20 px-2 py-0.5 text-xs font-semibold text-medium">
                {status[currentLap] === "VSC" ? "Virtual safety car" : status[currentLap] === "RED" ? "Red flag" : "Safety car"}
              </div>
            )}
          </Panel>
          <Panel title="Gap to the leader" actions={<span className="text-[11px] text-ink-3">dashed = simulated median</span>}>
            <GapChart field={race.gap_to_leader_by_lap} series={series} totalLaps={L} events={race.events}
              currentLap={Math.max(1, t)} onLap={(lap) => { setPlaying(false); setT(lap); }} />
          </Panel>
        </div>

        <Panel title={
          <div role="tablist" className="flex gap-1">
            {([["strategy", "Strategy builder"], ["optimise", "Optimiser"], ["race", "Classification"]] as const).map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} type="button" onClick={() => setTab(k)}
                className={`rounded px-2 py-1 text-sm ${tab === k ? "bg-surface-2 text-ink" : "font-normal text-ink-3 hover:text-ink"}`}>
                {label}
              </button>
            ))}
          </div>
        }>
          {tab === "strategy" && (
            <StrategyBuilder race={race} driver={focus} draft={draft}
              onDraft={(s) => setDrafts((all) => ({ ...all, [focus]: s }))}
              result={sim} onResult={(r) => { setSim(r); setView("simulated"); }} />
          )}
          {tab === "optimise" && (
            <OptimiserPanel race={race} driver={focus}
              onLoad={(stints) => { setDrafts((all) => ({ ...all, [focus]: stints })); setTab("strategy"); }} />
          )}
          {tab === "race" && <Classification race={race} focus={focus} onFocus={setFocus} />}
        </Panel>
      </div>
    </div>
  );
}

function DriverSelect({ label, value, options, color, onChange, allowNone }: {
  label: string; value: string; options: string[]; color: string; onChange: (d: string) => void; allowNone?: boolean;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      <span className="text-xs uppercase tracking-wide text-ink-3">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-sm text-ink">
        {allowNone && <option value="">—</option>}
        {options.map((d) => <option key={d} value={d}>{d}</option>)}
      </select>
    </label>
  );
}

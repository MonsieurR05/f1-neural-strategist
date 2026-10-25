import Link from "next/link";
import { Suspense } from "react";
import { api, ApiError, type RaceSummary } from "@/lib/api";
import { ErrorNote } from "@/components/ui";

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="max-w-3xl space-y-3">
        <h1 className="text-2xl font-semibold text-ink sm:text-3xl">Race strategy, simulated lap by lap</h1>
        <p className="text-ink-2">
          Pick a race, replay what happened, then change a driver&apos;s pit stops and tyres and see how
          the race could have played out. A neural lap-time model trained on 2021–2023 drives a Monte
          Carlo simulator with traffic, overtaking, pit stops and safety cars; an optimiser ranks every
          legal strategy and explains the trade-offs.
        </p>
        <p className="text-sm text-ink-3">
          Bahrain is the calibrated MVP circuit. Simulated results are counterfactual estimates and are
          always marked as such.
        </p>
      </section>
      <Suspense fallback={<p className="text-ink-3">Loading races…</p>}>
        <RaceList />
      </Suspense>
    </div>
  );
}

async function loadRaces(): Promise<RaceSummary[] | string> {
  try {
    return await api.races();
  } catch (e) {
    return e instanceof ApiError ? e.message : "Couldn't load races.";
  }
}

async function RaceList() {
  const races = await loadRaces();
  if (typeof races === "string") return <ErrorNote message={races} />;
  const bahrain = races.filter((r) => r.circuit === "Sakhir").reverse();
  const seasons = [...new Set(races.map((r) => r.year))].sort((a, b) => b - a);
  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink-3">Bahrain Grand Prix</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {bahrain.map((r) => (
            <Link key={r.race_id} href={`/race/${r.race_id}`}
              className="group rounded-lg border border-line bg-surface-1 p-4 transition hover:border-focus">
              <div className="tabular text-2xl font-semibold text-ink">{r.year}</div>
              <div className="text-sm text-ink-2">{r.total_laps} laps</div>
              <div className="mt-3 text-xs text-ink-3 group-hover:text-focus">
                {r.year === 2024 ? "Held-out validation race →" : r.year === 2025 ? "Test season →" : "Training season →"}
              </div>
            </Link>
          ))}
        </div>
      </section>
      {seasons.map((year) => (
        <section key={year}>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-3">{year} season</h2>
          <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-4">
            {races.filter((r) => r.year === year).map((r) => (
              <li key={r.race_id} className="flex items-center justify-between gap-2 text-sm">
                {r.simulatable ? (
                  <Link href={`/race/${r.race_id}`} className="text-ink-2 hover:text-focus">
                    {r.event_name.replace(" Grand Prix", " GP")}
                  </Link>
                ) : (
                  <span className="text-ink-3" title={r.wet ? "Wet race: not simulated" : "Unrepresentative qualifying: not simulated"}>
                    {r.event_name.replace(" Grand Prix", " GP")}
                  </span>
                )}
                {!r.simulatable && <span className="text-[11px] text-ink-3">{r.wet ? "wet" : "n/a"}</span>}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

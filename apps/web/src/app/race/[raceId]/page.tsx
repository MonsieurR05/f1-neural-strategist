import Link from "next/link";
import { Suspense } from "react";
import { api, ApiError, type RaceDetail, type Track } from "@/lib/api";
import { ErrorNote } from "@/components/ui";
import { RaceWorkspace } from "@/components/workspace/RaceWorkspace";

export default function RacePage(props: PageProps<"/race/[raceId]">) {
  return (
    <Suspense fallback={<p className="text-ink-3">Loading race…</p>}>
      <RaceLoader params={props.params} />
    </Suspense>
  );
}

async function load(raceId: string): Promise<{ race: RaceDetail; track: Track | null } | { error: string }> {
  try {
    const race = await api.race(raceId);
    if (!race.race.simulatable) {
      return { error: `${race.race.event_name} ${race.race.year} was a wet race or had unrepresentative qualifying, so it can't be simulated.` };
    }
    return { race, track: await api.track(race.race.circuit) };
  } catch (e) {
    return { error: e instanceof ApiError ? e.message : "Couldn't load the race." };
  }
}

async function RaceLoader({ params }: { params: Promise<{ raceId: string }> }) {
  const { raceId } = await params;
  const data = await load(raceId);
  if ("error" in data) {
    return (
      <div className="space-y-3">
        <ErrorNote message={data.error} />
        <Link href="/" className="text-sm text-focus">← All races</Link>
      </div>
    );
  }
  return <RaceWorkspace race={data.race} track={data.track} />;
}

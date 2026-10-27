// Typed client for the FastAPI service (services/api). Types mirror app/schemas.py.

export type Compound = "SOFT" | "MEDIUM" | "HARD";
export type Provenance = "historical" | "simulated";
export type EventsMode = "actual" | "sampled";

export interface Stint {
  compound: Compound;
  end_lap: number;
  age_at_start: number;
}

export interface RaceSummary {
  race_id: string;
  year: number;
  round: number;
  event_name: string;
  circuit: string;
  total_laps: number;
  wet: boolean;
  simulatable: boolean;
}

export interface DriverHistory {
  driver: string;
  team: string;
  grid_position: number | null;
  finish_position: number;
  finished: boolean;
  strategy: Stint[];
  strategy_label: string;
}

export interface LapEvent {
  lap: number;
  kind: "SC" | "VSC" | "RED";
}

export interface RaceDetail {
  provenance: Provenance;
  race: RaceSummary;
  drivers: DriverHistory[];
  events: LapEvent[];
  positions_by_lap: Record<string, (number | null)[]>;
  gap_to_leader_by_lap: Record<string, (number | null)[]>;
  leader_lap_time_s: (number | null)[];
  nominal_lap_s: number;
}

export interface Track {
  circuit: string;
  source: string;
  points: [number, number][];
  fraction: number[];
}

export interface FinishRow {
  driver: string;
  mean_pos: number;
  median_pos: number;
  p10_pos: number;
  p90_pos: number;
  p_win: number;
  p_podium: number;
  p_points: number;
  median_gap_s: number | null;
  mean_held_s: number;
}

export interface LapBands {
  median_pos: (number | null)[];
  p10_pos: (number | null)[];
  p90_pos: (number | null)[];
  median_gap_s: (number | null)[];
}

export interface SimulateResponse {
  provenance: Provenance;
  race_id: string;
  start_lap: number;
  runs: number;
  events: EventsMode;
  strategies: Record<string, string>;
  finish: FinishRow[];
  laps: Record<string, LapBands>;
  notes: string[];
}

export interface StrategyResult {
  rank: number;
  strategy: string;
  stints: Stint[];
  stops: number;
  as_raced: boolean;
  expected_pos: number;
  p10_pos: number;
  p90_pos: number;
  p_win: number;
  p_podium: number;
  p_points: number;
  time_vs_as_raced_s: number | null;
  p_ahead_of_as_raced: number | null;
  pit_stops_s: number;
  tyre_wear_s: number;
  compound_pace_s: number;
  traffic_s: number;
  extrapolated: boolean;
  explanation: string;
}

export interface OptimiseResponse {
  provenance: Provenance;
  race_id: string;
  driver: string;
  start_lap: number;
  events: EventsMode;
  candidates_evaluated: number;
  as_raced: string;
  strategies: StrategyResult[];
  notes: string[];
  caveats: string[];
}

// Server components reach the API directly; the browser uses the public URL. Trimmed because
// piping the value into `vercel env add` from PowerShell stores a trailing CRLF.
const SERVER_BASE = (process.env.API_URL ?? "http://127.0.0.1:8000").trim();
const BROWSER_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").trim();
const base = () => (typeof window === "undefined" ? SERVER_BASE : BROWSER_BASE);

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${base()}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, `Can't reach the strategy API at ${base()}. Is it running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  races: () => request<RaceSummary[]>("/races"),
  race: (raceId: string) => request<RaceDetail>(`/races/${encodeURIComponent(raceId)}`),
  track: async (circuit: string): Promise<Track | null> => {
    try {
      return await request<Track>(`/circuits/${encodeURIComponent(circuit)}/track`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  },
  simulate: (body: {
    race_id: string;
    start_lap?: number;
    runs?: number;
    seed?: number;
    events?: EventsMode;
    strategies?: Record<string, Stint[]>;
  }) => request<SimulateResponse>("/simulate", { method: "POST", body: JSON.stringify(body) }),
  optimise: (body: {
    race_id: string;
    driver: string;
    start_lap?: number;
    events?: EventsMode;
    max_stops?: number;
    top?: number;
  }) => request<OptimiseResponse>("/optimise", { method: "POST", body: JSON.stringify(body) }),
};

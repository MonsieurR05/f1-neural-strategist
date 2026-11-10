// Simulate in the browser (Rust → WebAssembly, in public/sim/worker.js) when possible, else
// via the API.
//
// The browser simulator is a port of the API's, held to it by parity tests in
// crates/strategist-sim. It has no CPU cap, quota or cold start, so it runs more races per
// simulation than the free API host allows. Random draws come from a different generator, so
// results agree with the API's statistically, not run for run.

import { api, type EventsMode, type SimulateResponse, type Stint } from "./api";

export const BROWSER_RUNS = 300;
const API_RUNS = 300; // the API caps this at its profile's max_runs (100 on Render)

/** Messages to and from public/sim/worker.js, which runs the WebAssembly module. */
interface WorkerRequest {
  id: number;
  race_id: string;
  runs: number;
  seed: number;
  events: EventsMode;
  strategies: Record<string, Stint[]>;
}

type WorkerReply = { id: number; ok: true; json: string } | { id: number; ok: false; error: string };

export interface SimulateRequest {
  race_id: string;
  seed: number;
  events: EventsMode;
  strategies?: Record<string, Stint[]>;
}

let worker: Worker | null = null;
let broken = false; // the browser simulator failed once: use the API from then on
let nextId = 0;
const pending = new Map<number, (reply: WorkerReply) => void>();
const cache = new Map<string, Promise<SimulateResponse>>(); // unchanged strategies, per race

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker("/sim/worker.js");
    worker.onmessage = (e: MessageEvent<WorkerReply>) => {
      pending.get(e.data.id)?.(e.data);
      pending.delete(e.data.id);
    };
    worker.onerror = (e) => {
      for (const resolve of pending.values()) resolve({ id: -1, ok: false, error: e.message || "worker failed" });
      pending.clear();
    };
  }
  return worker;
}

async function inBrowser(req: SimulateRequest): Promise<SimulateResponse> {
  const strategies = req.strategies ?? {};
  const message: WorkerRequest = { id: nextId++, race_id: req.race_id, runs: BROWSER_RUNS, seed: req.seed,
                                   events: req.events, strategies };
  const reply = await new Promise<WorkerReply>((resolve) => {
    pending.set(message.id, resolve);
    getWorker().postMessage(message);
  });
  if (!reply.ok) throw new Error(reply.error);
  const sim = JSON.parse(reply.json) as Omit<SimulateResponse, "provenance" | "race_id" | "events" | "notes">;
  const notes = [`Simulated in your browser: ${sim.runs} races (Rust compiled to WebAssembly).`];
  if (req.events === "actual" && Object.keys(strategies).length > 0) {
    notes.push("Real SC/VSC laps are kept even though changed strategies might have changed them.");
  }
  return { ...sim, provenance: "simulated", race_id: req.race_id, events: req.events, notes };
}

function viaApi(req: SimulateRequest): Promise<SimulateResponse> {
  return api.simulate({ ...req, runs: API_RUNS });
}

export async function simulate(req: SimulateRequest): Promise<SimulateResponse> {
  if (broken || typeof Worker === "undefined" || typeof WebAssembly === "undefined") return viaApi(req);
  const unchanged = !req.strategies || Object.keys(req.strategies).length === 0;
  const key = `${req.race_id}|${req.events}|${req.seed}`;
  if (unchanged && cache.has(key)) return cache.get(key)!;
  const run = inBrowser(req).catch((e: unknown) => {
    console.warn("Browser simulator unavailable, using the API:", e);
    broken = true;
    cache.delete(key);
    return viaApi(req);
  });
  if (unchanged) cache.set(key, run);
  return run;
}

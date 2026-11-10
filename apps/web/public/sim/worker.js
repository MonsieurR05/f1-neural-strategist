// Runs the Rust race simulator (WebAssembly, crates/strategist-sim) off the main thread.
// Plain JavaScript served as-is: Turbopack copies a TypeScript worker without compiling it.
// Next to this file: the module, the lap-time model, simulator parameters and one replay setup
// per race, written by `strategist.sim.export --web` and `crates/strategist-sim/scripts/build_web.mjs`.
// Messages in: {id, race_id, runs, seed, events, strategies}; out: {id, ok, json | error}.
// Types: WorkerRequest and WorkerReply in src/lib/simulate.ts.

let engine = null;
const setups = new Map();

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.text();
}

function pass(sim, text) {
  const bytes = new TextEncoder().encode(text);
  const ptr = sim.sim_alloc(bytes.length);
  new Uint8Array(sim.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function output(sim, ok) {
  const text = new TextDecoder().decode(new Uint8Array(sim.memory.buffer, sim.sim_output_ptr(), sim.sim_output_len()));
  if (!ok) throw new Error(text);
  return text;
}

async function load() {
  const [wasm, model, params] = await Promise.all([
    fetch("strategist_sim.wasm").then((r) => {
      if (!r.ok) throw new Error(`strategist_sim.wasm: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
    fetchText("model.json"),
    fetchText("params.json"),
  ]);
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const sim = instance.exports;
  output(sim, sim.sim_load(...pass(sim, model), ...pass(sim, params)));
  return sim;
}

async function simulate(req) {
  engine ??= load();
  const sim = await engine.catch((e) => {
    engine = null; // let a later click retry
    throw e;
  });
  if (!setups.has(req.race_id)) setups.set(req.race_id, fetchText(`races/${req.race_id}.json`));
  const setup = JSON.parse(await setups.get(req.race_id));
  if (req.events === "sampled") { // drawn from history, as the API does
    setup.events = null;
    setup.retirements = null;
  }
  for (const [driver, stints] of Object.entries(req.strategies)) {
    const car = setup.cars.find((c) => c.driver === driver);
    if (!car) throw new Error(`${driver} is not in this race`);
    car.strategy = stints;
  }
  return output(sim, sim.sim_simulate(...pass(sim, JSON.stringify({ setup, runs: req.runs, seed: req.seed }))));
}

self.onmessage = async (e) => {
  const reply = await simulate(e.data)
    .then((json) => ({ id: e.data.id, ok: true, json }))
    .catch((err) => ({ id: e.data.id, ok: false, error: String(err) }));
  self.postMessage(reply);
};

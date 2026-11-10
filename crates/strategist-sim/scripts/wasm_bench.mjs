// Load the WebAssembly build in Node, simulate a fixture race and time it. From this crate:
//
//   cargo build --release --target wasm32-unknown-unknown
//   node scripts/wasm_bench.mjs
//
// Needs the fixtures from `python -m strategist.sim.export`. The same four calls (alloc, load,
// simulate, read the output) are all a web page needs.
import { readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const read = (path) => readFile(new URL(path, root));
const wasm = await read("target/wasm32-unknown-unknown/release/strategist_sim.wasm");
const { instance } = await WebAssembly.instantiate(wasm, {});
const sim = instance.exports;

const encoder = new TextEncoder();
function pass(text) {
  const bytes = typeof text === "string" ? encoder.encode(text) : text;
  const ptr = sim.sim_alloc(bytes.length);
  new Uint8Array(sim.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}
function output(ok) {
  const bytes = new Uint8Array(sim.memory.buffer, sim.sim_output_ptr(), sim.sim_output_len());
  const text = new TextDecoder().decode(bytes);
  if (!ok) throw new Error(text);
  return text;
}

let start = performance.now();
output(sim.sim_load(...pass(await read("tests/fixtures/model.json")),
                    ...pass(await read("tests/fixtures/params.json"))));
console.log(`module ${(wasm.length / 1024).toFixed(0)} KB, model loaded in ${(performance.now() - start).toFixed(0)} ms`);

const { setup } = JSON.parse(await read("tests/fixtures/sim_replay_2024_bahrain.json"));
let result;
for (const [runs, repeats] of [[100, 7], [300, 5], [1000, 3]]) {
  let best = Infinity;
  for (let i = 0; i < repeats; i++) {
    start = performance.now();
    result = JSON.parse(output(sim.sim_simulate(...pass(JSON.stringify({ setup, runs, seed: 0 })))));
    best = Math.min(best, performance.now() - start);
  }
  console.log(`${String(runs).padStart(5)} runs: ${(best / 1000).toFixed(3)} s`);
}
console.log("finish (1000 runs):", result.finish.slice(0, 5)
  .map((r) => `${r.driver} ${r.mean_pos.toFixed(2)}`).join(", "));

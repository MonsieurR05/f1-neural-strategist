// Build the WebAssembly simulator and copy it into the web app (apps/web/public/sim), next to
// the model and race setups that `python -m strategist.sim.export --web apps/web/public/sim`
// writes. Run from this crate: `node scripts/build_web.mjs`. The output is committed, because
// Vercel's build has no Rust toolchain.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const crate = fileURLToPath(new URL("../", import.meta.url));
execFileSync("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown"], { cwd: crate, stdio: "inherit" });

const from = new URL("../target/wasm32-unknown-unknown/release/strategist_sim.wasm", import.meta.url);
const dir = new URL("../../../apps/web/public/sim/", import.meta.url);
mkdirSync(dir, { recursive: true });
const to = new URL("strategist_sim.wasm", dir);
copyFileSync(from, to);
console.log(`${fileURLToPath(to)} (${(statSync(to).size / 1024).toFixed(0)} KB)`);

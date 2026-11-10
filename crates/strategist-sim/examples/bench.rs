//! Time the simulator on a fixture race: `cargo run --release --example bench`.
//! Needs the fixtures from `python -m strategist.sim.export`.

use std::path::PathBuf;
use std::time::Instant;
use strategist_sim::{simulate, LapNet, RaceSetup, Rng, SimOptions, SimParams};

#[derive(serde::Deserialize)]
struct Case {
    setup: RaceSetup,
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).expect("run strategist.sim.export first");
    let net = LapNet::from_json(&read("model.json")).unwrap();
    let params: SimParams = serde_json::from_str(&read("params.json")).unwrap();
    let case: Case = serde_json::from_str(&read("sim_replay_2024_bahrain.json")).unwrap();

    // best of several repeats: single timings on a laptop vary by 50%
    for (runs, repeats) in [(100, 7), (300, 5), (1000, 3)] {
        let options = SimOptions { runs, target: None, variants: None, interactions: true,
                                   record_laps: true, members: None };
        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            let start = Instant::now();
            simulate(&net, &params, &case.setup, &options, &mut Rng::new(0)).unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        let laps = case.setup.total_laps - case.setup.start_lap;
        let rows = (runs * case.setup.cars.len() * laps) as f64;
        println!("{runs:5} runs: {best:6.3} s  ({:.2} µs per car-lap)", best / rows * 1e6);
    }
}

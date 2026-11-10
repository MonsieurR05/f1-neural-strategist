//! JSON in, JSON out: the interface the WebAssembly build exposes (`wasm.rs`), kept free of
//! anything browser-specific so it is tested natively.

use crate::draws::Rng;
use crate::model::LapNet;
use crate::sim::{simulate, RaceSetup, SimOptions, SimParams, Stint};
use crate::summary::{FinishRow, LapBands};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A loaded lap-time network and simulator parameters.
pub struct Engine {
    net: LapNet,
    params: SimParams,
}

#[derive(Deserialize)]
struct SimulateRequest {
    setup: RaceSetup,
    runs: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    members: Option<usize>,
}

/// The simulated part of the API's `SimulateResponse` (`services/api/app/schemas.py`).
#[derive(Serialize)]
struct SimulateResponse {
    start_lap: usize,
    runs: usize,
    strategies: BTreeMap<String, String>,
    finish: Vec<FinishRow>,
    /// Per driver, index 0 = lap `start_lap`.
    laps: BTreeMap<String, LapBands>,
}

/// "S-H-S (stops 12, 30)", as `Strategy.describe` in Python.
fn describe(stints: &[Stint]) -> String {
    let compounds: Vec<&str> = stints.iter().map(|s| s.compound.get(..1).unwrap_or("?")).collect();
    let stops: Vec<String> = stints[..stints.len().saturating_sub(1)].iter()
        .map(|s| s.end_lap.to_string()).collect();
    let tail = if stops.is_empty() { " (no stop)".to_string() } else { format!(" (stops {})", stops.join(", ")) };
    compounds.join("-") + &tail
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

impl Engine {
    /// From the JSON written by `strategist.sim.export`: `model_dict` and the parameters.
    pub fn new(model_json: &str, params_json: &str) -> Result<Engine, String> {
        Ok(Engine {
            net: LapNet::from_json(model_json)?,
            params: serde_json::from_str(params_json).map_err(|e| format!("parameters: {e}"))?,
        })
    }

    /// Simulate a race setup and summarise it: `{setup, runs, seed?, members?}` in, the
    /// finish table and per-lap position bands out.
    pub fn simulate(&self, request_json: &str) -> Result<String, String> {
        let req: SimulateRequest = serde_json::from_str(request_json).map_err(|e| format!("request: {e}"))?;
        let options = SimOptions { runs: req.runs, target: None, variants: None, interactions: true,
                                   record_laps: true, members: req.members };
        let res = simulate(&self.net, &self.params, &req.setup, &options, &mut Rng::new(req.seed))?;
        let laps = res.lap_bands().into_iter().map(|(driver, mut b)| {
            for v in [&mut b.median_pos, &mut b.p10_pos, &mut b.p90_pos] {
                v.iter_mut().for_each(|x| *x = round3(*x));
            }
            b.median_gap_s.iter_mut().for_each(|x| *x = x.map(round3));
            (driver, b)
        }).collect();
        let response = SimulateResponse {
            start_lap: req.setup.start_lap,
            runs: req.runs,
            strategies: req.setup.cars.iter().map(|c| (c.driver.clone(), describe(&c.strategy))).collect(),
            finish: res.finish_table(),
            laps,
        };
        serde_json::to_string(&response).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategies_are_described_like_the_python_ones() {
        let stint = |c: &str, end| Stint { compound: c.into(), end_lap: end, age_at_start: 0 };
        assert_eq!(describe(&[stint("SOFT", 12), stint("HARD", 30), stint("SOFT", 57)]), "S-H-S (stops 12, 30)");
        assert_eq!(describe(&[stint("MEDIUM", 57)]), "M (no stop)");
    }
}

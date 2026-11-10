//! Parity with the Python reference. The fixtures come from
//! `python -m strategist.sim.export` (run from the repository root); they are generated, not
//! committed, so these tests always compare against the current Python code and models.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use strategist_sim::{simulate, LapNet, RaceSetup, Replay, Row, SimOptions, SimParams, SimResult};

/// Percent of the qualifying lap. float32 rounding differs with summation order.
const NETWORK_TOL: f32 = 2e-4;
/// Seconds of cumulative race time (a race is about 5,500 s).
const TIME_TOL: f64 = 5e-3;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!(
        "missing {}: run `python -m strategist.sim.export` from the repository root first",
        path.display()))
}

fn net() -> LapNet {
    LapNet::from_json(&fixture("model.json")).expect("model.json")
}

#[derive(Deserialize)]
struct NetworkInputs {
    circuit: Vec<usize>,
    compound: Vec<usize>,
    context: Vec<[f32; 5]>,
    tyre_life: Vec<f32>,
    laps_remaining: Vec<f32>,
    fresh: Vec<f32>,
    gap: Vec<f32>,
    leader: Vec<f32>,
    fuel_frac: Vec<f32>,
    hardness: Vec<f32>,
}

#[derive(Deserialize)]
struct NetworkCase {
    inputs: NetworkInputs,
    mu: Vec<f32>,
    sigma: Vec<f32>,
    mu_2: Vec<f32>,
    sigma_2: Vec<f32>,
}

#[test]
fn network_matches_torch() {
    let net = net();
    let case: NetworkCase = serde_json::from_str(&fixture("network.json")).unwrap();
    let x = &case.inputs;
    assert!(case.mu.len() >= 100);
    let mut worst = 0.0f32;
    for i in 0..case.mu.len() {
        let row = Row { circuit: x.circuit[i], compound: x.compound[i], context: x.context[i],
                        tyre_life: x.tyre_life[i], laps_remaining: x.laps_remaining[i],
                        fresh: x.fresh[i], gap: x.gap[i], leader: x.leader[i],
                        fuel_frac: x.fuel_frac[i], hardness: x.hardness[i] };
        let (mu, sigma) = net.predict(&row, None);
        let (mu_2, sigma_2) = net.predict(&row, Some(2));
        for (got, want) in [(mu, case.mu[i]), (sigma, case.sigma[i]),
                            (mu_2, case.mu_2[i]), (sigma_2, case.sigma_2[i])] {
            worst = worst.max((got - want).abs());
            assert!((got - want).abs() < NETWORK_TOL, "row {i}: {got} vs torch {want}");
        }
    }
    println!("network: {} rows, largest difference {worst:.2e} %", case.mu.len());
}

#[derive(Deserialize)]
struct DrawsJson {
    normal: Vec<f64>,
    uniform: Vec<f64>,
    choice: Vec<i64>,
}

#[derive(Deserialize)]
struct Expected {
    shape: [usize; 3],
    times: Vec<Option<f64>>,
    laps_done: Vec<f64>,
    held_s: Vec<f64>,
    status: Vec<Vec<String>>,
    finish: Option<FinishColumns>,
    bands: Option<HashMap<String, Bands>>,
}

/// `SimResult.finish_table()` by column, best mean position first.
#[derive(Deserialize)]
struct FinishColumns {
    driver: Vec<String>,
    mean_pos: Vec<f64>,
    median_pos: Vec<f64>,
    p10_pos: Vec<f64>,
    p90_pos: Vec<f64>,
    p_win: Vec<f64>,
    p_podium: Vec<f64>,
    p_points: Vec<f64>,
    median_gap_s: Vec<Option<f64>>,
    mean_held_s: Vec<f64>,
}

#[derive(Deserialize)]
struct Bands {
    median_pos: Vec<f64>,
    p10_pos: Vec<f64>,
    p90_pos: Vec<f64>,
    median_gap_s: Vec<Option<f64>>,
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() < tol, "{what}: {got} vs Python {want}");
}

fn close_or_absent(got: Option<f64>, want: Option<f64>, what: &str) {
    match (got, want) {
        (Some(g), Some(w)) => close(g, w, TIME_TOL, what),
        (None, None) => {}
        _ => panic!("{what}: {got:?} vs Python {want:?}"),
    }
}

/// Cars whose position at a recorded lap is a coin toss in some run: out of the race with
/// the same number of laps completed as another car that is out. NumPy orders such ties
/// arbitrarily (an unstable sort); the Rust port keeps car order.
fn tied(res: &SimResult, lap: usize) -> Vec<bool> {
    let [_, runs, cars] = res.shape;
    let mut tied = vec![false; cars];
    for r in 0..runs {
        let out = |i: usize| !res.times[(lap * runs + r) * cars + i].is_finite();
        let done = |i: usize| res.laps_done[r * cars + i];
        for i in (0..cars).filter(|&i| out(i)) {
            tied[i] |= (0..cars).any(|j| j != i && out(j) && done(j) == done(i));
        }
    }
    tied
}

/// The finish table and per-lap bands the app shows, against the Python ones. Position
/// statistics are skipped where the reference itself is ambiguous (see `tied`).
fn check_summaries(name: &str, res: &SimResult, want: &Expected) -> usize {
    let (Some(finish), Some(bands)) = (&want.finish, &want.bands) else { return 0 };
    let index = |driver: &str| res.drivers.iter().position(|d| d == driver).unwrap();
    let mut compared = 0;

    let table = res.finish_table();
    let ambiguous = tied(res, res.shape[0] - 1);
    assert_eq!(table.len(), finish.driver.len());
    for (i, driver) in finish.driver.iter().enumerate() {
        let row = table.iter().find(|r| &r.driver == driver).unwrap();
        let what = format!("{name}: finish table, {driver}");
        if !ambiguous[index(driver)] {
            for (got, want) in [(row.mean_pos, finish.mean_pos[i]), (row.median_pos, finish.median_pos[i]),
                                (row.p10_pos, finish.p10_pos[i]), (row.p90_pos, finish.p90_pos[i]),
                                (row.p_win, finish.p_win[i]), (row.p_podium, finish.p_podium[i]),
                                (row.p_points, finish.p_points[i])] {
                close(got, want, 1e-9, &what);
            }
            compared += 1;
        }
        close(row.mean_held_s, finish.mean_held_s[i], TIME_TOL, &what);
        close_or_absent(row.median_gap_s, finish.median_gap_s[i], &what);
    }

    let got = res.lap_bands();
    for lap in 0..res.shape[0] - res.start_lap {
        let ambiguous = tied(res, res.start_lap + lap);
        for (driver, got) in &got {
            let want = &bands[driver];
            assert_eq!(got.median_pos.len(), want.median_pos.len(), "{name}: laps in bands");
            let what = format!("{name}: bands, {driver}, lap index {lap}");
            if !ambiguous[index(driver)] {
                close(got.median_pos[lap], want.median_pos[lap], 1e-9, &what);
                close(got.p10_pos[lap], want.p10_pos[lap], 1e-9, &what);
                close(got.p90_pos[lap], want.p90_pos[lap], 1e-9, &what);
                compared += 1;
            }
            close_or_absent(got.median_gap_s[lap], want.median_gap_s[lap], &what);
        }
    }
    compared
}

#[derive(Deserialize)]
struct SimCase {
    setup: RaceSetup,
    options: SimOptions,
    draws: DrawsJson,
    expected: Expected,
}

/// Replay a Python run's random draws and require the same race, lap by lap.
fn replay(name: &str) {
    let net = net();
    let params: SimParams = serde_json::from_str(&fixture("params.json")).unwrap();
    let case: SimCase = serde_json::from_str(&fixture(&format!("sim_{name}.json"))).unwrap();
    let mut draws = Replay::new(case.draws.normal, case.draws.uniform, case.draws.choice);
    let res = simulate(&net, &params, &case.setup, &case.options, &mut draws).unwrap();
    let want = &case.expected;

    assert!(draws.exhausted(), "{name}: Rust and Python drew different amounts of randomness");
    assert_eq!(res.shape, want.shape, "{name}: shape");
    let [_, runs, cars] = res.shape;
    for (r, row) in want.status.iter().enumerate() {
        let got: Vec<&str> = res.status[r % case.options.runs].iter().map(|s| s.name()).collect();
        assert_eq!(&got, row, "{name}: track status, run {r}");
    }
    assert_eq!(res.laps_done, want.laps_done, "{name}: laps completed");

    let (mut worst, mut finite) = (0.0f64, 0usize);
    for (i, (got, want)) in res.times.iter().zip(&want.times).enumerate() {
        let (lap, run, car) = (i / (runs * cars), i / cars % runs, i % cars);
        match want {
            Some(w) => {
                finite += 1;
                worst = worst.max((got - w).abs());
                assert!((got - w).abs() < TIME_TOL,
                        "{name}: lap {lap} run {run} car {car}: {got} vs Python {w}");
            }
            None => assert!(got.is_infinite(), "{name}: lap {lap} run {run} car {car} should have retired"),
        }
    }
    for (i, (got, w)) in res.held_s.iter().zip(&want.held_s).enumerate() {
        assert!((got - w).abs() < TIME_TOL, "{name}: time held, entry {i}: {got} vs Python {w}");
    }
    assert!(finite > 0);
    let summaries = check_summaries(name, &res, want);
    assert!(summaries > 0 || want.finish.is_none(), "{name}: no summaries compared");
    println!("{name}: {finite} times, largest difference {worst:.2e} s; {summaries} position summaries");
}

#[test]
fn replay_with_actual_events() {
    replay("replay_2024_bahrain");
}

#[test]
fn replay_through_a_safety_car() {
    replay("replay_2022_bahrain_safety_car");
}

#[test]
fn forecast_from_the_grid_with_sampled_events() {
    replay("forecast_2025_bahrain");
}

#[test]
fn strategy_variants_share_random_numbers() {
    replay("variants_2024_bahrain");
}

#[test]
fn free_passing_ablation() {
    replay("free_passing_2024_bahrain");
}

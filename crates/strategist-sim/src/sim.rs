//! Lap-by-lap Monte Carlo race simulator: a port of `strategist.sim.race`.
//!
//! The Python simulator is the reference and its module docstring explains the model. This
//! file follows it step for step, including the order in which random numbers are drawn, so
//! that replaying a Python run's draws reproduces its lap times (see `tests/parity.rs`).
//! Not ported: the in-race update (`RaceSetup.pace_paths`), which is off by default there.

use crate::draws::Draws;
use crate::model::{LapNet, RaceNet, TyreState, COMPOUNDS, DEFAULT_HARDNESS, GAP_CAP_S};
use serde::Deserialize;
use std::collections::HashMap;

const CLOSE_GAP_S: f64 = 1.0; // pairs closer than this at the start of a lap are "in a fight"
const PASS_MARGIN_S: f64 = 0.2;

#[derive(Deserialize, Clone, Debug)]
pub struct Stint {
    pub compound: String,
    /// Last lap on this set (the in-lap, if a stop follows).
    pub end_lap: i64,
    #[serde(default)]
    pub age_at_start: i64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Car {
    pub driver: String,
    #[serde(default)]
    pub team: String,
    /// Best qualifying lap: the pace anchor.
    pub q_s: f64,
    pub quali_gap_pct: f64,
    pub quali_missing: f64,
    pub grid_position: f64,
    pub strategy: Vec<Stint>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RaceSetup {
    pub circuit: String,
    pub total_laps: usize,
    pub track_temp: f64,
    pub cars: Vec<Car>,
    #[serde(default)]
    pub start_lap: usize,
    /// Cumulative times after `start_lap` laps; `None` for a standing start from the grid.
    pub start_times: Option<Vec<f64>>,
    /// Known neutralisations as (lap, "SC" | "VSC" | "RED"); `None` to sample them.
    pub events: Option<Vec<(usize, String)>>,
    /// Driver → last completed lap; `None` to sample retirements.
    pub retirements: Option<HashMap<String, i64>>,
    /// Pirelli hardness of the nominated compounds; typical values if `None`.
    pub hardness: Option<HashMap<String, f64>>,
}

/// Simulator parameters calibrated by `strategist.sim.params`.
#[derive(Deserialize, Clone, Debug)]
pub struct SimParams {
    pub pit_loss: HashMap<String, f64>,
    pub pit_loss_global: f64,
    pub pit_loss_sd: f64,
    pub pit_loss_neutral_ratio: f64,
    pub sc_lap_ratio: f64,
    pub vsc_lap_ratio: f64,
    pub sc_queue_gap: f64,
    pub sc_compress: f64,
    pub sc_first_lap_factor: f64,
    pub overtake_intercept: HashMap<String, f64>,
    pub overtake_intercept_global: f64,
    pub overtake_pace: f64,
    pub overtake_gap: f64,
    pub held_gap_low: f64,
    pub held_gap_high: f64,
    pub common_var_share: f64,
    pub driver_var_share: f64,
    #[serde(default)]
    pub sc_hazard: HashMap<String, f64>,
    #[serde(default)]
    pub sc_hazard_global: f64,
    #[serde(default)]
    pub vsc_hazard_global: f64,
    #[serde(default)]
    pub sc_durations: Vec<i64>,
    #[serde(default)]
    pub vsc_durations: Vec<i64>,
    #[serde(default)]
    pub dnf_hazard: f64,
    #[serde(default)]
    pub lap1_ratio: HashMap<String, f64>,
    #[serde(default = "default_lap1_ratio")]
    pub lap1_ratio_global: f64,
    #[serde(default = "default_grid_spacing")]
    pub grid_spacing_s: f64,
    #[serde(default = "default_start_noise")]
    pub start_noise_s: f64,
}

fn default_lap1_ratio() -> f64 { 1.1 }
fn default_grid_spacing() -> f64 { 0.25 }
fn default_start_noise() -> f64 { 0.6 }

#[derive(Deserialize, Clone, Debug)]
pub struct SimOptions {
    /// Runs per strategy variant.
    pub runs: usize,
    /// Simulate several strategies for this driver at once, sharing random numbers.
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub variants: Option<Vec<Vec<Stint>>>,
    #[serde(default = "yes")]
    pub interactions: bool,
    /// `false` keeps only the finish (much less memory when screening many variants).
    #[serde(default = "yes")]
    pub record_laps: bool,
    /// Limit the lap-time ensemble (faster, slightly noisier screening).
    #[serde(default)]
    pub members: Option<usize>,
}

fn yes() -> bool { true }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status { Green, Sc, Vsc, Red }

impl Status {
    fn parse(kind: &str) -> Result<Status, String> {
        match kind {
            "GREEN" => Ok(Status::Green),
            "SC" => Ok(Status::Sc),
            "VSC" => Ok(Status::Vsc),
            "RED" => Ok(Status::Red),
            other => Err(format!("unknown track status {other:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Status::Green => "GREEN",
            Status::Sc => "SC",
            Status::Vsc => "VSC",
            Status::Red => "RED",
        }
    }
}

/// Runs are laid out as variant × runs per variant; run `r` uses the draws of `r % runs`.
pub struct SimResult {
    pub drivers: Vec<String>,
    /// Cumulative time, `[laps, runs, cars]`; `laps` is `total_laps + 1`, or 1 (the finish)
    /// without `record_laps`. Retired cars are infinite.
    pub times: Vec<f64>,
    pub shape: [usize; 3],
    /// Laps completed, `[runs, cars]`.
    pub laps_done: Vec<f64>,
    /// Time lost stuck behind other cars on green laps, `[runs, cars]`.
    pub held_s: Vec<f64>,
    pub start_lap: usize,
    pub variants: usize,
    /// Track status per draw and lap, `[runs per variant][total_laps + 1]`.
    pub status: Vec<Vec<Status>>,
}

struct Plan {
    compound: Vec<usize>,
    age: Vec<f64>,
    fresh: Vec<f64>,
    pits: Vec<bool>,
}

/// Per lap 1..=total (index = lap): compound, tyre age (1 on the first lap of a fresh set),
/// whether the set was fresh when fitted, and whether the car stops at the end of the lap.
fn plan(stints: &[Stint], total: usize) -> Result<Plan, String> {
    let mut p = Plan { compound: vec![0; total + 1], age: vec![0.0; total + 1],
                       fresh: vec![0.0; total + 1], pits: vec![false; total + 1] };
    let mut start = 1i64;
    for (i, s) in stints.iter().enumerate() {
        let compound = COMPOUNDS.iter().position(|c| *c == s.compound)
            .ok_or_else(|| format!("unknown compound {:?}", s.compound))?;
        let last = i + 1 == stints.len();
        let end = if last { total as i64 } else { s.end_lap.min(total as i64) };
        for lap in start.max(1)..=end {
            p.compound[lap as usize] = compound;
            p.age[lap as usize] = (s.age_at_start + (lap - start) + 1) as f64;
            p.fresh[lap as usize] = if s.age_at_start == 0 { 1.0 } else { 0.0 };
        }
        start = end + 1;
        if !last && s.end_lap >= 1 && s.end_lap < total as i64 {
            p.pits[s.end_lap as usize] = true;
        }
    }
    Ok(p)
}

fn sample_events<D: Draws>(draws: &mut D, p: &SimParams, setup: &RaceSetup, runs: usize) -> Vec<Vec<Status>> {
    let l = setup.total_laps;
    let h_sc = p.sc_hazard.get(&setup.circuit).copied().unwrap_or(p.sc_hazard_global);
    let h_vsc = p.vsc_hazard_global;
    let sc_d: &[i64] = if p.sc_durations.is_empty() { &[4] } else { &p.sc_durations };
    let vsc_d: &[i64] = if p.vsc_durations.is_empty() { &[2] } else { &p.vsc_durations };
    let mut status = vec![vec![Status::Green; l + 1]; runs];
    let mut u = [0.0f64];
    for row in status.iter_mut() {
        let mut lap = (setup.start_lap + 1).max(1);
        while lap <= l {
            draws.uniform(&mut u);
            let (kind, options) = if u[0] < h_sc {
                (Status::Sc, sc_d)
            } else if u[0] < h_sc + h_vsc {
                (Status::Vsc, vsc_d)
            } else {
                lap += 1;
                continue;
            };
            let d = draws.choice(options).max(0) as usize;
            for x in row.iter_mut().take(lap + d).skip(lap) {
                *x = kind;
            }
            lap += d + 1;
        }
    }
    status
}

pub fn simulate<D: Draws>(net: &LapNet, p: &SimParams, setup: &RaceSetup, opt: &SimOptions,
                          draws: &mut D) -> Result<SimResult, String> {
    let (m_runs, c, l) = (opt.runs, setup.cars.len(), setup.total_laps);
    if m_runs == 0 || c == 0 || l == 0 {
        return Err("need at least one run, one car and one lap".into());
    }
    let variants = opt.variants.as_deref().filter(|v| !v.is_empty());
    let k_variants = variants.map_or(1, <[_]>::len);
    let r_runs = k_variants * m_runs;
    let target = match (variants, &opt.target) {
        (Some(_), Some(d)) => Some(setup.cars.iter().position(|car| &car.driver == d)
            .ok_or_else(|| format!("{d} is not in this race"))?),
        (Some(_), None) => return Err("variants need a target driver".into()),
        _ => None,
    };

    // one slot per distinct tyre plan: every car, then the target's variants
    let mut plans = setup.cars.iter().map(|car| plan(&car.strategy, l)).collect::<Result<Vec<_>, _>>()?;
    let mut slot_car: Vec<usize> = (0..c).collect();
    if let (Some(vs), Some(ti)) = (variants, target) {
        for v in vs {
            plans.push(plan(v, l)?);
            slot_car.push(ti);
        }
    }
    let slot_of = |variant: usize, car: usize| if target == Some(car) { c + variant } else { car };

    let q: Vec<f64> = setup.cars.iter().map(|car| car.q_s).collect();
    let contexts: Vec<_> = setup.cars.iter().map(|car| net.context(
        [car.quali_gap_pct, car.quali_missing, car.grid_position, setup.track_temp, l as f64])).collect();
    let mut hardness = DEFAULT_HARDNESS;
    if let Some(h) = &setup.hardness {
        for (i, name) in COMPOUNDS.iter().enumerate() {
            if let Some(&v) = h.get(*name) {
                hardness[i] = v as f32;
            }
        }
    }
    let mut race_net = RaceNet::new(net, opt.members, &setup.circuit, &contexts, hardness);

    // pace randomness: race-wide shift per run + race-long offset per (run, car) + lap noise
    let common = p.common_var_share.clamp(0.0, 0.95);
    let driver = p.driver_var_share.clamp(0.0, 0.95 - common);
    let lap_sd = (1.0 - common - driver).sqrt();
    let mut z_common = vec![0.0; m_runs];
    let mut offset_z = vec![0.0; m_runs * c];
    draws.normal(&mut z_common);
    draws.normal(&mut offset_z);
    for (i, z) in offset_z.iter_mut().enumerate() {
        *z = common.sqrt() * z_common[i / c] + driver.sqrt() * *z;
    }

    let status = match &setup.events {
        Some(events) => {
            let mut row = vec![Status::Green; l + 1];
            for (lap, kind) in events {
                if (1..=l).contains(lap) {
                    row[*lap] = Status::parse(kind)?;
                }
            }
            vec![row; m_runs]
        }
        None => sample_events(draws, p, setup, m_runs),
    };

    // last lap each car completes, per (draw, car)
    let mut retire = vec![(l + 1) as i64; m_runs * c];
    match &setup.retirements {
        Some(known) => {
            for (i, car) in setup.cars.iter().enumerate() {
                if let Some(&lap) = known.get(&car.driver) {
                    (0..m_runs).for_each(|m| retire[m * c + i] = lap);
                }
            }
        }
        None => {
            let mut u = vec![0.0; m_runs * c * l];
            draws.uniform(&mut u);
            for (i, laps) in u.chunks(l).enumerate() {
                if let Some(first) = laps.iter().position(|x| *x < p.dnf_hazard) {
                    retire[i] = (first + 1) as i64;
                }
            }
        }
    }

    let mut current = vec![0.0f64; r_runs * c];
    let first_lap = match &setup.start_times {
        Some(start) => {
            if start.len() != c {
                return Err("start_times must have one entry per car".into());
            }
            current.chunks_mut(c).for_each(|row| row.copy_from_slice(start));
            setup.start_lap + 1
        }
        None => { // standing start: lap 1 from the grid
            let ratio = p.lap1_ratio.get(&setup.circuit).copied().unwrap_or(p.lap1_ratio_global);
            let pole = q.iter().copied().fold(f64::INFINITY, f64::min);
            let mut z = vec![0.0; m_runs * c];
            draws.normal(&mut z);
            for r in 0..r_runs {
                for (i, car) in setup.cars.iter().enumerate() {
                    let grid = if car.grid_position > 0.0 { car.grid_position } else { c as f64 };
                    current[r * c + i] = pole * ratio + p.grid_spacing_s * (grid - 1.0)
                        + p.start_noise_s * z[(r % m_runs) * c + i];
                }
            }
            2
        }
    };
    for r in 0..r_runs {
        for i in 0..c {
            if retire[(r % m_runs) * c + i] < first_lap as i64 - 1 {
                current[r * c + i] = f64::INFINITY;
            }
        }
    }
    let recorded = if opt.record_laps { l + 1 } else { 1 };
    let mut times = vec![f64::INFINITY; recorded * r_runs * c];
    if opt.record_laps {
        let at = (first_lap - 1) * r_runs * c;
        times[at..at + r_runs * c].copy_from_slice(&current);
    }
    let mut laps_done: Vec<f64> = current.iter()
        .map(|t| if t.is_finite() { (first_lap - 1) as f64 } else { 0.0 }).collect();
    let mut held_s = vec![0.0f64; r_runs * c];
    let mut sc_lap_no = vec![0usize; r_runs]; // laps into the current safety-car period

    let pit_loss = p.pit_loss.get(&setup.circuit).copied().unwrap_or(p.pit_loss_global);
    let pass_a = p.overtake_intercept.get(&setup.circuit).copied().unwrap_or(p.overtake_intercept_global);

    let mut noise = vec![0.0; m_runs * c];
    let mut pit_z = vec![0.0; m_runs * c];
    let mut pass_u = vec![0.0; m_runs * c];
    let mut hold_u = vec![0.0; m_runs * c];
    let mut tyres = Vec::with_capacity(plans.len());
    let (mut order, mut running) = (vec![0usize; c], vec![false; c]);
    let (mut key, mut sorted_prev, mut gap) = (vec![0.0f64; c], vec![0.0f64; c], vec![0.0f64; c]);
    let (mut mu, mut t, mut pits) = (vec![0.0f64; c], vec![0.0f64; c], vec![false; c]);

    #[allow(clippy::needless_range_loop)] // lap indexes plans, status and times alike
    for lap in first_lap..=l {
        let laps_remaining = (l - lap) as f64;
        tyres.clear();
        tyres.extend(plans.iter().zip(&slot_car).map(|(pl, &car)| TyreState {
            car, compound: pl.compound[lap], age: pl.age[lap] as f32, fresh: pl.fresh[lap] as f32 }));
        race_net.set_lap(&tyres, laps_remaining as f32, (laps_remaining / l as f64) as f32);

        draws.normal(&mut noise);
        draws.normal(&mut pit_z);
        if opt.interactions {
            draws.uniform(&mut pass_u);
            draws.uniform(&mut hold_u);
        }

        for r in 0..r_runs {
            let (m, variant) = (r % m_runs, r / m_runs);
            let kind = status[m][lap];
            let sc = matches!(kind, Status::Sc | Status::Red);
            let vsc = kind == Status::Vsc;
            let green = !(sc || vsc);
            let prev = &current[r * c..(r + 1) * c];

            for i in 0..c {
                running[i] = prev[i].is_finite() && retire[m * c + i] >= lap as i64;
                key[i] = if running[i] { prev[i] } else { f64::INFINITY };
                order[i] = i;
            }
            order.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
            for (pos, &i) in order.iter().enumerate() {
                sorted_prev[pos] = prev[i];
                gap[i] = if pos == 0 { f64::NAN } else { prev[i] - sorted_prev[pos - 1] };
            }
            let leader = order[0];

            for i in 0..c {
                pits[i] = plans[slot_of(variant, i)].pits[lap];
                if !running[i] { // retired: its lap time is never used
                    mu[i] = 0.0;
                    t[i] = f64::INFINITY;
                    continue;
                }
                let capped = if i == leader || !gap[i].is_finite() { GAP_CAP_S } else { gap[i].min(GAP_CAP_S) };
                let (mu_pct, sigma_pct) = race_net.predict(slot_of(variant, i), capped as f32, i == leader);
                // the Python simulator forms these two factors in float32 before scaling by q
                mu[i] = f64::from(1.0f32 + mu_pct / 100.0f32) * q[i];
                let sigma = f64::from(sigma_pct / 100.0f32) * q[i];
                let lap_time = if vsc {
                    p.vsc_lap_ratio * mu[i]
                } else {
                    mu[i] + sigma * (offset_z[m * c + i] + lap_sd * noise[m * c + i])
                };
                t[i] = prev[i] + lap_time;
            }

            // safety car: leader at SC pace, everyone else closes up into a queue. On the lap
            // it is deployed the field hardly bunches (cars slow at different points of the lap)
            sc_lap_no[r] = if sc { sc_lap_no[r] + 1 } else { 0 };
            if sc {
                let factor = if sc_lap_no[r] == 1 { p.sc_first_lap_factor } else { p.sc_compress };
                let mut ahead = sorted_prev[0] + p.sc_lap_ratio * mu[leader];
                if running[leader] {
                    t[leader] = ahead;
                }
                for pos in 1..c {
                    let old_gap = sorted_prev[pos] - sorted_prev[pos - 1];
                    ahead += if old_gap.is_finite() { p.sc_queue_gap.max(old_gap * factor) } else { f64::INFINITY };
                    if running[order[pos]] {
                        t[order[pos]] = ahead;
                    }
                }
            }
            for i in 0..c {
                if pits[i] {
                    let mut loss = pit_loss + p.pit_loss_sd * pit_z[m * c + i];
                    if !green {
                        loss *= p.pit_loss_neutral_ratio;
                    }
                    t[i] += loss.max(0.0);
                }
            }

            if opt.interactions {
                for pos in 1..c {
                    let (i, j) = (order[pos], order[pos - 1]);
                    let (ti, tj) = (t[i], t[j]);
                    let ok = ti.is_finite() && tj.is_finite() && running[i] && running[j];
                    let either_pits = pits[i] || pits[j];
                    let start_gap = if gap[i].is_finite() { gap[i] } else { GAP_CAP_S };
                    let close = ok && green && !either_pits && start_gap < CLOSE_GAP_S;
                    let logit = pass_a + p.overtake_pace * (mu[j] - mu[i]) + p.overtake_gap * start_gap;
                    let prob = 1.0 / (1.0 + (-logit.clamp(-30.0, 30.0)).exp());
                    let passed = close && pass_u[m * c + pos] < prob;
                    let held = close && !passed;
                    let mut new = if passed && ti >= tj { tj - PASS_MARGIN_S } else { ti };
                    if held {
                        let following = p.held_gap_low + (p.held_gap_high - p.held_gap_low) * hold_u[m * c + pos];
                        new = new.max(tj + following);
                    }
                    // no overtaking under VSC/SC (except via the pit lane)
                    if ok && !green && !either_pits {
                        new = new.max(tj + PASS_MARGIN_S);
                    }
                    if held && new.is_finite() {
                        held_s[r * c + i] += new - ti;
                    }
                    t[i] = new;
                }
            }

            for i in 0..c {
                let now = if running[i] { t[i] } else { f64::INFINITY };
                current[r * c + i] = now;
                if now.is_finite() {
                    laps_done[r * c + i] += 1.0;
                }
            }
        }
        if opt.record_laps {
            let at = lap * r_runs * c;
            times[at..at + r_runs * c].copy_from_slice(&current);
        }
    }
    if !opt.record_laps {
        times.copy_from_slice(&current);
    }
    Ok(SimResult {
        drivers: setup.cars.iter().map(|car| car.driver.clone()).collect(),
        times, shape: [recorded, r_runs, c], laps_done, held_s,
        start_lap: setup.start_lap, variants: k_variants, status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stint(compound: &str, end_lap: i64, age_at_start: i64) -> Stint {
        Stint { compound: compound.into(), end_lap, age_at_start }
    }

    #[test]
    fn tyre_plan_resets_age_after_a_stop() {
        let p = plan(&[stint("SOFT", 3, 2), stint("HARD", 99, 0)], 6).unwrap();
        assert_eq!(p.compound[1..], [0, 0, 0, 2, 2, 2]);
        assert_eq!(p.age[1..], [3.0, 4.0, 5.0, 1.0, 2.0, 3.0]);
        assert_eq!(p.fresh[1..], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.pits[1..], [false, false, true, false, false, false]);
    }

    #[test]
    fn a_stop_on_the_last_lap_is_ignored_and_unknown_compounds_are_refused() {
        let p = plan(&[stint("MEDIUM", 6, 0), stint("HARD", 99, 0)], 6).unwrap();
        assert!(p.pits.iter().all(|x| !x));
        assert!(plan(&[stint("INTER", 6, 0)], 6).is_err());
    }
}

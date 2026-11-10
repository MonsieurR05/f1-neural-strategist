//! What the app shows from a simulation: the finish table and per-lap position bands.
//! Ports of `SimResult.positions`, `finish_table` and `lap_bands` in `strategist.sim.race`.

use crate::sim::SimResult;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct FinishRow {
    pub driver: String,
    pub mean_pos: f64,
    pub median_pos: f64,
    pub p10_pos: f64,
    pub p90_pos: f64,
    pub p_win: f64,
    pub p_podium: f64,
    pub p_points: f64,
    /// Median gap to the winner over the runs the car finishes; `None` if it never does.
    pub median_gap_s: Option<f64>,
    pub mean_held_s: f64,
}

/// One value per lap from `start_lap` to the finish.
#[derive(Serialize, Clone, Debug)]
pub struct LapBands {
    pub median_pos: Vec<f64>,
    pub p10_pos: Vec<f64>,
    pub p90_pos: Vec<f64>,
    /// `None` on laps where the car is out in every run.
    pub median_gap_s: Vec<Option<f64>>,
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
}

/// Linear interpolation between order statistics (NumPy's default).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

impl SimResult {
    fn lap(&self, index: usize) -> &[f64] {
        let [_, runs, cars] = self.shape;
        &self.times[index * runs * cars..(index + 1) * runs * cars]
    }

    /// Position (1 = leader) per run and car at a recorded lap (`None`: the finish). Retired
    /// cars rank behind running ones, later retirements ahead of earlier ones; cars out on
    /// the same lap keep car order (NumPy's order for such ties is arbitrary).
    pub fn positions(&self, lap: Option<usize>) -> Vec<f64> {
        let [recorded, runs, cars] = self.shape;
        let t = self.lap(lap.unwrap_or(recorded - 1));
        let mut pos = vec![0.0; runs * cars];
        let mut order: Vec<usize> = Vec::with_capacity(cars);
        for r in 0..runs {
            let at = r * cars;
            let key = |i: usize| if t[at + i].is_finite() { t[at + i] } else { 1e12 - self.laps_done[at + i] * 1e6 };
            order.clear();
            order.extend(0..cars);
            order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
            for (rank, &i) in order.iter().enumerate() {
                pos[at + i] = (rank + 1) as f64;
            }
        }
        pos
    }

    /// Gap to the leader per run and car at a recorded lap; NaN for cars that are out.
    fn gaps(&self, index: usize) -> Vec<f64> {
        let cars = self.shape[2];
        let mut gap = self.lap(index).to_vec();
        for row in gap.chunks_mut(cars) {
            let lead = row.iter().copied().filter(|t| t.is_finite()).fold(f64::INFINITY, f64::min);
            row.iter_mut().for_each(|t| *t = if t.is_finite() { *t - lead } else { f64::NAN });
        }
        gap
    }

    fn column(values: &[f64], car: usize, cars: usize) -> Vec<f64> {
        values.iter().skip(car).step_by(cars).copied().collect()
    }

    /// Finishing statistics per driver over all runs, best mean position first.
    pub fn finish_table(&self) -> Vec<FinishRow> {
        let [recorded, runs, cars] = self.shape;
        let (pos, gap) = (self.positions(None), self.gaps(recorded - 1));
        let n = runs as f64;
        let mut rows: Vec<FinishRow> = (0..cars).map(|i| {
            let p = sorted(Self::column(&pos, i, cars));
            let finished = sorted(Self::column(&gap, i, cars).into_iter().filter(|g| g.is_finite()).collect());
            let share = |limit: f64| p.iter().filter(|x| **x <= limit).count() as f64 / n;
            FinishRow {
                driver: self.drivers[i].clone(),
                mean_pos: p.iter().sum::<f64>() / n,
                median_pos: median(&p),
                p10_pos: quantile(&p, 0.1),
                p90_pos: quantile(&p, 0.9),
                p_win: share(1.0),
                p_podium: share(3.0),
                p_points: share(10.0),
                median_gap_s: if finished.is_empty() { None } else { Some(median(&finished)) },
                mean_held_s: Self::column(&self.held_s, i, cars).iter().sum::<f64>() / n,
            }
        }).collect();
        rows.sort_by(|a, b| a.mean_pos.total_cmp(&b.mean_pos));
        rows
    }

    /// Position bands and median gap per lap for every driver, in car order. Needs every lap
    /// recorded.
    pub fn lap_bands(&self) -> Vec<(String, LapBands)> {
        let [recorded, _, cars] = self.shape;
        let mut bands: Vec<LapBands> = (0..cars).map(|_| LapBands {
            median_pos: Vec::new(), p10_pos: Vec::new(), p90_pos: Vec::new(), median_gap_s: Vec::new(),
        }).collect();
        for lap in self.start_lap..recorded {
            let (pos, gap) = (self.positions(Some(lap)), self.gaps(lap));
            for (i, b) in bands.iter_mut().enumerate() {
                let p = sorted(Self::column(&pos, i, cars));
                let g = sorted(Self::column(&gap, i, cars).into_iter().filter(|g| g.is_finite()).collect());
                b.median_pos.push(median(&p));
                b.p10_pos.push(quantile(&p, 0.1));
                b.p90_pos.push(quantile(&p, 0.9));
                b.median_gap_s.push(if g.is_empty() { None } else { Some(median(&g)) });
            }
        }
        self.drivers.iter().cloned().zip(bands).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_statistics_match_numpy() {
        let v = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(median(&v), 3.0);
        assert_eq!(median(&v[..3]), 2.0);
        assert!((quantile(&v, 0.1) - 1.3).abs() < 1e-12);
        assert!((quantile(&v, 0.9) - 6.8).abs() < 1e-12);
        assert_eq!(quantile(&v[..1], 0.9), 1.0);
    }
}

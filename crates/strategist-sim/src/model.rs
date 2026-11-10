//! Forward pass of the structured lap-time network (`strategist.models.lapnet`).
//!
//! Two paths compute the same numbers:
//!
//! - [`LapNet::predict`] evaluates one row the way the PyTorch module does. It is the
//!   reference the parity tests compare with torch.
//! - [`RaceNet`] is what the simulator uses. Within a race most inputs are fixed per car or
//!   per lap, so it evaluates base pace, conditions, fuel and the tyre terms once and keeps
//!   only the three sub-networks that see the gap ahead (traffic, interaction, σ) in the
//!   per-run loop, with the static part of their first layer folded in beforehand.
//!
//! Arithmetic is `f32`, as in torch. Summation order differs from torch's, so results agree
//! to rounding (about 1e-6 of a percent), not bit for bit.

use serde::Deserialize;
use std::collections::HashMap;

pub const COMPOUNDS: [&str; 3] = ["SOFT", "MEDIUM", "HARD"];
/// Hardness of the typical nomination, used when the race is not in the compound table.
pub const DEFAULT_HARDNESS: [f32; 3] = [4.0, 3.0, 2.0];
pub const GAP_CAP_S: f64 = 10.0;

const KNOTS: [f32; 3] = [10.0, 20.0, 30.0];
const HARDNESS_KNOTS: [f32; 5] = [1.5, 2.5, 3.5, 4.5, 5.5];
const HARDNESS_SCALE: f32 = 0.4;
const SIGMA_FLOOR: f32 = 0.05;
const INTERACTION_BOUND: f32 = 0.5;
/// Context columns: quali_gap_pct, quali_missing, grid_position, track_temp, total_laps.
pub const N_CONTEXT: usize = 5;
const TEMP: usize = 3;
const RACE_CONTEXT: [usize; 4] = [0, 1, 2, 4];
const NB: usize = 1 + KNOTS.len();
const NH: usize = 1 + HARDNESS_KNOTS.len();
const MAX_HIDDEN: usize = 64;

#[derive(Deserialize)]
struct LayerJson {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

#[derive(Deserialize)]
struct MemberJson {
    circuit_emb: Vec<Vec<f32>>,
    compound_emb: Vec<Vec<f32>>,
    tau: f32,
    base: Vec<LayerJson>,
    conditions: Vec<LayerJson>,
    fuel: Vec<LayerJson>,
    offset: Vec<LayerJson>,
    deg: Vec<LayerJson>,
    traffic: Vec<LayerJson>,
    sigma: Vec<LayerJson>,
    interaction: Option<Vec<LayerJson>>,
}

#[derive(Deserialize)]
struct ModelJson {
    circuit_index: HashMap<String, usize>,
    context_mean: Vec<f64>,
    context_std: Vec<f64>,
    members: Vec<MemberJson>,
}

/// A linear layer stored column by column, so `out += column · x` vectorises.
struct Dense {
    n_in: usize,
    n_out: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Dense {
    fn new(layer: &LayerJson) -> Result<Dense, String> {
        let n_out = layer.w.len();
        let n_in = layer.w.first().map_or(0, Vec::len);
        if n_out == 0 || n_in == 0 || layer.b.len() != n_out || layer.w.iter().any(|r| r.len() != n_in) {
            return Err("malformed layer".into());
        }
        let mut w = vec![0.0; n_in * n_out];
        for (j, row) in layer.w.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                w[i * n_out + j] = v;
            }
        }
        Ok(Dense { n_in, n_out, w, b: layer.b.clone() })
    }

    #[inline]
    fn add_col(&self, i: usize, x: f32, acc: &mut [f32]) {
        let col = &self.w[i * self.n_out..(i + 1) * self.n_out];
        for (a, w) in acc.iter_mut().zip(col) {
            *a += w * x;
        }
    }

    #[inline]
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        out.copy_from_slice(&self.b);
        for (i, &xi) in x.iter().enumerate() {
            self.add_col(i, xi, out);
        }
    }
}

/// Linear → SiLU → Linear → SiLU → Linear.
struct Mlp {
    l1: Dense,
    l2: Dense,
    l3: Dense,
}

impl Mlp {
    fn new(layers: &[LayerJson]) -> Result<Mlp, String> {
        let [a, b, c] = layers else { return Err("expected three layers".into()) };
        let mlp = Mlp { l1: Dense::new(a)?, l2: Dense::new(b)?, l3: Dense::new(c)? };
        if mlp.l1.n_out > MAX_HIDDEN || mlp.l2.n_out > MAX_HIDDEN || mlp.l2.n_in != mlp.l1.n_out
            || mlp.l3.n_in != mlp.l2.n_out {
            return Err("unsupported layer sizes".into());
        }
        Ok(mlp)
    }

    fn hidden(&self) -> usize {
        self.l1.n_out
    }

    /// From the first layer's pre-activation `h1` (overwritten) to the outputs.
    #[inline]
    fn tail(&self, h1: &mut [f32], out: &mut [f32]) {
        let mut h2 = [0.0f32; MAX_HIDDEN];
        let h2 = &mut h2[..self.l2.n_out];
        silu(h1);
        self.l2.apply(h1, h2);
        silu(h2);
        self.l3.apply(h2, out);
    }

    fn forward(&self, x: &[f32], out: &mut [f32]) {
        let mut h1 = [0.0f32; MAX_HIDDEN];
        let h1 = &mut h1[..self.l1.n_out];
        self.l1.apply(x, h1);
        self.tail(h1, out);
    }

    fn scalar(&self, x: &[f32]) -> f32 {
        let mut out = [0.0f32];
        self.forward(x, &mut out);
        out[0]
    }
}

/// `eˣ` to about 1e-7 relative error, without branches or library calls so that loops over
/// it vectorise (on every target, including WebAssembly). The activations are where the
/// simulator spends its time: the platform `expf` made the whole run 20× slower on Windows.
/// Cephes' `expf`: `x = k·ln 2 + r`, a degree-5 polynomial in `r`, then scale by `2ᵏ`.
#[inline(always)]
#[allow(clippy::excessive_precision)] // Cephes' published coefficients
fn fast_exp(x: f32) -> f32 {
    const ROUND: f32 = 12_582_912.0; // 1.5 · 2²³: adding and subtracting rounds to nearest
    let x = x.clamp(-87.0, 88.0);
    let k = (x * std::f32::consts::LOG2_E + ROUND) - ROUND;
    let r = x - k * 0.693_359_375 - k * -2.121_944_4e-4;
    let p = ((((1.987_569_15e-4 * r + 1.398_199_95e-3) * r + 8.333_451_9e-3) * r
        + 4.166_579_6e-2) * r + 1.666_666_55e-1) * r + 0.5;
    let e = p * (r * r) + r + 1.0;
    // 2ᵏ without a float-to-int conversion (which stays scalar on SSE2): between 2²³ and 2²⁴
    // a float's mantissa bits are its integer part, so k + 127 can be shifted into the exponent
    const BIAS: f32 = 127.0 + 8_388_608.0;
    e * f32::from_bits((k + BIAS).to_bits() << 23)
}

#[inline]
fn silu(h: &mut [f32]) {
    for x in h.iter_mut() {
        *x *= 1.0 / (1.0 + fast_exp(-*x));
    }
}

/// `tanh x` from [`fast_exp`]; absolute error about 1e-7.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let e = fast_exp(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

/// `log(1 + eˣ)`, linear above 20 like torch.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// One input row, already standardised where the network expects it.
#[derive(Clone, Debug)]
pub struct Row {
    pub circuit: usize,
    pub compound: usize,
    pub context: [f32; N_CONTEXT],
    pub tyre_life: f32,
    pub laps_remaining: f32,
    pub fresh: f32,
    pub gap: f32,
    pub leader: f32,
    pub fuel_frac: f32,
    pub hardness: f32,
}

struct Member {
    circuit_emb: Vec<Vec<f32>>,
    compound_emb: Vec<Vec<f32>>,
    tau: f32,
    base: Mlp,
    conditions: Mlp,
    fuel: Dense,
    offset: Mlp,
    deg: Mlp,
    traffic: Mlp,
    sigma: Mlp,
    interaction: Option<Mlp>,
}

impl Member {
    fn new(m: &MemberJson) -> Result<Member, String> {
        let [fuel] = m.fuel.as_slice() else { return Err("fuel must be one layer".into()) };
        let member = Member {
            circuit_emb: m.circuit_emb.clone(),
            compound_emb: m.compound_emb.clone(),
            tau: m.tau,
            base: Mlp::new(&m.base)?,
            conditions: Mlp::new(&m.conditions)?,
            fuel: Dense::new(fuel)?,
            offset: Mlp::new(&m.offset)?,
            deg: Mlp::new(&m.deg)?,
            traffic: Mlp::new(&m.traffic)?,
            sigma: Mlp::new(&m.sigma)?,
            interaction: m.interaction.as_deref().map(Mlp::new).transpose()?,
        };
        let e = member.emb();
        let d = member.compound_emb.first().map_or(0, Vec::len);
        let sizes_ok = member.circuit_emb.iter().all(|v| v.len() == e)
            && member.compound_emb.len() == COMPOUNDS.len()
            && member.base.l1.n_in == e + RACE_CONTEXT.len()
            && member.conditions.l1.n_in == 1
            && member.fuel.n_in == e
            && member.offset.l1.n_in == e + 1
            && member.offset.l3.n_out == 1 + NH
            && member.deg.l1.n_in == e + 1
            && member.deg.l3.n_out == (1 + NH) * NB
            && member.traffic.l1.n_in == e + 3
            && member.sigma.l1.n_in == e + d + 7
            && member.interaction.as_ref().is_none_or(|i| i.l1.n_in == e + RACE_CONTEXT.len() + 4);
        if !sizes_ok || e == 0 {
            return Err("network shape does not match lap_net".into());
        }
        Ok(member)
    }

    fn emb(&self) -> usize {
        self.circuit_emb.first().map_or(0, Vec::len)
    }

    /// `[age, τ·softplus((age − k)/τ) …] / 10`: non-decreasing and convex in tyre age.
    fn age_basis(&self, age: f32) -> [f32; NB] {
        let mut b = [age / 10.0; NB];
        for (i, k) in KNOTS.iter().enumerate() {
            b[i + 1] = self.tau * softplus((age - k) / self.tau) / 10.0;
        }
        b
    }

    /// Fresh-tyre offset and degradation weights for a compound of this hardness.
    fn tyre_terms(&self, ec: &[f32], fresh: f32, hardness: f32) -> (f32, [f32; NB]) {
        let mut ctx = ec.to_vec();
        ctx.push(fresh);
        let mut psi = [hardness - 1.0; NH];
        for (i, k) in HARDNESS_KNOTS.iter().enumerate() {
            psi[i + 1] = HARDNESS_SCALE * softplus((hardness - k) / HARDNESS_SCALE);
        }
        let mut o = [0.0f32; 1 + NH];
        self.offset.forward(&ctx, &mut o);
        let mut gain = 0.0f32;
        for j in 0..NH {
            gain += softplus(o[1 + j]) * psi[j];
        }
        let mut w = [0.0f32; (1 + NH) * NB];
        self.deg.forward(&ctx, &mut w);
        let mut weights = [0.0f32; NB];
        for (b, weight) in weights.iter_mut().enumerate() {
            let mut extra = 0.0f32;
            for j in 0..NH {
                extra += softplus(w[(1 + j) * NB + b]) * psi[j];
            }
            *weight = softplus(w[b]) + extra;
        }
        (o[0] - gain, weights)
    }

    fn race_context(context: &[f32; N_CONTEXT]) -> [f32; 4] {
        RACE_CONTEXT.map(|i| context[i])
    }

    /// Mean and σ (percent of the qualifying lap) for one row, as the torch module computes them.
    fn forward(&self, r: &Row) -> (f32, f32) {
        let ec = &self.circuit_emb[r.circuit];
        let ek = &self.compound_emb[r.compound];
        let race_ctx = Self::race_context(&r.context);
        let near = (-r.gap).exp();

        let base = self.base.scalar(&[ec.as_slice(), &race_ctx].concat());
        let conditions = self.conditions.scalar(&[r.context[TEMP]]);
        let mut rate = [0.0f32];
        self.fuel.apply(ec, &mut rate);
        let fuel = softplus(rate[0]) * r.laps_remaining / 10.0;
        let (offset, weights) = self.tyre_terms(ec, r.fresh, r.hardness);
        let basis = self.age_basis(r.tyre_life);
        let deg: f32 = weights.iter().zip(&basis).map(|(w, b)| w * b).sum();
        let traffic = self.traffic.scalar(&[ec.as_slice(), &[near, r.gap / 10.0, r.leader]].concat());
        let interaction = match &self.interaction {
            Some(net) => {
                let x = [ec.as_slice(), &race_ctx, &[r.fuel_frac, near, r.leader, r.fresh]].concat();
                INTERACTION_BOUND * net.scalar(&x).tanh()
            }
            None => 0.0,
        };
        let sig_in = [ec.as_slice(), ek.as_slice(),
                      &[r.hardness / 6.0, r.tyre_life / 30.0, near, r.leader, r.fuel_frac,
                        r.context[0], r.context[1]]].concat();
        let sigma = softplus(self.sigma.scalar(&sig_in)) + SIGMA_FLOOR;
        (base + conditions + fuel + offset + deg + traffic + interaction, sigma)
    }
}

/// Ensemble mean and σ: a Gaussian mixture's mean and total standard deviation.
#[inline]
fn mixture(parts: &[(f32, f32)]) -> (f32, f32) {
    let n = parts.len() as f32;
    let mu = parts.iter().map(|p| p.0).sum::<f32>() / n;
    let second = parts.iter().map(|p| p.1 * p.1 + p.0 * p.0).sum::<f32>() / n;
    (mu, (second - mu * mu).max(SIGMA_FLOOR * SIGMA_FLOOR).sqrt())
}

/// The lap-time network: an ensemble of members plus the input standardisation.
pub struct LapNet {
    circuit_index: HashMap<String, usize>,
    context_mean: [f64; N_CONTEXT],
    context_std: [f64; N_CONTEXT],
    members: Vec<Member>,
}

impl LapNet {
    /// Parse the JSON written by `strategist.sim.export.model_dict`.
    pub fn from_json(json: &str) -> Result<LapNet, String> {
        let model: ModelJson = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let members = model.members.iter().map(Member::new).collect::<Result<Vec<_>, _>>()?;
        let fixed = |v: Vec<f64>| <[f64; N_CONTEXT]>::try_from(v).map_err(|_| "context size".to_string());
        if members.is_empty() {
            return Err("model has no ensemble members".into());
        }
        Ok(LapNet {
            circuit_index: model.circuit_index,
            context_mean: fixed(model.context_mean)?,
            context_std: fixed(model.context_std)?,
            members,
        })
    }

    pub fn members(&self) -> usize {
        self.members.len()
    }

    /// Embedding row for a circuit; 0 is the "unknown circuit" embedding.
    pub fn circuit(&self, name: &str) -> usize {
        self.circuit_index.get(name).copied().unwrap_or(0)
    }

    /// Standardise raw context values (missing values become the mean, i.e. 0).
    pub fn context(&self, raw: [f64; N_CONTEXT]) -> [f32; N_CONTEXT] {
        let mut out = [0.0f32; N_CONTEXT];
        for i in 0..N_CONTEXT {
            let z = (raw[i] - self.context_mean[i]) / self.context_std[i];
            out[i] = if z.is_nan() { 0.0 } else { z as f32 };
        }
        out
    }

    fn take(&self, members: Option<usize>) -> &[Member] {
        &self.members[..members.map_or(self.members.len(), |n| n.clamp(1, self.members.len()))]
    }

    /// Mixture mean and σ in percent of the qualifying lap, from the first `members` members.
    pub fn predict(&self, row: &Row, members: Option<usize>) -> (f32, f32) {
        let parts: Vec<(f32, f32)> = self.take(members).iter().map(|m| m.forward(row)).collect();
        mixture(&parts)
    }
}

/// What one car's tyres look like on one lap.
#[derive(Clone, Copy, Debug)]
pub struct TyreState {
    pub car: usize,
    pub compound: usize,
    pub age: f32,
    pub fresh: f32,
}

struct MemberRace<'a> {
    m: &'a Member,
    base: Vec<f32>,                    // per car: base + conditions
    fuel_rate: f32,
    tyre: Vec<[(f32, [f32; NB]); 2]>,  // [compound][fresh]: offset, degradation weights
    traffic_pre: Vec<f32>,             // first-layer pre-activations with the fixed inputs in
    interaction_pre: Vec<f32>,         // per car
    sigma_pre: Vec<f32>,               // per car
    sigma_compound: Vec<f32>,          // per compound: embedding and hardness columns
    // per lap and slot, set by `RaceNet::set_lap`
    mu_static: Vec<f32>,
    interaction_lap: Vec<f32>,
    sigma_lap: Vec<f32>,
}

/// The network specialised to one race: circuit, cars and compound hardness are fixed.
pub struct RaceNet<'a> {
    members: Vec<MemberRace<'a>>,
    emb: usize,
    compound_dim: usize,
}

impl<'a> RaceNet<'a> {
    /// `contexts`: standardised context per car. `hardness`: per compound in `COMPOUNDS` order.
    pub fn new(net: &'a LapNet, members: Option<usize>, circuit: &str,
               contexts: &[[f32; N_CONTEXT]], hardness: [f32; 3]) -> RaceNet<'a> {
        let ci = net.circuit(circuit);
        let take = net.take(members);
        let emb = take[0].emb();
        let compound_dim = take[0].compound_emb[0].len();
        let members = take.iter().map(|m| {
            let ec = &m.circuit_emb[ci];
            let mut rate = [0.0f32];
            m.fuel.apply(ec, &mut rate);

            let fixed = |mlp: &Mlp| {
                let mut pre = mlp.l1.b.clone();
                for (i, &x) in ec.iter().enumerate() {
                    mlp.l1.add_col(i, x, &mut pre);
                }
                pre
            };
            let traffic_pre = fixed(&m.traffic);
            let mut base = Vec::with_capacity(contexts.len());
            let mut interaction_pre = Vec::new();
            let mut sigma_pre = Vec::new();
            for ctx in contexts {
                let race_ctx = Member::race_context(ctx);
                base.push(m.base.scalar(&[ec.as_slice(), &race_ctx].concat())
                          + m.conditions.scalar(&[ctx[TEMP]]));
                if let Some(net) = &m.interaction {
                    let mut pre = fixed(net);
                    for (i, &x) in race_ctx.iter().enumerate() {
                        net.l1.add_col(emb + i, x, &mut pre);
                    }
                    interaction_pre.extend(pre);
                }
                let mut pre = fixed(&m.sigma);
                m.sigma.l1.add_col(emb + compound_dim + 5, ctx[0], &mut pre);
                m.sigma.l1.add_col(emb + compound_dim + 6, ctx[1], &mut pre);
                sigma_pre.extend(pre);
            }
            let mut tyre = Vec::new();
            let mut sigma_compound = Vec::new();
            for (k, &h) in hardness.iter().enumerate() {
                tyre.push([m.tyre_terms(ec, 0.0, h), m.tyre_terms(ec, 1.0, h)]);
                let mut cols = vec![0.0f32; m.sigma.hidden()];
                for (i, &x) in m.compound_emb[k].iter().enumerate() {
                    m.sigma.l1.add_col(emb + i, x, &mut cols);
                }
                m.sigma.l1.add_col(emb + compound_dim, h / 6.0, &mut cols);
                sigma_compound.extend(cols);
            }
            MemberRace { m, base, fuel_rate: softplus(rate[0]), tyre, traffic_pre, interaction_pre,
                         sigma_pre, sigma_compound, mu_static: Vec::new(),
                         interaction_lap: Vec::new(), sigma_lap: Vec::new() }
        }).collect();
        RaceNet { members, emb, compound_dim }
    }

    /// Fix the inputs that every run shares on this lap: one tyre state per slot.
    pub fn set_lap(&mut self, slots: &[TyreState], laps_remaining: f32, fuel_frac: f32) {
        let (e, d) = (self.emb, self.compound_dim);
        for mr in &mut self.members {
            let m = mr.m;
            let hs = m.sigma.hidden();
            mr.mu_static.clear();
            mr.interaction_lap.clear();
            mr.sigma_lap.clear();
            for s in slots {
                let (offset, weights) = mr.tyre[s.compound][usize::from(s.fresh != 0.0)];
                let basis = m.age_basis(s.age);
                let deg: f32 = weights.iter().zip(&basis).map(|(w, b)| w * b).sum();
                let fuel = mr.fuel_rate * laps_remaining / 10.0;
                mr.mu_static.push(mr.base[s.car] + fuel + offset + deg);

                if let Some(net) = &m.interaction {
                    let hi = net.hidden();
                    let at = mr.interaction_lap.len();
                    mr.interaction_lap.extend_from_slice(&mr.interaction_pre[s.car * hi..(s.car + 1) * hi]);
                    let pre = &mut mr.interaction_lap[at..];
                    net.l1.add_col(e + 4, fuel_frac, pre);
                    net.l1.add_col(e + 7, s.fresh, pre);
                }
                let at = mr.sigma_lap.len();
                mr.sigma_lap.extend_from_slice(&mr.sigma_pre[s.car * hs..(s.car + 1) * hs]);
                let pre = &mut mr.sigma_lap[at..];
                for (p, c) in pre.iter_mut().zip(&mr.sigma_compound[s.compound * hs..(s.compound + 1) * hs]) {
                    *p += c;
                }
                m.sigma.l1.add_col(e + d + 1, s.age / 30.0, pre);
                m.sigma.l1.add_col(e + d + 4, fuel_frac, pre);
            }
        }
    }

    /// Mixture mean and σ (percent of the qualifying lap) for a slot, given the gap ahead.
    #[inline]
    pub fn predict(&self, slot: usize, gap: f32, leader: bool) -> (f32, f32) {
        let (e, d) = (self.emb, self.compound_dim);
        let near = fast_exp(-gap);
        let lead = if leader { 1.0 } else { 0.0 };
        let mut parts = [(0.0f32, 0.0f32); 16];
        let n = self.members.len().min(parts.len());
        let mut h = [0.0f32; MAX_HIDDEN];
        let mut out = [0.0f32];
        for (mr, part) in self.members.iter().zip(parts.iter_mut()) {
            let m = mr.m;

            let ht = &mut h[..m.traffic.hidden()];
            ht.copy_from_slice(&mr.traffic_pre);
            m.traffic.l1.add_col(e, near, ht);
            m.traffic.l1.add_col(e + 1, gap / 10.0, ht);
            m.traffic.l1.add_col(e + 2, lead, ht);
            m.traffic.tail(ht, &mut out);
            let traffic = out[0];

            let interaction = match &m.interaction {
                Some(net) => {
                    let hi = net.hidden();
                    let hx = &mut h[..hi];
                    hx.copy_from_slice(&mr.interaction_lap[slot * hi..(slot + 1) * hi]);
                    net.l1.add_col(e + 5, near, hx);
                    net.l1.add_col(e + 6, lead, hx);
                    net.tail(hx, &mut out);
                    INTERACTION_BOUND * fast_tanh(out[0])
                }
                None => 0.0,
            };

            let hs = m.sigma.hidden();
            let hx = &mut h[..hs];
            hx.copy_from_slice(&mr.sigma_lap[slot * hs..(slot + 1) * hs]);
            m.sigma.l1.add_col(e + d + 2, near, hx);
            m.sigma.l1.add_col(e + d + 3, lead, hx);
            m.sigma.tail(hx, &mut out);

            *part = (mr.mu_static[slot] + traffic + interaction, softplus(out[0]) + SIGMA_FLOOR);
        }
        mixture(&parts[..n])
    }
}

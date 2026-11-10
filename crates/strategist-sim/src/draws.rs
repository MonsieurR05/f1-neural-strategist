//! Random numbers for the simulator, behind three primitives.
//!
//! The simulator asks only for standard normals, uniforms on [0, 1) and a pick from a list,
//! in the same order as the Python simulator. [`Rng`] generates them; [`Replay`] plays back
//! the draws a Python run recorded (`strategist.sim.export`), which is how the parity tests
//! make both simulators see identical luck.

pub trait Draws {
    /// Fill with standard normal draws.
    fn normal(&mut self, out: &mut [f64]);
    /// Fill with uniform draws on [0, 1).
    fn uniform(&mut self, out: &mut [f64]);
    /// One element of `options`, uniformly.
    fn choice(&mut self, options: &[i64]) -> i64;
}

/// xoshiro256++ seeded through SplitMix64; normals by the polar Box–Muller method.
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()], spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn gauss(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let (u, v) = (2.0 * self.unit() - 1.0, 2.0 * self.unit() - 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }
}

impl Draws for Rng {
    fn normal(&mut self, out: &mut [f64]) {
        out.iter_mut().for_each(|x| *x = self.gauss());
    }

    fn uniform(&mut self, out: &mut [f64]) {
        out.iter_mut().for_each(|x| *x = self.unit());
    }

    fn choice(&mut self, options: &[i64]) -> i64 {
        options[((self.unit() * options.len() as f64) as usize).min(options.len() - 1)]
    }
}

/// Plays back recorded draws. Panics if the simulator asks for more than were recorded.
pub struct Replay {
    normal: Vec<f64>,
    uniform: Vec<f64>,
    choice: Vec<i64>,
    at: [usize; 3],
}

impl Replay {
    pub fn new(normal: Vec<f64>, uniform: Vec<f64>, choice: Vec<i64>) -> Replay {
        Replay { normal, uniform, choice, at: [0; 3] }
    }

    /// True when every recorded draw has been used: the draw sequences match in length.
    pub fn exhausted(&self) -> bool {
        self.at == [self.normal.len(), self.uniform.len(), self.choice.len()]
    }
}

impl Draws for Replay {
    fn normal(&mut self, out: &mut [f64]) {
        out.copy_from_slice(&self.normal[self.at[0]..self.at[0] + out.len()]);
        self.at[0] += out.len();
    }

    fn uniform(&mut self, out: &mut [f64]) {
        out.copy_from_slice(&self.uniform[self.at[1]..self.at[1] + out.len()]);
        self.at[1] += out.len();
    }

    fn choice(&mut self, options: &[i64]) -> i64 {
        let picked = self.choice[self.at[2]];
        assert!(options.contains(&picked), "recorded choice {picked} is not an option");
        self.at[2] += 1;
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_reproducible_and_roughly_standard() {
        let mut a = vec![0.0; 20_000];
        let mut b = vec![0.0; 20_000];
        Rng::new(7).normal(&mut a);
        Rng::new(7).normal(&mut b);
        assert_eq!(a, b);
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "mean {mean}, var {var}");

        let mut u = vec![0.0; 20_000];
        Rng::new(7).uniform(&mut u);
        assert!(u.iter().all(|x| (0.0..1.0).contains(x)));
        assert!((u.iter().sum::<f64>() / u.len() as f64 - 0.5).abs() < 0.02);
    }
}

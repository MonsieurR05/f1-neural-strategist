//! Rust port of the F1 Neural Race Strategist's lap-time network and race simulator.
//!
//! The Python package (`packages/strategist`) is the reference implementation: it trains the
//! models and produces every published result. This crate reproduces its simulator so the
//! same races can be simulated where Python can't run, such as in the browser.
//! `tests/parity.rs` holds it to the Python outputs.

pub mod draws;
pub mod engine;
pub mod model;
pub mod sim;
pub mod summary;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use draws::{Draws, Replay, Rng};
pub use engine::Engine;
pub use model::{LapNet, Row};
pub use sim::{simulate, Car, RaceSetup, SimOptions, SimParams, SimResult, Status, Stint};
pub use summary::{FinishRow, LapBands};

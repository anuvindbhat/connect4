//! # Connect 4 AI Profiling Tool
//!
//! A minimal profiling tool designed to exercise the AI engine's search function
//! with a high depth. This is intended to be used with external profiling tools
//! (like `perf`, `flamegraph`, or `VTune`) to identify performance bottlenecks
//! in the search and evaluation logic.

use connect4::Player;
use connect4::config::HeuristicWeights;
use connect4::engine::DynamicEngine;
use connect4::game::{DynamicBoardGeometry, DynamicBoardState};

/// Entry point for the AI profiling tool.
///
/// Runs a series of high-depth searches from the starting position to
/// provide a consistent workload for performance profiling.
fn main() {
    let geo = DynamicBoardGeometry::new(7, 6);
    let state = DynamicBoardState::new(&geo);
    let mut engine = DynamicEngine::new(geo, HeuristicWeights::default(), Some(64));
    for _ in 0..10 {
        engine.reset_tt();
        let _ = engine.find_best_move(&state, Player::Red, 11, false);
    }
}

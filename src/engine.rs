//! High-level search engine abstraction for the Connect 4 AI.
//!
//! This module provides the core `Engine` and `DynamicEngine` types that orchestrate
//! the AI search process. It manages the lifecycle of transposition tables (TT),
//! handles bitboard variant selection (u64 vs u128), and provides a clean interface
//! for finding best moves and evaluating positions.
//!
//! The engine is designed to be efficient by reusing transposition tables where possible
//! via a pooling mechanism, avoiding expensive reallocations between moves or games.

use crate::ai;
use crate::config::HeuristicWeights;
use crate::game::{BoardGeometry, BoardState, DynamicBoardGeometry, DynamicBoardState};
use crate::tt::{TTGuard, TTManaged, TranspositionTable};
use crate::types::{Bitboard, Player};

/// Controls how the Transposition Table (TT) is managed and owned within an `Engine`.
///
/// This enum allows the engine to either own its own dedicated table or use a pooled table
/// managed by thread-local storage. Using a pooled table is highly recommended for multi-threaded
/// searches or short-lived engine instances to avoid the high cost of allocating and clearing
/// large tables.
pub enum TTHolder<T: Bitboard + TTManaged> {
    /// The engine owns a dedicated transposition table.
    /// This table is allocated specifically for this engine and is dropped when the engine is dropped.
    Owned(TranspositionTable<T>),
    /// The engine uses a pooled transposition table.
    /// The table is wrapped in a `TTGuard`, which will automatically return it to the
    /// type-specific thread-local pool when the engine is dropped.
    Pooled(TTGuard<T>),
    /// No transposition table is used.
    /// Searching without a TT is significantly slower and should only be used for testing.
    None,
}

impl<T: Bitboard + TTManaged> TTHolder<T> {
    /// Returns a mutable reference to the underlying transposition table, if one exists.
    pub fn as_mut(&mut self) -> Option<&mut TranspositionTable<T>> {
        match self {
            Self::Owned(tt) => Some(tt),
            Self::Pooled(tt) => Some(tt),
            Self::None => None,
        }
    }

    /// Resets the underlying transposition table by incrementing its generation.
    /// This effectively clears the table in O(1) time.
    pub fn reset(&mut self) {
        if let Some(tt) = self.as_mut() {
            tt.reset();
        }
    }
}

/// A high-performance search engine specialized for a specific bitboard type `T`.
///
/// An `Engine` instance encapsulates the board geometry, heuristic weights, and
/// transposition table state required to perform deep searches. It is decoupled
/// from the `BoardState` itself, allowing the same engine to be used for multiple
/// different positions on boards of the same size.
pub struct Engine<T: Bitboard + TTManaged> {
    /// The geometry (width, height, etc.) of the boards this engine is configured for.
    geometry: BoardGeometry<T>,
    /// The heuristic weights used to evaluate non-terminal board positions.
    weights: HeuristicWeights,
    /// The transposition table holder, managing the memory and lifecycle of search results.
    tt: TTHolder<T>,
}

impl<T: Bitboard + TTManaged> Engine<T> {
    /// Creates a new `Engine` with the given geometry, heuristic weights, and transposition table holder.
    pub fn new(geometry: BoardGeometry<T>, weights: HeuristicWeights, tt: TTHolder<T>) -> Self {
        Self {
            geometry,
            weights,
            tt,
        }
    }

    /// Resets the Transposition Table.
    ///
    /// This is useful between games to ensure no stale search results interfere with new searches.
    pub fn reset_tt(&mut self) {
        self.tt.reset();
    }

    /// Finds the best move for the current state.
    ///
    /// # Arguments
    ///
    /// * `state` - The current board position.
    /// * `curr_p` - The player whose turn it is to move.
    /// * `depth` - The maximum search depth.
    /// * `randomize` - If true, select moves using Boltzmann selection.
    ///
    /// # Returns
    ///
    /// The column index of the best move, or `None` if no moves are possible.
    pub fn find_best_move(
        &mut self,
        state: &BoardState<T>,
        curr_p: Player,
        depth: u32,
        randomize: bool,
    ) -> Option<u32> {
        ai::find_best_move(
            state,
            &self.geometry,
            curr_p,
            depth,
            randomize,
            &self.weights,
            self.tt.as_mut(),
        )
    }

    /// Finds the best move for the current state with detailed results.
    ///
    /// Similar to `find_best_move`, but returns a `SearchReport` containing additional
    /// information such as the number of nodes visited and TT statistics.
    pub fn find_best_move_detailed(
        &mut self,
        state: &BoardState<T>,
        curr_p: Player,
        depth: u32,
        randomize: bool,
    ) -> ai::SearchReport {
        ai::find_best_move_detailed(
            state,
            &self.geometry,
            curr_p,
            depth,
            randomize,
            &self.weights,
            self.tt.as_mut(),
        )
    }

    /// Returns scores for all possible moves in the current state.
    ///
    /// This performs a search for each legal move and returns a vector where each element
    /// is the score for the corresponding column, or `None` if the column is full.
    pub fn get_column_scores(
        &mut self,
        state: &BoardState<T>,
        curr_p: Player,
        depth: u32,
    ) -> Vec<Option<i32>> {
        ai::get_column_scores(
            state,
            &self.geometry,
            curr_p,
            depth,
            &self.weights,
            self.tt.as_mut(),
        )
    }

    /// Evaluates the current position.
    ///
    /// This performs a search up to the specified depth and returns the evaluation
    /// score from the perspective of the current player.
    pub fn evaluate_position(&mut self, state: &BoardState<T>, curr_p: Player, depth: u32) -> i32 {
        ai::evaluate_position(
            state,
            &self.geometry,
            curr_p,
            depth,
            &self.weights,
            self.tt.as_mut(),
        )
    }
}

/// Dynamic wrapper for the Search Engine, choosing between u64 and u128 variants.
///
/// `DynamicEngine` provides a uniform interface for performing AI operations regardless
/// of the board size. It internally dispatches to either a `Small` (u64) or `Large` (u128)
/// engine variant based on the board's dimensions.
pub enum DynamicEngine {
    /// Variant for small boards (up to 7x9 or 8x8) using 64-bit bitboards.
    Small(Engine<u64>),
    /// Variant for large boards (up to 12x10) using 128-bit bitboards.
    Large(Engine<u128>),
}

impl DynamicEngine {
    /// Creates a new `DynamicEngine` with the given geometry and weights.
    ///
    /// If `tt_size_mb` is provided, it will use a pooled transposition table of that size.
    /// Otherwise, it will not use a transposition table.
    #[must_use]
    pub fn new(
        geometry: DynamicBoardGeometry,
        weights: HeuristicWeights,
        tt_size_mb: Option<usize>,
    ) -> Self {
        match geometry {
            DynamicBoardGeometry::Small(g) => {
                let tt = tt_size_mb
                    .map(TranspositionTable::get_pooled)
                    .map_or(TTHolder::None, TTHolder::Pooled);
                Self::Small(Engine::new(g, weights, tt))
            }
            DynamicBoardGeometry::Large(g) => {
                let tt = tt_size_mb
                    .map(TranspositionTable::get_pooled)
                    .map_or(TTHolder::None, TTHolder::Pooled);
                Self::Large(Engine::new(g, weights, tt))
            }
        }
    }

    /// Creates a new `DynamicEngine` with an owned transposition table of the given size.
    ///
    /// Unlike `new`, this always allocates a dedicated transposition table for the engine.
    #[must_use]
    pub fn new_with_owned_tt(
        geometry: DynamicBoardGeometry,
        weights: HeuristicWeights,
        tt_size_mb: usize,
    ) -> Self {
        match geometry {
            DynamicBoardGeometry::Small(g) => {
                let tt = TTHolder::Owned(TranspositionTable::new(tt_size_mb));
                Self::Small(Engine::new(g, weights, tt))
            }
            DynamicBoardGeometry::Large(g) => {
                let tt = TTHolder::Owned(TranspositionTable::new(tt_size_mb));
                Self::Large(Engine::new(g, weights, tt))
            }
        }
    }

    /// Resets the Transposition Table.
    pub fn reset_tt(&mut self) {
        match self {
            Self::Small(e) => e.reset_tt(),
            Self::Large(e) => e.reset_tt(),
        }
    }

    /// Finds the best move for the current state.
    ///
    /// # Panics
    ///
    /// Panics if the engine variant doesn't match the board state variant.
    pub fn find_best_move(
        &mut self,
        state: &DynamicBoardState,
        curr_p: Player,
        depth: u32,
        randomize: bool,
    ) -> Option<u32> {
        match (self, state) {
            (Self::Small(e), DynamicBoardState::Small(s)) => {
                e.find_best_move(s, curr_p, depth, randomize)
            }
            (Self::Large(e), DynamicBoardState::Large(s)) => {
                e.find_best_move(s, curr_p, depth, randomize)
            }
            _ => panic!("Engine and Board variant mismatch"),
        }
    }

    /// Finds the best move for the current state with detailed results.
    ///
    /// # Panics
    ///
    /// Panics if the engine variant doesn't match the board state variant.
    pub fn find_best_move_detailed(
        &mut self,
        state: &DynamicBoardState,
        curr_p: Player,
        depth: u32,
        randomize: bool,
    ) -> ai::SearchReport {
        match (self, state) {
            (Self::Small(e), DynamicBoardState::Small(s)) => {
                e.find_best_move_detailed(s, curr_p, depth, randomize)
            }
            (Self::Large(e), DynamicBoardState::Large(s)) => {
                e.find_best_move_detailed(s, curr_p, depth, randomize)
            }
            _ => panic!("Engine and Board variant mismatch"),
        }
    }

    /// Returns scores for all possible moves in the current state.
    ///
    /// # Panics
    ///
    /// Panics if the engine variant doesn't match the board state variant.
    pub fn get_column_scores(
        &mut self,
        state: &DynamicBoardState,
        curr_p: Player,
        depth: u32,
    ) -> Vec<Option<i32>> {
        match (self, state) {
            (Self::Small(e), DynamicBoardState::Small(s)) => e.get_column_scores(s, curr_p, depth),
            (Self::Large(e), DynamicBoardState::Large(s)) => e.get_column_scores(s, curr_p, depth),
            _ => panic!("Engine and Board variant mismatch"),
        }
    }

    /// Evaluates the current position.
    ///
    /// # Panics
    ///
    /// Panics if the engine variant doesn't match the board state variant.
    pub fn evaluate_position(
        &mut self,
        state: &DynamicBoardState,
        curr_p: Player,
        depth: u32,
    ) -> i32 {
        match (self, state) {
            (Self::Small(e), DynamicBoardState::Small(s)) => e.evaluate_position(s, curr_p, depth),
            (Self::Large(e), DynamicBoardState::Large(s)) => e.evaluate_position(s, curr_p, depth),
            _ => panic!("Engine and Board variant mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::DynamicBoardGeometry;

    #[test]
    fn test_tt_holder_owned() {
        let tt = TranspositionTable::<u64>::new(1);
        let mut holder = TTHolder::Owned(tt);
        assert!(holder.as_mut().is_some());
        holder.reset();
    }

    #[test]
    fn test_tt_holder_pooled() {
        let tt = TranspositionTable::<u64>::get_pooled(1);
        let mut holder = TTHolder::Pooled(tt);
        assert!(holder.as_mut().is_some());
        holder.reset();
    }

    #[test]
    fn test_tt_holder_none() {
        let mut holder = TTHolder::<u64>::None;
        assert!(holder.as_mut().is_none());
        holder.reset(); // Should not panic
    }

    #[test]
    fn test_dynamic_engine_selection() {
        let weights = HeuristicWeights::default();

        // Standard 7x6 (49 bits) should be Small
        let geo_small = DynamicBoardGeometry::new(7, 6);
        let engine_small = DynamicEngine::new(geo_small, weights, None);
        assert!(matches!(engine_small, DynamicEngine::Small(_)));

        // Giant 9x7 (72 bits) should be Large
        let geo_large = DynamicBoardGeometry::new(9, 7);
        let engine_large = DynamicEngine::new(geo_large, weights, None);
        assert!(matches!(engine_large, DynamicEngine::Large(_)));
    }

    #[test]
    fn test_dynamic_engine_owned_tt() {
        let weights = HeuristicWeights::default();
        let geo = DynamicBoardGeometry::new(7, 6);
        let _ = DynamicEngine::new_with_owned_tt(geo, weights, 1);
    }

    #[test]
    fn test_dynamic_engine_reset_tt() {
        let weights = HeuristicWeights::default();
        let geo = DynamicBoardGeometry::new(7, 6);
        let mut engine = DynamicEngine::new(geo, weights, Some(1));
        engine.reset_tt();
    }

    #[test]
    fn test_dynamic_engine_find_best_move() {
        let weights = HeuristicWeights::default();
        let geo = DynamicBoardGeometry::new(7, 6);
        let mut engine = DynamicEngine::new(geo.clone(), weights, None);
        let state = DynamicBoardState::new(&geo);

        let m = engine.find_best_move(&state, Player::Red, 2, false);
        assert!(m.is_some());

        let report = engine.find_best_move_detailed(&state, Player::Red, 2, false);
        assert!(report.best_move.is_some());
    }

    #[test]
    fn test_dynamic_engine_evaluate() {
        let weights = HeuristicWeights::default();
        let geo = DynamicBoardGeometry::new(7, 6);
        let mut engine = DynamicEngine::new(geo.clone(), weights, None);
        let state = DynamicBoardState::new(&geo);

        // At depth 5, Red (first player) should have a positive evaluation
        // reflecting the center-control advantage.
        let score = engine.evaluate_position(&state, Player::Red, 5);
        assert!(
            score > 0,
            "Empty board should have positive eval for first player at depth 5, got {score}"
        );

        let scores = engine.get_column_scores(&state, Player::Red, 2);
        assert_eq!(scores.len(), 7);
    }

    #[test]
    #[should_panic(expected = "Engine and Board variant mismatch")]
    fn test_dynamic_engine_variant_mismatch_panic() {
        let weights = HeuristicWeights::default();

        // Engine is Small (u64)
        let geo_small = DynamicBoardGeometry::new(7, 6);
        let mut engine = DynamicEngine::new(geo_small, weights, None);

        // State is Large (u128)
        let geo_large = DynamicBoardGeometry::new(9, 7);
        let state_large = DynamicBoardState::new(&geo_large);

        // This should panic
        let _ = engine.find_best_move(&state_large, Player::Red, 1, false);
    }
}

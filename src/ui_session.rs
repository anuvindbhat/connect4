//! # UI Session Management
//!
//! This module provides the `UiGameSession` struct, which acts as a View-Model for the
//! TUI application. It wraps the core `GameSession` and manages cosmetic state such as
//! piece falling animations and current cursor position.

#![cfg(feature = "tui")]

use connect4::GameSession;
use connect4::types::{Difficulty, GameMode, Player};
use std::collections::VecDeque;

/// Represents a piece currently performing a falling animation on the board.
///
/// Cosmetic pieces are used to provide visual feedback for moves before or during
/// the update to the logical board state.
pub struct FallingPiece {
    /// The zero-indexed column where the piece is falling.
    pub col: u32,
    /// The zero-indexed row where the piece will eventually settle.
    pub target_row: u32,
    /// The current vertical coordinate of the piece (expressed in row units).
    pub current_y: f64,
    /// The current vertical velocity of the piece (expressed in rows per second).
    pub velocity: f64,
    /// The point in time when the piece's physics state was last updated.
    pub last_update: std::time::Instant,
    /// The player whose piece is falling.
    pub player: Player,
}

/// A View-Model that orchestrates the relationship between game logic and TUI presentation.
///
/// `UiGameSession` encapsulates a `GameSession` and adds metadata required for a responsive
/// and animated user interface, such as the currently selected column for move entry
/// and a queue of active cosmetic animations.
pub struct UiGameSession {
    /// The underlying core game session managing logic, history, and state.
    pub core: GameSession,
    /// The currently highlighted column index in the UI.
    pub selected_column: u32,
    /// A queue of pieces currently in the process of falling, used for rendering animations.
    pub falling_pieces: VecDeque<FallingPiece>,
}

impl UiGameSession {
    /// Creates a new `UiGameSession` with the specified configuration.
    ///
    /// This constructor initializes the core game session and sets the initial UI state,
    /// such as centering the selection cursor.
    ///
    /// # Parameters
    /// - `cols`: Number of columns for the board.
    /// - `rows`: Number of rows for the board.
    /// - `mode`: The game mode (`Single`, `LocalTwo`, `Remote`).
    /// - `diff`: AI difficulty level.
    /// - `human`: The player designated as human.
    ///
    /// # Returns
    /// A new instance of `UiGameSession` with the cursor centered on the board.
    #[must_use]
    pub fn new(cols: u32, rows: u32, mode: GameMode, diff: Difficulty, human: Player) -> Self {
        Self {
            core: GameSession::new(cols, rows, mode, diff, human),
            selected_column: cols / 2,
            falling_pieces: VecDeque::new(),
        }
    }
}

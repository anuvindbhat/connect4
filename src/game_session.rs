//! # Game Session Management
//!
//! This module provides the `GameSession` struct, which manages the high-level state
//! of a Connect 4 game. It coordinates the game board, player turns, move history,
//! and integration with the AI and analytics engines.
//!
//! ### State Machine
//! The session acts as a state machine that transitions between player turns and
//! eventually to a terminal state (Win or Draw) through the execution of moves.
//! The flow generally follows:
//! 1.  **Initialization**: `GameSession::new` sets up the board and players.
//! 2.  **Move Execution**: `execute_move` handles a player's action, updates the board,
//!     checks for victory or a full board, and toggles the active player.
//! 3.  **Terminal State**: Once a win or draw is detected, the game reaches a terminal
//!     state, though the session object still holds the final board state for review.
//!
//! ### Engine Integration
//! `GameSession` maintains two AI engines:
//! - `ai_engine`: Used for making decisions when the AI is playing.
//! - `analytics_engine`: Used for background analysis to provide hints or evaluations.

use crate::config::{HeuristicWeights, TT_SIZE_MB};
use crate::engine::DynamicEngine;
use crate::error::Connect4Error;
use crate::game::Board;
use crate::types::{Difficulty, GameMode, Player};
use std::sync::{Arc, Mutex};

/// Represents the outcome of a move attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveResult {
    /// The move was successful, and the game continues with the next player.
    Success {
        /// The row where the piece landed.
        row: u32,
        /// The player whose turn is next.
        next_player: Player,
    },
    /// The move resulted in a win for the current player.
    Win {
        /// The row where the winning piece landed.
        row: u32,
        /// The player who won the game.
        winner: Player,
    },
    /// The move resulted in the board becoming full, ending the game in a draw.
    Draw {
        /// The row where the final piece landed.
        row: u32,
    },
}

/// Manages the state and orchestration of a Connect 4 game session.
///
/// `GameSession` is responsible for:
/// - Maintaining the logical game board (`Board`).
/// - Tracking the `current_player` whose turn it is.
/// - Storing a `move_history` of all pieces dropped.
/// - Interfacing with the AI (`ai_engine`) and background analytics (`analytics_engine`).
/// - Managing turn transitions and terminal state (Win/Draw) detection.
pub struct GameSession {
    /// The bitboard-based game engine.
    pub board: Board,
    /// The player who is currently taking their turn.
    pub current_player: Player,
    /// The player controlled by the local user.
    pub local_player: Player,
    /// A history of all moves made in the session, stored as a vector of (column, player).
    pub move_history: Vec<(u32, Player)>,
    /// Cached evaluation scores for each column, typically updated in the background.
    pub cached_scores: Vec<Option<i32>>,
    /// Flag indicating if the `cached_scores` are no longer consistent with the board.
    pub is_stale_analytics: bool,
    /// Flag indicating if the AI search engine is currently performing a background search.
    pub is_ai_thinking: bool,
    /// The configuration for how players interact (e.g. `Single` vs `Remote`).
    pub game_mode: GameMode,
    /// The AI difficulty level chosen for this session.
    pub difficulty: Difficulty,
    /// The high-performance AI engine for move selection.
    pub ai_engine: Arc<Mutex<DynamicEngine>>,
    /// A separate AI engine used for background position analysis.
    pub analytics_engine: Arc<Mutex<DynamicEngine>>,
}

impl GameSession {
    /// Creates a new `GameSession` with the specified board dimensions and configuration.
    ///
    /// Initializes the logical `Board` and both the primary AI and background analytics
    /// engines using the provided geometry and difficulty level.
    ///
    /// # Arguments
    ///
    /// * `cols` - The number of columns in the board.
    /// * `rows` - The number of rows in the board.
    /// * `mode` - The game mode (e.g., `Single`, `LocalTwo`, or `Remote`).
    /// * `diff` - The difficulty level for the AI engine.
    /// * `human` - The player (Red or Yellow) controlled by the local user.
    ///
    /// # Returns
    ///
    /// A fully initialized `GameSession` instance.
    #[must_use]
    pub fn new(cols: u32, rows: u32, mode: GameMode, diff: Difficulty, human: Player) -> Self {
        tracing::info!(
            "Creating new GameSession: size={}x{}, mode={:?}, difficulty={:?}, local_player={:?}",
            cols,
            rows,
            mode,
            diff,
            human
        );
        let board = Board::new(cols, rows);
        let ai_engine = Arc::new(Mutex::new(DynamicEngine::new(
            board.geometry.clone(),
            HeuristicWeights::default(),
            Some(TT_SIZE_MB),
        )));
        let analytics_engine = Arc::new(Mutex::new(DynamicEngine::new(
            board.geometry.clone(),
            HeuristicWeights::default(),
            Some(TT_SIZE_MB),
        )));

        Self {
            board,
            current_player: Player::Red,
            local_player: human,
            move_history: Vec::new(),
            cached_scores: vec![None; cols as usize],
            is_stale_analytics: true,
            is_ai_thinking: false,
            game_mode: mode,
            difficulty: diff,
            ai_engine,
            analytics_engine,
        }
    }

    /// Performs a move transaction, updating the game state and checking for a winner or draw.
    ///
    /// This method is the primary driver of the session's state machine:
    /// 1. Drops a piece in the logical board for the `current_player`.
    /// 2. Appends the move to the `move_history`.
    /// 3. Updates the `is_stale_analytics` flag to indicate that cached evaluations are invalid.
    /// 4. Resolves the game state:
    ///    - If a win is detected, returns `MoveResult::Win`.
    ///    - If the board is full, returns `MoveResult::Draw`.
    ///    - Otherwise, toggles the `current_player` and returns `MoveResult::Success`.
    ///
    /// # Arguments
    ///
    /// * `col` - The 0-indexed column where the current player should drop their piece.
    ///
    /// # Returns
    ///
    /// * `Ok(MoveResult)` - The outcome of the move, which can be `Success`, `Win`, or `Draw`.
    /// * `Err(Connect4Error)` - If the move is invalid (e.g., the column is full or out of bounds).
    ///
    /// # Errors
    ///
    /// Returns a `Connect4Error::ColumnFull` if the selected column is full, or
    /// `Connect4Error::InvalidColumn` if the column index is out of bounds.
    pub fn execute_move(&mut self, col: u32) -> Result<MoveResult, Connect4Error> {
        let player = self.current_player;
        let row = self.board.drop_piece(col, player).map_err(|e| match e {
            crate::game::GameError::ColumnFull => Connect4Error::ColumnFull,
            crate::game::GameError::InvalidColumn => Connect4Error::InvalidColumn,
        })?;
        self.move_history.push((col, player));

        self.is_stale_analytics = true;

        // Terminal State Resolution
        let result = if self.board.has_won(player) {
            tracing::info!("Move results in WIN for {:?}", player);
            MoveResult::Win {
                row,
                winner: player,
            }
        } else if self.board.is_full() {
            tracing::info!("Move results in DRAW");
            MoveResult::Draw { row }
        } else {
            self.current_player = player.other();
            tracing::debug!(
                "Move successful in col {}: row={}, next_player={:?}",
                col,
                row,
                self.current_player
            );
            MoveResult::Success {
                row,
                next_player: self.current_player,
            }
        };

        Ok(result)
    }
}

#[cfg(test)]
/// Unit tests for the `GameSession` struct and its state machine logic.
mod tests {
    use super::*;

    /// Verifies that a new session is correctly initialized with the provided parameters.
    #[test]
    fn test_session_initialization() {
        let session = GameSession::new(7, 6, GameMode::Single, Difficulty::Medium, Player::Red);
        assert_eq!(
            session.board.columns(),
            7,
            "Session must initialize with correct columns"
        );
        assert_eq!(
            session.board.rows(),
            6,
            "Session must initialize with correct rows"
        );
        assert_eq!(
            session.current_player,
            Player::Red,
            "Session must start with Red player"
        );
        assert_eq!(
            session.local_player,
            Player::Red,
            "Session must correctly identify local player"
        );
        assert_eq!(
            session.game_mode,
            GameMode::Single,
            "Session must have correct game mode"
        );
        assert_eq!(
            session.difficulty,
            Difficulty::Medium,
            "Session must have correct difficulty"
        );
        assert!(
            session.move_history.is_empty(),
            "New session must have empty move history"
        );
    }

    /// Tests a sequence of moves to ensure turn-taking and history tracking work correctly.
    #[test]
    fn test_session_move_sequence() {
        let mut session = GameSession::new(7, 6, GameMode::LocalTwo, Difficulty::Easy, Player::Red);

        // Move 1: Red
        let res1 = session.execute_move(3).unwrap();
        assert!(matches!(
            res1,
            MoveResult::Success {
                row: 0,
                next_player: Player::Yellow
            }
        ));
        assert_eq!(session.current_player, Player::Yellow);
        assert_eq!(session.move_history, vec![(3, Player::Red)]);

        // Move 2: Yellow
        let res2 = session.execute_move(3).unwrap();
        assert!(matches!(
            res2,
            MoveResult::Success {
                row: 1,
                next_player: Player::Red
            }
        ));
        assert_eq!(session.current_player, Player::Red);
        assert_eq!(
            session.move_history,
            vec![(3, Player::Red), (3, Player::Yellow)]
        );
    }

    /// Verifies that the session correctly identifies a win condition.
    #[test]
    fn test_session_win() {
        let mut session = GameSession::new(7, 6, GameMode::LocalTwo, Difficulty::Easy, Player::Red);
        // Red wins vertically in col 0
        for _ in 0..3 {
            session.execute_move(0).unwrap(); // Red
            session.execute_move(1).unwrap(); // Yellow
        }
        let res = session.execute_move(0).unwrap(); // Red wins
        assert!(matches!(
            res,
            MoveResult::Win {
                row: 3,
                winner: Player::Red
            }
        ));
        assert_eq!(session.current_player, Player::Red);
    }

    /// Verifies that the session correctly identifies a draw condition.
    #[test]
    fn test_session_draw() {
        let mut session = GameSession::new(2, 2, GameMode::LocalTwo, Difficulty::Easy, Player::Red);
        // 2x2 board, win-length is still 4, so it's always a draw if filled.
        // R Y
        // Y R
        session.execute_move(0).unwrap(); // R (0,0)
        session.execute_move(1).unwrap(); // Y (1,0)
        session.execute_move(1).unwrap(); // R (1,1)
        let res = session.execute_move(0).unwrap(); // Y (0,1)
        assert!(matches!(res, MoveResult::Draw { row: 1 }));
        assert!(session.board.is_full());
    }

    /// Verifies that invalid moves (out of bounds or full columns) are correctly rejected.
    #[test]
    fn test_session_invalid_moves() {
        let mut session = GameSession::new(7, 6, GameMode::LocalTwo, Difficulty::Easy, Player::Red);

        // Out of bounds
        assert!(matches!(
            session.execute_move(7),
            Err(Connect4Error::InvalidColumn)
        ));

        // Column full
        for _ in 0..6 {
            session.execute_move(0).unwrap();
        }
        assert!(matches!(
            session.execute_move(0),
            Err(Connect4Error::ColumnFull)
        ));
    }

    /// Verifies that making a move correctly flags analytics as stale.
    #[test]
    fn test_session_analytics_staleness() {
        let mut session = GameSession::new(7, 6, GameMode::Single, Difficulty::Medium, Player::Red);

        // Initially stale
        assert!(session.is_stale_analytics);

        // Simulate an analytics update
        session.is_stale_analytics = false;
        session.cached_scores[3] = Some(100);

        // Making a move must make it stale again
        session.execute_move(3).unwrap();
        assert!(session.is_stale_analytics);
        // Note: The implementation doesn't currently clear cached_scores array,
        // it just sets the stale flag. Let's verify that.
        assert_eq!(session.cached_scores[3], Some(100));
    }

    /// Verifies that the move history remains consistent after multiple moves.
    #[test]
    fn test_session_move_history_consistency() {
        let mut session = GameSession::new(7, 6, GameMode::LocalTwo, Difficulty::Easy, Player::Red);
        let moves: [u32; 5] = [3, 4, 3, 5, 2];
        for &m in &moves {
            session.execute_move(m).unwrap();
        }

        assert_eq!(session.move_history.len(), 5);
        for (i, &(col, player)) in session.move_history.iter().enumerate() {
            assert_eq!(col, moves[i]);
            let expected_p = if i % 2 == 0 {
                Player::Red
            } else {
                Player::Yellow
            };
            assert_eq!(player, expected_p);
        }
    }
}

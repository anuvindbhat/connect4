//! # Configuration
//!
//! Centralized configuration constants and structures for the Connect 4 engine,
//! AI search, and networking protocols.

// ========================================================================================
// AI SEARCH CONFIGURATION
// ========================================================================================

/// Search depth for the Easy AI.
pub const AI_EASY_DEPTH: u32 = 1;

/// Search depth for the Medium AI.
pub const AI_MEDIUM_DEPTH: u32 = 3;

/// Search depth for the Hard AI.
pub const AI_HARD_DEPTH: u32 = 5;

/// Search depth for the Expert AI.
pub const AI_EXPERT_DEPTH: u32 = 7;

/// Search depth for the Grandmaster AI.
pub const AI_GRANDMASTER_DEPTH: u32 = 9;

/// Search depth used for the background "Analysis" pane.
pub const ANALYTICS_DEPTH: u32 = 13;

/// Controls the "chaos" of stochastic move selection.
/// A higher temperature leads to more random moves, while a lower temperature
/// makes the AI more deterministic and "optimal".
pub const TEMPERATURE: f64 = 15.0;

/// Default size for Transposition Tables in Megabytes.
pub const TT_SIZE_MB: usize = 128;

// ========================================================================================
// HEURISTIC SCORING CONSTANTS (INTERNAL DEFAULTS)
// ========================================================================================

/// Score value awarded for reaching a terminal winning state.
/// This is treated as infinity for all practical search purposes.
pub const SCORE_WIN: i32 = 100_000_000;

/// The threshold at or above which a score is considered a tactical win/loss.
/// Calibrated to 1,000 to allow for maximum possible Connect 4 distance (42)
/// while remaining far above the highest heuristic fork score (10,000).
pub const WIN_THRESHOLD: i32 = SCORE_WIN - 1000;

/// Heuristic score for an "Immediate Fork" (multiple simultaneous winning threats).
/// Set to 10,000 to represent a decisive advantage. If a player has a fork,
/// the opponent can only block one threat, making the win effectively guaranteed.
/// This value is 10x higher than a single threat to prioritize it above all else.
const SCORE_FORK_IMMEDIATE: i32 = 10000;

/// Heuristic score for an immediate 3-in-a-window threat (one playable move away from winning).
/// This is the "Base Unit" of tactical value. All other scores are calibrated relative to this.
const SCORE_THREAT_IMMEDIATE: i32 = 1000;

/// Heuristic score for a future (unblocked) 3-in-a-window threat.
const SCORE_THREAT_FUTURE: i32 = 450;

/// Heuristic score for an immediately playable 2-in-a-window set up
/// (an empty cell in a 2-in-a-window that is immediately playable).
const SCORE_SETUP_IMMEDIATE: i32 = 300;

/// Heuristic score for a 3-in-a-window structural pattern (not necessarily an immediate threat).
const SCORE_THREE: i32 = 200;

/// Heuristic score for a 2-in-a-window connection.
const SCORE_TWO: i32 = 50;

/// Weighting factor for potential winning windows (Mobility).
/// Rewards players for keeping their options open and "bottlenecking" the opponent.
const WEIGHT_POTENTIAL_WINDOW: i32 = 50;

// ========================================================================================
// POSITIONAL WEIGHTS (3-TIER GRADIENT)
// ========================================================================================

/// Weighting factor for "The Core" (Tier 0).
/// The primary central foundation hubs and high-connectivity intersections.
const WEIGHT_CORE: i32 = 15;

/// Weighting factor for "The Inner Shell" (Tier 1).
/// The secondary strategic support cells and inner flanks.
const WEIGHT_INNER: i32 = 8;

/// Weighting factor for "The Outer Perimeter" (Tier 2).
/// The edge columns and top rows (mathematically the weakest).
const WEIGHT_OUTER: i32 = 1;

/// A collection of weights used by the AI heuristic evaluation and search parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicWeights {
    /// Score value awarded for reaching a terminal winning state.
    pub score_win: i32,
    /// Heuristic score for an "Immediate Fork" (multiple simultaneous winning threats).
    pub score_fork_immediate: i32,
    /// Heuristic score for an immediate 3-in-a-window threat.
    pub score_threat_immediate: i32,
    /// Heuristic score for a future (unblocked) 3-in-a-window threat.
    pub score_threat_future: i32,
    /// Heuristic score for an immediately playable 2-in-a-window set up.
    pub score_setup_immediate: i32,
    /// Heuristic score for a 3-in-a-window structural pattern.
    pub score_three: i32,
    /// Heuristic score for a 2-in-a-window connection.
    pub score_two: i32,
    /// Weighting factor for potential winning windows (Mobility).
    pub weight_potential_window: i32,
    /// Weighting factor for "The Core" (Tier 0).
    pub weight_core: i32,
    /// Weighting factor for "The Inner Shell" (Tier 1).
    pub weight_inner: i32,
    /// Weighting factor for "The Outer Perimeter" (Tier 2).
    pub weight_outer: i32,
}

impl Default for HeuristicWeights {
    /// Creates a new `HeuristicWeights` instance with default values.
    fn default() -> Self {
        Self {
            score_win: SCORE_WIN,
            score_fork_immediate: SCORE_FORK_IMMEDIATE,
            score_threat_immediate: SCORE_THREAT_IMMEDIATE,
            score_threat_future: SCORE_THREAT_FUTURE,
            score_setup_immediate: SCORE_SETUP_IMMEDIATE,
            score_three: SCORE_THREE,
            score_two: SCORE_TWO,
            weight_potential_window: WEIGHT_POTENTIAL_WINDOW,
            weight_core: WEIGHT_CORE,
            weight_inner: WEIGHT_INNER,
            weight_outer: WEIGHT_OUTER,
        }
    }
}

// ========================================================================================
// BOARD CONSTRAINTS
// ========================================================================================

/// Maximum number of columns supported by the bitboard engine.
/// Limited by `u128` capacity when combined with rows.
pub const MAX_COLUMNS: u32 = 9;

/// Maximum number of rows supported by the bitboard engine.
pub const MAX_ROWS: u32 = 7;

/// Vertical height allocated per column in the bitboard (includes sentinel).
/// We need 1 extra bit per column to act as a sentinel/barrier so that bit-shifts
/// don't wrap around to the next column.
///
/// For example, in a 6-row board, the bits 0-5 are the rows, and bit 6 is the sentinel.
/// This ensures that a vertical or diagonal check doesn't accidentally cross from
/// the top of column X to the bottom of column X+1.
pub const BITBOARD_COL_HEIGHT: u32 = MAX_ROWS + 1;

// ========================================================================================
// NETWORKING CONFIGURATION
// ========================================================================================

/// Default timeout for network operations (handshakes, connections) in milliseconds.
pub const NETWORK_TIMEOUT_MS: u64 = 5000;

/// Maximum size of a single network frame in bytes.
/// Set to 64KB to prevent memory exhaustion from malicious peers.
pub const MAX_FRAME_SIZE: usize = 64 * 1024;

/// Maximum length of a player's name in the network protocol.
pub const MAX_NAME_LEN: usize = 32;

/// Maximum length of a chat message in the network protocol.
pub const MAX_CHAT_LEN: usize = 512;

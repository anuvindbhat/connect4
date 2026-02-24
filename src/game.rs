//! # Connect 4 Game Engine
//!
//! This module provides the core logic and data structures for the Connect 4 game,
//! utilizing an efficient bitboard-based engine.
//!
//! ## Bitboard Representation
//! The board is represented as a series of columns, where each column includes a
//! "sentinel bit" (an extra row at the top) to prevent bitwise operations from
//! wrapping between columns.
//!
//! For a board with $W$ columns and $H$ rows:
//! - Total bits required: $W \times (H + 1)$
//! - Column $c$, Row $r$ bit index: $c \times (H + 1) + r$
//!
//! This representation allows for O(1) win detection, piece dropping, and
//! board state analysis using bitwise shifts and masks.

use crate::config::MAX_COLUMNS;
use crate::types::{Bitboard, BoardStats, Cell, Player};
use std::fmt;

// ========================================================================================
// DATA STRUCTURES
// ========================================================================================

/// Errors that can occur during game operations.
#[derive(Debug)]
pub enum GameError {
    /// The specified column is already full and cannot accept more pieces.
    ColumnFull,
    /// The specified column index is outside the valid range for the board's geometry.
    InvalidColumn,
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::ColumnFull => write!(f, "Column is full"),
            GameError::InvalidColumn => write!(f, "Invalid column index"),
        }
    }
}

impl std::error::Error for GameError {}

// ========================================================================================
// BITBOARD ENGINE
// ========================================================================================

/// Immutable geometry and pre-calculated masks for a specific board configuration.
///
/// This struct holds all the constant information about a board of a certain size,
/// allowing the `BoardState` to remain lightweight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardGeometry<T: Bitboard> {
    /// Number of columns in the board.
    pub(crate) columns: u32,
    /// Number of rows in the board.
    pub(crate) rows: u32,
    /// Mask with bits set at the bottom row of every column.
    pub(crate) bottom_mask: T,
    /// Mask with bits set for every playable cell on the board.
    pub(crate) board_mask: T,
    /// Mask with bits set at the top row of every column.
    pub(crate) top_row_mask: T,
    /// Three-tier weight masks (Core, Inner, Outer) for heuristic evaluation.
    pub(crate) weight_masks: [T; 3],
    /// Individual masks for each column.
    pub(crate) column_masks: Vec<T>,
    /// The starting bit index for each column.
    pub(crate) column_starts: Vec<u32>,
    /// The order in which columns should be searched (typically center-out).
    pub(crate) search_order: Vec<u32>,
}

impl<T: Bitboard> BoardGeometry<T> {
    /// Returns the number of columns in this geometry.
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.columns
    }

    /// Returns the number of rows in this geometry.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Creates a new board geometry with pre-calculated masks for the given dimensions.
    ///
    /// This method initializes all masks and search orders required for the bitboard engine.
    /// It ensures that the board dimensions are valid and fit within the bitboard capacity.
    ///
    /// # Panics
    /// - Panics if `columns` or `rows` is 0.
    /// - Panics if `columns` exceeds `MAX_COLUMNS`.
    /// - Panics if the total bits required ($columns \times (rows + 1)$) exceeds the bitboard capacity.
    #[must_use]
    pub fn new(columns: u32, rows: u32) -> Self {
        assert!(columns > 0, "Board columns must be greater than 0");
        assert!(rows > 0, "Board rows must be greater than 0");
        assert!(
            columns <= MAX_COLUMNS,
            "Board columns {columns} exceeds MAX_COLUMNS {MAX_COLUMNS}"
        );
        let h = rows + 1;
        assert!(
            (columns * h) <= T::BITS,
            "Board size exceeds bitboard capacity"
        );
        let mut bottom_mask = T::zero();
        let mut board_mask = T::zero();
        let mut top_row_mask = T::zero();
        let mut column_masks = Vec::with_capacity(columns as usize);
        let mut column_starts = Vec::with_capacity(columns as usize);

        for c in 0..columns {
            let col_start = c * h;
            column_starts.push(col_start);
            bottom_mask |= T::one() << col_start;
            top_row_mask |= T::one() << (col_start + rows - 1);

            let mut col_mask = T::zero();
            for r in 0..rows {
                let bit = T::one() << (col_start + r);
                board_mask |= bit;
                col_mask |= bit;
            }
            column_masks.push(col_mask);
        }

        tracing::debug!(
            "Creating BoardGeometry: {}x{}, board_mask: {:032X}",
            columns,
            rows,
            board_mask.to_u128()
        );

        let weight_masks = Self::generate_heatmap(columns, rows);

        // Calculate optimal search order (center-out) using integer distances.
        // For even-width boards, stable sort preserves original column order
        // for ties (e.g. 3 before 4 on an 8-column board).
        let mut search_order: Vec<u32> = (0..columns).collect();
        let mid_target = i32::try_from(columns).expect("Columns fit in i32") - 1;
        search_order.sort_by_key(|&col| {
            (2 * i32::try_from(col).expect("Column index fits in i32") - mid_target).abs()
        });

        Self {
            columns,
            rows,
            bottom_mask,
            board_mask,
            top_row_mask,
            weight_masks,
            column_masks,
            column_starts,
            search_order,
        }
    }

    /// Generates a three-tier geometric potential heatmap for heuristic evaluation.
    ///
    /// The heatmap is calculated using three factors for each cell:
    /// 1. **Connectivity**: The number of potential winning windows (4-in-a-row) passing through the cell.
    /// 2. **Gravity**: A bonus for cells in lower rows to prioritize establishing a foundation.
    /// 3. **Centrality**: A bonus for proximity to the center column(s).
    ///
    /// The resulting scores are partitioned into three tiers: Core (15%), Inner (35%), and Outer (50%).
    fn generate_heatmap(columns: u32, rows: u32) -> [T; 3] {
        tracing::debug!(
            "Generating geometric potential heatmap for {}x{} board",
            columns,
            rows
        );
        // We calculate a "Geometric Potential Score" for every cell using 3 factors:
        // 1. Connectivity: Number of possible 4-windows passing through the cell.
        // 2. Gravity: Bonus for lower rows to prioritize foundations.
        // 3. Centrality: Symmetrical bonus for proximity to the center column(s).
        let h = rows + 1;
        let mut cell_scores = Vec::with_capacity((columns * rows) as usize);
        for c in 0..columns {
            for r in 0..rows {
                let mut windows = 0;
                // Check all 4 directions for potential windows
                for (dc, dr) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                    // A window of 4 can include this cell at 4 different relative offsets
                    for offset in 0..4 {
                        let start_c =
                            i32::try_from(c).expect("Column index fits in i32") - (dc * offset);
                        let start_r =
                            i32::try_from(r).expect("Row index fits in i32") - (dr * offset);
                        let end_c = start_c + (dc * 3);
                        let end_r = start_r + (dr * 3);

                        if start_c >= 0
                            && start_c < i32::try_from(columns).expect("Columns fit in i32")
                            && start_r >= 0
                            && start_r < i32::try_from(rows).expect("Rows fit in i32")
                            && end_c >= 0
                            && end_c < i32::try_from(columns).expect("Columns fit in i32")
                            && end_r >= 0
                            && end_r < i32::try_from(rows).expect("Rows fit in i32")
                        {
                            windows += 1;
                        }
                    }
                }

                let gravity = rows - 1 - r;
                let centrality = (columns - 1).saturating_sub((2 * c).abs_diff(columns - 1));

                // Score: (Windows * 2) + (Gravity * 4) + (Centrality * 3)
                // Note: `centrality` already contains a factor of 2 relative to standard float-based
                // centrality, so multiplying by 3 gives us the required 6x factor.
                let score = (windows * 2) + (gravity * 4) + (centrality * 3);

                let bit = T::one() << (c * h + r);
                cell_scores.push((bit, score));
            }
        }

        // Sort by score descending to identify the most valuable cells.
        cell_scores.sort_by_key(|&(_, s)| -i32::try_from(s).expect("Score fits in i32"));

        let mut weight_masks = [T::zero(); 3];
        let n = cell_scores.len();

        // 3-Tier Percentile Split: 15% (Core) / 35% (Inner) / 50% (Outer)
        let tier0_idx = ((n * 15) / 100).saturating_sub(1);
        let tier1_idx = ((n * 50) / 100).saturating_sub(1); // Cumulative 15+35

        let thresholds = [
            cell_scores.get(tier0_idx).map_or(0, |&(_, s)| s),
            cell_scores.get(tier1_idx).map_or(0, |&(_, s)| s),
        ];
        tracing::debug!("Heatmap tier thresholds: {:?}", thresholds);

        for (bit, score) in cell_scores {
            if score >= thresholds[0] {
                weight_masks[0] |= bit; // Core
            } else if score >= thresholds[1] {
                weight_masks[1] |= bit; // Inner
            } else {
                weight_masks[2] |= bit; // Outer
            }
        }
        weight_masks
    }
}

/// Dynamic wrapper for board geometry, choosing between `u64` and `u128` bitboards.
///
/// This allows the engine to optimize for smaller boards (up to 8x7) while still
/// supporting larger custom boards (up to 9x13 or similar).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicBoardGeometry {
    /// Geometry for smaller boards that fit within a 64-bit integer.
    Small(BoardGeometry<u64>),
    /// Geometry for larger boards that require a 128-bit integer.
    Large(BoardGeometry<u128>),
}

impl DynamicBoardGeometry {
    /// Creates a new dynamic geometry based on board dimensions.
    ///
    /// Automatically selects `u64` if the board (including sentinel bits) fits,
    /// otherwise selects `u128`.
    #[must_use]
    pub fn new(columns: u32, rows: u32) -> Self {
        let h = rows + 1;
        if columns * h <= 64 {
            Self::Small(BoardGeometry::<u64>::new(columns, rows))
        } else {
            Self::Large(BoardGeometry::<u128>::new(columns, rows))
        }
    }

    /// Returns the number of columns in this geometry.
    #[must_use]
    pub fn columns(&self) -> u32 {
        match self {
            Self::Small(g) => g.columns,
            Self::Large(g) => g.columns,
        }
    }

    /// Returns the number of rows in this geometry.
    #[must_use]
    pub fn rows(&self) -> u32 {
        match self {
            Self::Small(g) => g.rows,
            Self::Large(g) => g.rows,
        }
    }
}

/// Raw bitboard state for both players.
///
/// This struct is the core of the game engine, representing the board as two
/// bitboards (one for each player). It is designed to be extremely lightweight
/// (fits in 16 or 32 bytes) and is `Copy` for easy use in search algorithms.
///
/// ## Bitboard Tricks
///
/// ### Sentinel Bit Strategy
/// Each column in the bitboard is represented by $H$ rows plus one additional
/// "sentinel bit" at the top. This extra bit serves as a barrier:
/// - It prevents horizontal and diagonal shifts from "wrapping" between columns.
/// - It allows us to detect when a column is full without a separate check.
///
/// ### Sliding Intersections for Win Detection
/// To detect 4-in-a-row in O(1) time, we use a series of shifts and intersections.
/// For a given shift value $S$ (representing a direction like vertical or diagonal):
/// 1. `m = bits & (bits >> S)`: Groups of 2 adjacent bits in direction $S$.
/// 2. `m & (m >> (2 * S))`: Groups of 4 adjacent bits in direction $S$.
///
/// If the resulting bitboard is non-zero, the player has won.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BoardState<T: Bitboard> {
    /// Bitboard for the Red player.
    pub(crate) red: T,
    /// Bitboard for the Yellow player.
    pub(crate) yellow: T,
}

impl<T: Bitboard> BoardState<T> {
    /// Returns the bits for a specific player.
    #[must_use]
    pub fn bits(&self, player: Player) -> T {
        match player {
            Player::Red => self.red,
            Player::Yellow => self.yellow,
        }
    }

    /// Computes a stable 128-bit hash of the board state.
    ///
    /// This hash is used for transposition tables, desync detection, and analytics.
    /// It ensures that `Red` and `Yellow` positions are not confused by XOR-ing
    /// one player's bits with a rotated version of the other's.
    #[must_use]
    pub fn hash(&self) -> u128 {
        // We can rotate by 64 even if the underlying bitboard is u64.
        // That will just cause the upper 64 bits to be the yellow bits (which
        // is good for a hash).
        self.red.to_u128() ^ self.yellow.to_u128().rotate_left(64)
    }

    /// Returns the cell state (Red, Yellow, or Empty) at a specific (col, row).
    ///
    /// # Panics
    /// Panics if `col` or `row` are out of bounds for the given geometry.
    #[must_use]
    pub fn get_cell(&self, col: u32, row: u32, geo: &BoardGeometry<T>) -> Cell {
        assert!(
            col < geo.columns && row < geo.rows,
            "get_cell called out-of-bounds: col {col}, row {row}"
        );
        let bit_idx = geo.column_starts[col as usize] + row;
        let bit = T::one() << bit_idx;
        if (self.red & bit) != T::zero() {
            Cell::Occupied(Player::Red)
        } else if (self.yellow & bit) != T::zero() {
            Cell::Occupied(Player::Yellow)
        } else {
            Cell::Empty
        }
    }

    /// Attempts to drop a piece into a specific column for the given player.
    ///
    /// If successful, returns the new `BoardState` and the bit index of the placed piece.
    /// Returns `None` if the column is full.
    #[must_use]
    pub fn drop_piece_with_index(
        &self,
        col: u32,
        player: Player,
        geo: &BoardGeometry<T>,
    ) -> Option<(Self, u32)> {
        let next_bit = self.get_next_bit(col, geo)?;
        let bit_index = next_bit.trailing_zeros();
        let mut next_state = *self;
        match player {
            Player::Red => next_state.red |= next_bit,
            Player::Yellow => next_state.yellow |= next_bit,
        }
        Some((next_state, bit_index))
    }

    /// Attempts to drop a piece into a specific column for the given player.
    ///
    /// Returns `None` if the column is full.
    #[must_use]
    pub fn drop_piece(&self, col: u32, player: Player, geo: &BoardGeometry<T>) -> Option<Self> {
        self.drop_piece_with_index(col, player, geo)
            .map(|(state, _)| state)
    }

    /// Finds the bit representing the next empty row in a specific column.
    ///
    /// # Implementation Details
    /// This function uses a bitwise carry trick to find the first empty row without iteration:
    /// 1. `occupied` combines both players' bits.
    /// 2. `col_occupied + bottom_bit` propagates a carry bit up through the contiguous block of pieces.
    /// 3. The carry stops at the first zero (empty cell) or the top (sentinel bit).
    /// 4. Masking with `col_mask` ensures the result is valid and stays within the column.
    ///
    /// # Panics
    /// Panics if `col` is out of bounds for the given geometry.
    #[must_use]
    pub fn get_next_bit(&self, col: u32, geo: &BoardGeometry<T>) -> Option<T> {
        assert!(
            col < geo.columns,
            "get_next_bit called with out-of-bounds column: {col} (max {})",
            geo.columns
        );
        let col_mask = geo.column_masks[col as usize];
        let occupied = self.red | self.yellow;
        let col_occupied = occupied & col_mask;
        let bottom_bit = geo.bottom_mask & col_mask;

        // Carry chain optimization: finds the lowest unset bit in the column.
        // If the column is full, the carry will hit the sentinel bit (rows+1),
        // and masking with `col_mask` (which excludes sentinel bits) will yield zero.
        let next_bit = col_occupied.wrapping_add(bottom_bit) & col_mask;

        if next_bit == T::zero() {
            None
        } else {
            Some(next_bit)
        }
    }

    /// Checks if the specified player has achieved a Connect 4.
    ///
    /// # Implementation Details
    /// Uses the "Sliding Intersection" bitboard trick to check all four directions
    /// (Vertical, Horizontal, and both Diagonals) in O(1) time.
    ///
    /// # Panics
    /// Panics if the board height (rows + 1) cannot be represented as `u32`.
    #[must_use]
    pub fn has_won(&self, player: Player, geo: &BoardGeometry<T>) -> bool {
        let bits = self.bits(player);
        let h = geo.rows + 1;

        let check = |shift: u32| -> bool {
            let m = bits & (bits >> shift);
            (m & (m >> (2 * shift))) != T::zero()
        };

        // 1: Vertical
        // h: Horizontal
        // h+1: Diagonal (/)
        // h-1: Diagonal (\)
        check(1) || check(h) || check(h + 1) || check(h - 1)
    }

    /// Calculates the length of the longest chain passing through a specific bit index.
    ///
    /// This is used to update board statistics and analyze threats. It checks
    /// all four directions from the specified bit.
    ///
    /// # Panics
    /// Panics if the board height (rows + 1) cannot be represented as `u32`.
    #[must_use]
    pub fn calculate_chain_length_at(
        &self,
        player: Player,
        bit_idx: u32,
        geo: &BoardGeometry<T>,
    ) -> u32 {
        let bits = self.bits(player);
        let h = geo.rows + 1;
        let mut max_chain = 1;

        for shift in [1, h, h + 1, h - 1] {
            let mut count = 1;

            // Check in positive direction
            let mut current_idx = bit_idx.wrapping_add(shift);
            while current_idx < T::BITS
                && (bits >> current_idx) & T::one() != T::zero()
                && (geo.board_mask >> current_idx) & T::one() != T::zero()
            {
                count += 1;
                current_idx = current_idx.wrapping_add(shift);
            }

            // Check in negative direction
            let mut current_idx = bit_idx.wrapping_sub(shift);
            while current_idx < T::BITS
                && (bits >> current_idx) & T::one() != T::zero()
                && (geo.board_mask >> current_idx) & T::one() != T::zero()
            {
                count += 1;
                current_idx = current_idx.wrapping_sub(shift);
            }

            max_chain = max_chain.max(count);
        }

        max_chain
    }

    /// Checks if the board is completely full.
    ///
    /// Uses the `top_row_mask` to quickly verify if the highest playable row
    /// in every column is occupied.
    #[must_use]
    pub fn is_full(&self, geo: &BoardGeometry<T>) -> bool {
        let occupied = self.red | self.yellow;
        (occupied & geo.top_row_mask) == geo.top_row_mask
    }
}

/// Dynamic wrapper for board state, supporting both `u64` and `u128` bitboards.
///
/// This enum allows the game to switch between bitboard sizes at runtime while
/// providing a consistent interface for operations like dropping pieces and
/// checking for wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicBoardState {
    /// State for boards fitting in 64 bits.
    Small(BoardState<u64>),
    /// State for boards requiring 128 bits.
    Large(BoardState<u128>),
}

impl DynamicBoardState {
    /// Creates an empty board state for the provided geometry.
    #[must_use]
    pub fn new(geo: &DynamicBoardGeometry) -> Self {
        match geo {
            DynamicBoardGeometry::Small(_) => Self::Small(BoardState::<u64>::default()),
            DynamicBoardGeometry::Large(_) => Self::Large(BoardState::<u128>::default()),
        }
    }

    /// Returns the cell state (Red, Yellow, or Empty) at a specific (col, row).
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn get_cell(&self, col: u32, row: u32, geo: &DynamicBoardGeometry) -> Cell {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => s.get_cell(col, row, g),
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => s.get_cell(col, row, g),
            _ => panic!("State/Geometry variant mismatch in get_cell"),
        }
    }

    /// Attempts to drop a piece into a specific column for the given player.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn drop_piece(&self, col: u32, player: Player, geo: &DynamicBoardGeometry) -> Option<Self> {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => {
                s.drop_piece(col, player, g).map(Self::Small)
            }
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => {
                s.drop_piece(col, player, g).map(Self::Large)
            }
            _ => panic!("State/Geometry variant mismatch in drop_piece"),
        }
    }

    /// Attempts to drop a piece into a specific column, returning the new state and bit index.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn drop_piece_with_index(
        &self,
        col: u32,
        player: Player,
        geo: &DynamicBoardGeometry,
    ) -> Option<(Self, u32)> {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => s
                .drop_piece_with_index(col, player, g)
                .map(|(ns, idx)| (Self::Small(ns), idx)),
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => s
                .drop_piece_with_index(col, player, g)
                .map(|(ns, idx)| (Self::Large(ns), idx)),
            _ => panic!("State/Geometry variant mismatch in drop_piece_with_index"),
        }
    }

    /// Checks if the specified player has achieved a Connect 4.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn has_won(&self, player: Player, geo: &DynamicBoardGeometry) -> bool {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => s.has_won(player, g),
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => s.has_won(player, g),
            _ => panic!("State/Geometry variant mismatch in has_won"),
        }
    }

    /// Calculates the length of the longest chain passing through a specific bit index.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn calculate_chain_length_at(
        &self,
        player: Player,
        bit_idx: u32,
        geo: &DynamicBoardGeometry,
    ) -> u32 {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => {
                s.calculate_chain_length_at(player, bit_idx, g)
            }
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => {
                s.calculate_chain_length_at(player, bit_idx, g)
            }
            _ => panic!("State/Geometry variant mismatch in calculate_chain_length_at"),
        }
    }

    /// Checks if the board is full.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn is_full(&self, geo: &DynamicBoardGeometry) -> bool {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => s.is_full(g),
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => s.is_full(g),
            _ => panic!("State/Geometry variant mismatch in is_full"),
        }
    }

    /// Computes a stable 128-bit hash of the board state.
    #[must_use]
    pub fn hash(&self) -> u128 {
        match self {
            Self::Small(s) => s.hash(),
            Self::Large(s) => s.hash(),
        }
    }

    /// Returns the bit index of the next empty row in a specific column.
    ///
    /// # Panics
    /// Panics if the engine variant doesn't match the board state variant.
    #[must_use]
    pub fn get_next_bit_index(&self, col: u32, geo: &DynamicBoardGeometry) -> Option<u32> {
        match (self, geo) {
            (Self::Small(s), DynamicBoardGeometry::Small(g)) => {
                s.get_next_bit(col, g).map(u64::trailing_zeros)
            }
            (Self::Large(s), DynamicBoardGeometry::Large(g)) => {
                s.get_next_bit(col, g).map(u128::trailing_zeros)
            }
            _ => panic!("State/Geometry variant mismatch in get_next_bit_index"),
        }
    }
}

/// Threat level for a player based on the opponent's potential moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel {
    /// No immediate threats detected.
    Stable,
    /// The opponent has a potential 3-in-a-row (Connect 3).
    Caution,
    /// The opponent can win on their next move.
    Critical,
}

/// High-level wrapper for the game board, combining geometry, state, and statistics.
///
/// This is the primary interface for interacting with the game engine. It manages
/// the dynamic selection of bitboard sizes and tracks live statistics like
/// the longest chain for each player.
#[derive(Clone)]
pub struct Board {
    /// The immutable geometry and masks for the board.
    pub geometry: DynamicBoardGeometry,
    /// The current bitboard state of the game.
    pub state: DynamicBoardState,
    /// Live statistics for the current game session.
    pub stats: BoardStats,
}

impl Board {
    /// Creates a new empty board with the given dimensions.
    ///
    /// This initializes the appropriate dynamic geometry and state based on the size.
    #[must_use]
    pub fn new(columns: u32, rows: u32) -> Self {
        let geometry = DynamicBoardGeometry::new(columns, rows);
        let state = DynamicBoardState::new(&geometry);
        Self {
            geometry,
            state,
            stats: BoardStats::default(),
        }
    }

    /// Drops a piece into the specified column for the given player.
    ///
    /// Updates the board state and live statistics if successful.
    ///
    /// # Errors
    /// - Returns `GameError::InvalidColumn` if the column index is out of bounds.
    /// - Returns `GameError::ColumnFull` if the column is already at maximum capacity.
    pub fn drop_piece(&mut self, col: u32, player: Player) -> Result<u32, GameError> {
        if col >= self.columns() {
            return Err(GameError::InvalidColumn);
        }
        let row = self.get_first_empty_row(col).ok_or(GameError::ColumnFull)?;
        let next_state = self
            .state
            .drop_piece(col, player, &self.geometry)
            .ok_or(GameError::ColumnFull)?;
        self.state = next_state;

        tracing::debug!(
            "Board drop: player {:?}, col {}, row {}, hash {:016X}",
            player,
            col,
            row,
            self.state.hash()
        );

        let chain_len = self.calculate_chain_length(col, row);
        match player {
            Player::Red => {
                self.stats.red_longest_chain = self.stats.red_longest_chain.max(chain_len);
            }
            Player::Yellow => {
                self.stats.yellow_longest_chain = self.stats.yellow_longest_chain.max(chain_len);
            }
        }
        Ok(row)
    }

    /// Returns the number of columns in this board.
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.geometry.columns()
    }

    /// Returns the number of rows in this board.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.geometry.rows()
    }

    /// Checks if the specified player has won the game.
    #[must_use]
    pub fn has_won(&self, player: Player) -> bool {
        self.state.has_won(player, &self.geometry)
    }

    /// Checks if the board is completely full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.state.is_full(&self.geometry)
    }

    /// Returns the stable 128-bit hash of the current board state.
    #[must_use]
    pub fn state_hash(&self) -> u128 {
        self.state.hash()
    }

    /// Returns the current board statistics.
    #[must_use]
    pub fn stats(&self) -> BoardStats {
        self.stats
    }

    /// Returns the cell state (Red, Yellow, or Empty) at a specific (col, row).
    #[must_use]
    pub fn get_cell(&self, col: u32, row: u32) -> Cell {
        self.state.get_cell(col, row, &self.geometry)
    }

    /// Returns the first empty row index in the specified column, if any.
    #[must_use]
    pub fn get_first_empty_row(&self, col: u32) -> Option<u32> {
        let h = self.rows() + 1;
        self.state
            .get_next_bit_index(col, &self.geometry)
            .map(|idx| idx % h)
    }

    /// Calculates the length of the longest chain passing through a specific cell.
    ///
    /// # Panics
    /// Panics if the bit index for the given cell exceeds `u32` capacity.
    #[must_use]
    pub fn calculate_chain_length(&self, col: u32, row: u32) -> u32 {
        let Cell::Occupied(player) = self.get_cell(col, row) else {
            return 0;
        };
        let h = self.rows() + 1;
        let bit_idx = col * h + row;
        self.state
            .calculate_chain_length_at(player, bit_idx, &self.geometry)
    }

    /// Analyzes the board for immediate and future threats to the given player.
    ///
    /// Evaluates if the opponent can win on their next move or if they have a
    /// potential Connect 3.
    #[must_use]
    pub fn get_threat_status(&self, player: Player) -> ThreatLevel {
        let opponent = player.other();
        let mut opponent_win_threat = false;
        let mut opponent_connect3_threat = false;

        for col in 0..self.columns() {
            if let Some((temp_state, bit_idx)) =
                self.state
                    .drop_piece_with_index(col, opponent, &self.geometry)
            {
                if temp_state.has_won(opponent, &self.geometry) {
                    opponent_win_threat = true;
                    break;
                }

                if temp_state.calculate_chain_length_at(opponent, bit_idx, &self.geometry) >= 3 {
                    opponent_connect3_threat = true;
                }
            }
        }

        if opponent_win_threat {
            ThreatLevel::Critical
        } else if opponent_connect3_threat {
            ThreatLevel::Caution
        } else {
            ThreatLevel::Stable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_chain_length_at() {
        let geo = DynamicBoardGeometry::new(7, 6);
        let mut state = DynamicBoardState::new(&geo);

        // 1. Horizontal test
        // Place: (2,0), (3,0), (4,0)
        state = state.drop_piece(2, Player::Red, &geo).unwrap();
        let res1 = state.drop_piece_with_index(3, Player::Red, &geo).unwrap();
        state = res1.0;
        let bit_mid = res1.1;
        state = state.drop_piece(4, Player::Red, &geo).unwrap();
        assert_eq!(
            state.calculate_chain_length_at(Player::Red, bit_mid, &geo),
            3
        );

        // 2. Vertical test
        // Place: (0,0), (0,1), (0,2), (0,3)
        let mut v_state = DynamicBoardState::new(&geo);
        v_state = v_state.drop_piece(0, Player::Yellow, &geo).unwrap();
        v_state = v_state.drop_piece(0, Player::Yellow, &geo).unwrap();
        let res2 = v_state
            .drop_piece_with_index(0, Player::Yellow, &geo)
            .unwrap();
        v_state = res2.0;
        let bit_v = res2.1;
        v_state = v_state.drop_piece(0, Player::Yellow, &geo).unwrap();
        assert_eq!(
            v_state.calculate_chain_length_at(Player::Yellow, bit_v, &geo),
            4
        );

        // 3. Diagonal (/) test
        let mut d1_state = DynamicBoardState::new(&geo);
        // Construct:
        // Col 0: R
        // Col 1: Y, R
        // Col 2: Y, Y, R
        d1_state = d1_state.drop_piece(0, Player::Red, &geo).unwrap();
        d1_state = d1_state.drop_piece(1, Player::Yellow, &geo).unwrap();
        d1_state = d1_state.drop_piece(1, Player::Red, &geo).unwrap();
        d1_state = d1_state.drop_piece(2, Player::Yellow, &geo).unwrap();
        d1_state = d1_state.drop_piece(2, Player::Yellow, &geo).unwrap();
        let res3 = d1_state
            .drop_piece_with_index(2, Player::Red, &geo)
            .unwrap();
        d1_state = res3.0;
        let bit_d1 = res3.1;
        assert_eq!(
            d1_state.calculate_chain_length_at(Player::Red, bit_d1, &geo),
            3
        );

        // 4. Diagonal (\) test
        let mut d2_state = DynamicBoardState::new(&geo);
        // Construct:
        // Col 0: Y, Y, R
        // Col 1: Y, R
        // Col 2: R
        d2_state = d2_state.drop_piece(0, Player::Yellow, &geo).unwrap();
        d2_state = d2_state.drop_piece(0, Player::Yellow, &geo).unwrap();
        let res4 = d2_state
            .drop_piece_with_index(0, Player::Red, &geo)
            .unwrap();
        d2_state = res4.0;
        let bit_d2 = res4.1;
        d2_state = d2_state.drop_piece(1, Player::Yellow, &geo).unwrap();
        d2_state = d2_state.drop_piece(1, Player::Red, &geo).unwrap();
        d2_state = d2_state.drop_piece(2, Player::Red, &geo).unwrap();
        assert_eq!(
            d2_state.calculate_chain_length_at(Player::Red, bit_d2, &geo),
            3
        );
    }

    #[test]
    fn test_horizontal_win() {
        let mut board = Board::new(7, 6);
        for i in 0..4 {
            board.drop_piece(i, Player::Red).unwrap();
        }
        assert!(board.has_won(Player::Red));
    }

    #[test]
    fn test_vertical_win() {
        let mut board = Board::new(7, 6);
        for _ in 0..4 {
            board.drop_piece(0, Player::Red).unwrap();
        }
        assert!(board.has_won(Player::Red));
    }

    #[test]
    fn test_is_full() {
        let mut board = Board::new(7, 6);
        for c in 0..7 {
            for _ in 0..6 {
                board.drop_piece(c, Player::Red).unwrap();
            }
        }
        assert!(board.is_full());
    }

    #[test]
    fn test_get_first_empty_row() {
        let mut board = Board::new(7, 6);
        assert_eq!(board.get_first_empty_row(0), Some(0));
        board.drop_piece(0, Player::Red).unwrap();
        assert_eq!(board.get_first_empty_row(0), Some(1));
        for _ in 0..5 {
            board.drop_piece(0, Player::Yellow).unwrap();
        }
        assert_eq!(board.get_first_empty_row(0), None);
    }

    #[test]
    fn test_get_cell() {
        let mut board = Board::new(7, 6);
        assert_eq!(board.get_cell(0, 0), Cell::Empty);
        board.drop_piece(0, Player::Red).unwrap();
        assert_eq!(board.get_cell(0, 0), Cell::Occupied(Player::Red));
        board.drop_piece(0, Player::Yellow).unwrap();
        assert_eq!(board.get_cell(0, 1), Cell::Occupied(Player::Yellow));
    }

    #[test]
    fn test_bitboard_win_directions() {
        let mut b1 = Board::new(7, 6);
        for _ in 0..4 {
            b1.drop_piece(0, Player::Red).unwrap();
        }
        assert!(b1.has_won(Player::Red));

        let mut b2 = Board::new(7, 6);
        for c in 0..4 {
            b2.drop_piece(c, Player::Yellow).unwrap();
        }
        assert!(b2.has_won(Player::Yellow));

        let mut b3 = Board::new(7, 6);
        for i in 0..4 {
            for _ in 0..i {
                b3.drop_piece(i, Player::Red).unwrap();
            }
            b3.drop_piece(i, Player::Yellow).unwrap();
        }
        assert!(b3.has_won(Player::Yellow));

        let mut b4 = Board::new(7, 6);
        for i in 0..4 {
            let col = 3 - i;
            for _ in 0..i {
                b4.drop_piece(col, Player::Red).unwrap();
            }
            b4.drop_piece(col, Player::Yellow).unwrap();
        }
        assert!(b4.has_won(Player::Yellow));
    }

    #[test]
    fn test_bitboard_consistency() {
        let mut board = Board::new(7, 6);
        let moves = [0, 1, 0, 1, 0, 1, 0];
        for (i, &col) in moves.iter().enumerate() {
            let player = if i % 2 == 0 {
                Player::Red
            } else {
                Player::Yellow
            };
            board.drop_piece(col, player).unwrap();
        }
        assert!(board.has_won(Player::Red));
    }

    #[test]
    fn test_bitboard_vertical_win() {
        let mut board = Board::new(7, 6);
        for _ in 0..4 {
            board.drop_piece(0, Player::Red).unwrap();
        }
        assert!(board.has_won(Player::Red));
    }

    #[test]
    fn test_bitboard_diagonal_win() {
        let mut board = Board::new(7, 6);
        board.drop_piece(0, Player::Red).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Red).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Red).unwrap();
        assert!(board.has_won(Player::Red));
    }

    #[test]
    fn test_full_column_bitwise() {
        let mut board = Board::new(7, 6);
        for _ in 0..6 {
            board.drop_piece(0, Player::Red).unwrap();
        }
        assert!(
            board.drop_piece(0, Player::Red).is_err(),
            "Drop should fail on full column"
        );
    }

    #[test]
    fn test_check_win_alternate() {
        let mut board = Board::new(7, 6);
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Red).unwrap();
        board.drop_piece(2, Player::Red).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(0, Player::Red).unwrap();
        assert!(
            board.has_won(Player::Red),
            "Should detect backslash diagonal win"
        );
    }

    #[test]
    #[should_panic(expected = "Board columns must be greater than 0")]
    fn test_geometry_zero_panic_bug() {
        let _ = BoardGeometry::<u128>::new(0, 6);
    }

    #[test]
    #[should_panic(expected = "Board size exceeds bitboard capacity")]
    fn test_board_geometry_overflow_safeguard() {
        // 9 columns * (14 rows + 1 sentinel) = 135 bits (> 128)
        // This is within MAX_COLUMNS (9) but exceeds capacity.
        let _ = BoardGeometry::<u128>::new(9, 14);
    }

    #[test]
    fn test_heatmap_symmetry_and_height() {
        // Test standard board (7x6)
        let geo = BoardGeometry::<u64>::new(7, 6);
        let h = geo.rows + 1;

        // Symmetry Check: Every cell (c, r) should be in the same tier as (cols-1-c, r)
        for c in 0..geo.columns {
            for r in 0..geo.rows {
                let bit = 1u64 << (c * h + r);
                let mir = 1u64 << ((geo.columns - 1 - c) * h + r);
                let t_bit = (0..geo.weight_masks.len()).find(|&i| (geo.weight_masks[i] & bit) != 0);
                let t_mir = (0..geo.weight_masks.len()).find(|&i| (geo.weight_masks[i] & mir) != 0);
                assert_eq!(t_bit, t_mir, "Asymmetry at Col {c}, Row {r}");
            }
        }

        // Height Preference: Cell (3,0) should have a higher (or equal) tier than (3,5)
        let bit_bot = 1u64 << (3 * h);
        let bit_top = 1u64 << (3 * h + 5);
        let tier_bot = (0..geo.weight_masks.len())
            .find(|&i| (geo.weight_masks[i] & bit_bot) != 0)
            .unwrap();
        let tier_top = (0..geo.weight_masks.len())
            .find(|&i| (geo.weight_masks[i] & bit_top) != 0)
            .unwrap();
        // Tier 0 is most valuable, so lower is better
        assert!(tier_bot <= tier_top);

        // Test even-width board (8x7)
        let geo_even = BoardGeometry::<u64>::new(8, 7);
        let h_even = geo_even.rows + 1;
        for c in 0..geo_even.columns {
            for r in 0..geo_even.rows {
                let bit = 1u64 << (c * h_even + r);
                let mir = 1u64 << ((geo_even.columns - 1 - c) * h_even + r);
                let t_bit = (0..geo_even.weight_masks.len())
                    .find(|&i| (geo_even.weight_masks[i] & bit) != 0);
                let t_mir = (0..geo_even.weight_masks.len())
                    .find(|&i| (geo_even.weight_masks[i] & mir) != 0);
                assert_eq!(t_bit, t_mir, "Asymmetry at Col {c}, Row {r} (Even Board)");
            }
        }
    }

    #[test]
    fn test_hash_symmetry_bug() {
        let mut b = Board::new(7, 6);
        // Player Red at (0,0), Player Yellow at (1,0)
        b.drop_piece(0, Player::Red).unwrap();
        b.drop_piece(1, Player::Yellow).unwrap();
        let h1 = b.state_hash();

        let mut b2 = Board::new(7, 6);
        // Swap colors: Player Yellow at (0,0), Player Red at (1,0)
        b2.drop_piece(0, Player::Yellow).unwrap();
        b2.drop_piece(1, Player::Red).unwrap();
        let h2 = b2.state_hash();

        assert_ne!(
            h1, h2,
            "Hash should distinguish between player colors at same positions"
        );
    }

    #[test]
    fn test_smallest_boards() {
        // 1x4 board: Only vertical wins possible
        let mut b1 = Board::new(1, 4);
        for _ in 0..3 {
            b1.drop_piece(0, Player::Red).unwrap();
        }
        assert!(!b1.has_won(Player::Red));
        b1.drop_piece(0, Player::Red).unwrap();
        assert!(b1.has_won(Player::Red));

        // 4x1 board: Only horizontal wins possible
        let mut b2 = Board::new(4, 1);
        for i in 0..3 {
            b2.drop_piece(i, Player::Yellow).unwrap();
        }
        assert!(!b2.has_won(Player::Yellow));
        b2.drop_piece(3, Player::Yellow).unwrap();
        assert!(b2.has_won(Player::Yellow));
    }

    #[test]
    fn test_max_board_size() {
        // 9x7 is the maximum size allowed by current bitboard capacity (9 * (7+1) = 72 bits < 128)
        let mut b = Board::new(9, 7);
        // Vertical win in the last column
        for _ in 0..4 {
            b.drop_piece(8, Player::Red).unwrap();
        }
        assert!(b.has_won(Player::Red));

        // Horizontal win across the middle
        let mut b2 = Board::new(9, 7);
        for i in 3..7 {
            b2.drop_piece(i, Player::Yellow).unwrap();
        }
        assert!(b2.has_won(Player::Yellow));
    }

    #[test]
    fn test_board_stats_live_update() {
        let mut b = Board::new(7, 6);
        assert_eq!(b.stats().red_longest_chain, 0);
        assert_eq!(b.stats().yellow_longest_chain, 0);

        b.drop_piece(0, Player::Red).unwrap();
        assert_eq!(b.stats().red_longest_chain, 1);

        b.drop_piece(0, Player::Red).unwrap();
        assert_eq!(b.stats().red_longest_chain, 2);

        b.drop_piece(1, Player::Yellow).unwrap();
        b.drop_piece(2, Player::Yellow).unwrap();
        b.drop_piece(3, Player::Yellow).unwrap();
        assert_eq!(b.stats().yellow_longest_chain, 3);

        b.drop_piece(0, Player::Red).unwrap();
        b.drop_piece(0, Player::Red).unwrap(); // Red wins
        assert_eq!(b.stats().red_longest_chain, 4);
    }

    #[test]
    fn test_is_full_complex_draw() {
        let mut b = Board::new(4, 4);
        // Fill a 4x4 board in a way that no one wins
        // R R Y Y
        // Y Y R R
        // R R Y Y
        // Y Y R R
        let pattern = [
            [Player::Yellow, Player::Yellow, Player::Red, Player::Red],
            [Player::Red, Player::Red, Player::Yellow, Player::Yellow],
            [Player::Yellow, Player::Yellow, Player::Red, Player::Red],
            [Player::Red, Player::Red, Player::Yellow, Player::Yellow],
        ];

        for row_pattern in &pattern {
            for (c, &p) in row_pattern.iter().enumerate() {
                b.drop_piece(u32::try_from(c).unwrap(), p).unwrap();
            }
        }

        assert!(b.is_full());
        assert!(!b.has_won(Player::Red));
        assert!(!b.has_won(Player::Yellow));
    }

    #[test]
    fn test_diagonal_boundary_wrapping_prevention() {
        // Ensure that pieces on the edge of columns don't "wrap" to form diagonal wins
        // across the sentinel bit.
        let mut b = Board::new(7, 6);
        // Place Red pieces at the top of col 0, 1, 2 etc.
        // This is a bit hard to construct manually but the goal is to test the bitboard shifts.

        // We use the internal state access for this specific test
        match (&mut b.state, &b.geometry) {
            (DynamicBoardState::Small(s), DynamicBoardGeometry::Small(g)) => {
                let h = g.rows + 1;
                s.red |= 1u64 << (h - 1); // Row 5, Col 0
                s.red |= 1u64 << (2 * h - 1); // Row 5, Col 1
                s.red |= 1u64 << (3 * h - 1); // Row 5, Col 2
            }
            (DynamicBoardState::Large(s), DynamicBoardGeometry::Large(g)) => {
                let h = g.rows + 1;
                s.red |= 1u128 << (h - 1); // Row 5, Col 0
                s.red |= 1u128 << (2 * h - 1); // Row 5, Col 1
                s.red |= 1u128 << (3 * h - 1); // Row 5, Col 2
            }
            _ => panic!("State/Geometry mismatch"),
        }

        assert!(!b.has_won(Player::Red));

        // Let's just set the bit directly for the win condition test
        match (&mut b.state, &b.geometry) {
            (DynamicBoardState::Small(s), DynamicBoardGeometry::Small(g)) => {
                let h = g.rows + 1;
                s.red |= 1u64 << (4 * h - 1); // Row 5, Col 3
            }
            (DynamicBoardState::Large(s), DynamicBoardGeometry::Large(g)) => {
                let h = g.rows + 1;
                s.red |= 1u128 << (4 * h - 1); // Row 5, Col 3
            }
            _ => panic!("State/Geometry mismatch"),
        }
        assert!(b.has_won(Player::Red));
    }

    #[test]
    fn test_bitboard_stress_random_moves() {
        use rand::RngExt;
        let mut rng = rand::rng();

        for _ in 0..100 {
            let cols = rng.random_range(5..=9);
            let rows = rng.random_range(5..=7);
            if cols * (rows + 1) > 128 {
                continue;
            }

            let mut b = Board::new(cols, rows);
            let mut moves = 0;
            let max_moves = cols * rows;

            while moves < max_moves {
                let col = rng.random_range(0..cols);
                let player = if moves % 2 == 0 {
                    Player::Red
                } else {
                    Player::Yellow
                };

                if b.drop_piece(col, player).is_ok() {
                    moves += 1;
                    if b.has_won(player) {
                        break;
                    }
                }
            }
            // Just ensuring it doesn't panic and state is consistent
            match (b.state, &b.geometry) {
                (DynamicBoardState::Small(s), DynamicBoardGeometry::Small(g)) => {
                    assert!(s.red & s.yellow == 0);
                    assert!((s.red | s.yellow) & !g.board_mask == 0);
                }
                (DynamicBoardState::Large(s), DynamicBoardGeometry::Large(g)) => {
                    assert!(s.red & s.yellow == 0);
                    assert!((s.red | s.yellow) & !g.board_mask == 0);
                }
                _ => panic!("State/Geometry variants must match"),
            }
        }
    }

    #[test]
    fn test_heatmap_values_in_bounds() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let all_masked = geo.weight_masks.iter().fold(0, |acc, &w| acc | w);
        assert_eq!(
            all_masked, geo.board_mask,
            "Heatmap masks must cover exactly the playable board"
        );

        for i in 0..geo.weight_masks.len() {
            for j in 0..geo.weight_masks.len() {
                if i != j {
                    assert_eq!(
                        geo.weight_masks[i] & geo.weight_masks[j],
                        0,
                        "Heatmap masks must be disjoint"
                    );
                }
            }
        }
    }

    #[test]
    fn test_u64_board_standard() {
        // Standard 7x6 board fits in u64 (7 * 7 = 49 bits)
        let mut board = Board::new(7, 6);
        for i in 0..4 {
            board.drop_piece(i, Player::Red).unwrap();
        }
        assert!(board.has_won(Player::Red));
        assert!(matches!(board.state, DynamicBoardState::Small(_)));
    }

    #[test]
    fn test_u64_board_large() {
        // Large 8x7 board fits in u64 (8 * 8 = 64 bits)
        let mut board = Board::new(8, 7);
        for i in 0..4 {
            board.drop_piece(i, Player::Yellow).unwrap();
        }
        assert!(board.has_won(Player::Yellow));
        assert!(matches!(board.state, DynamicBoardState::Small(_)));
    }
}

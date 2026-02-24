//! Core type definitions for the Connect 4 game engine and user interface.
//!
//! This module contains the fundamental data structures and traits used throughout
//! the application, including player representation, board dimensions, difficulty levels,
//! and the `Bitboard` trait which enables efficient game logic across different board sizes.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Trait defining the required bitwise operations for Connect 4 engine logic.
///
/// Supports both `u64` (Standard/Large) and `u128` (Giant) boards. This abstraction
/// allows the game engine to remain generic over the board's bit-level representation
/// while maintaining high performance through specialized bitwise operations.
pub trait Bitboard:
    Sized
    + Copy
    + Default
    + Eq
    + fmt::Debug
    + Send
    + Sync
    + 'static
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Shl<u32, Output = Self>
    + std::ops::Shr<u32, Output = Self>
    + std::ops::Shl<i32, Output = Self>
    + std::ops::Shr<i32, Output = Self>
    + std::ops::Shl<usize, Output = Self>
    + std::ops::Shr<usize, Output = Self>
    + std::ops::BitAndAssign
    + std::ops::BitOrAssign
{
    /// The total number of bits available in this bitboard type.
    const BITS: u32;

    /// Returns a bitboard with all bits set to zero.
    #[must_use]
    fn zero() -> Self;

    /// Returns a bitboard with only the least significant bit (bit 0) set to one.
    #[must_use]
    fn one() -> Self;

    /// Returns the number of bits set to one in the bitboard (population count).
    #[must_use]
    fn count_ones(self) -> u32;

    /// Returns the number of trailing zero bits in the bitboard.
    ///
    /// This is typically used to find the index of the first set bit.
    #[must_use]
    fn trailing_zeros(self) -> u32;

    /// Performs wrapping addition, discarding any overflow.
    #[must_use]
    fn wrapping_add(self, other: Self) -> Self;

    /// Performs wrapping subtraction, discarding any underflow.
    #[must_use]
    fn wrapping_sub(self, other: Self) -> Self;

    /// Converts the bitboard to a unified 128-bit representation.
    ///
    /// For `u64` implementations, this will zero-extend the value to 128 bits.
    #[must_use]
    fn to_u128(self) -> u128;
}

/// Implementation of the `Bitboard` trait for `u64`.
impl Bitboard for u64 {
    /// The number of bits in a `u64`.
    const BITS: u32 = u64::BITS;

    /// Returns 0.
    #[inline]
    fn zero() -> Self {
        0
    }
    /// Returns 1.
    #[inline]
    fn one() -> Self {
        1
    }
    /// Calls `u64::count_ones`.
    #[inline]
    fn count_ones(self) -> u32 {
        self.count_ones()
    }
    /// Calls `u64::trailing_zeros`.
    #[inline]
    fn trailing_zeros(self) -> u32 {
        self.trailing_zeros()
    }
    /// Calls `u64::wrapping_add`.
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    /// Calls `u64::wrapping_sub`.
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    /// Converts to `u128`.
    #[inline]
    fn to_u128(self) -> u128 {
        u128::from(self)
    }
}

/// Implementation of the `Bitboard` trait for `u128`.
impl Bitboard for u128 {
    /// The number of bits in a `u128`.
    const BITS: u32 = u128::BITS;

    /// Returns 0.
    #[inline]
    fn zero() -> Self {
        0
    }
    /// Returns 1.
    #[inline]
    fn one() -> Self {
        1
    }
    /// Calls `u128::count_ones`.
    #[inline]
    fn count_ones(self) -> u32 {
        self.count_ones()
    }
    /// Calls `u128::trailing_zeros`.
    #[inline]
    fn trailing_zeros(self) -> u32 {
        self.trailing_zeros()
    }
    /// Calls `u128::wrapping_add`.
    #[inline]
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    /// Calls `u128::wrapping_sub`.
    #[inline]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    /// Returns self as `u128`.
    #[inline]
    fn to_u128(self) -> u128 {
        self
    }
}

/// Represents the two players in a Connect 4 game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    /// The Red player, traditionally moves first.
    Red,
    /// The Yellow player, traditionally moves second.
    Yellow,
}

impl Player {
    /// An array containing all possible player variants.
    pub const ALL: [Self; 2] = [Self::Red, Self::Yellow];

    /// Returns the opponent of the current player.
    ///
    /// If the current player is Red, returns Yellow, and vice versa.
    #[must_use]
    pub fn other(self) -> Self {
        match self {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        }
    }

    /// Returns a 0-based index for the player.
    ///
    /// Red maps to 0 and Yellow maps to 1. This is useful for array indexing
    /// and UI menu selections.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Player::Red => 0,
            Player::Yellow => 1,
        }
    }
}

impl fmt::Display for Player {
    /// Formats the player for display, typically as "Red" or "Yellow".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::Red => write!(f, "Red"),
            Player::Yellow => write!(f, "Yellow"),
        }
    }
}

/// Represents the state of a single cell on the game board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// The cell is empty and can be played into.
    Empty,
    /// The cell is occupied by a player.
    Occupied(Player),
}

/// Statistics related to the current state of the game board.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BoardStats {
    /// The length of the longest contiguous chain of pieces for the Red player.
    pub red_longest_chain: u32,
    /// The length of the longest contiguous chain of pieces for the Yellow player.
    pub yellow_longest_chain: u32,
}

/// Defines the AI difficulty levels available in the game.
///
/// Each level corresponds to a specific search depth for the minimax algorithm.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Difficulty {
    /// Easiest level, suitable for beginners.
    Easy,
    /// Medium level, providing a moderate challenge.
    Medium,
    /// Hard level, suitable for experienced players.
    Hard,
    /// Expert level, very challenging.
    Expert,
    /// Grandmaster level, the highest difficulty utilizing maximum search depth.
    Grandmaster,
}

impl Difficulty {
    /// An array containing all possible difficulty variants.
    pub const ALL: [Self; 5] = [
        Self::Easy,
        Self::Medium,
        Self::Hard,
        Self::Expert,
        Self::Grandmaster,
    ];

    /// Returns the search depth associated with this difficulty level.
    ///
    /// Depths are retrieved from the global configuration.
    #[must_use]
    pub const fn depth(self) -> u32 {
        use crate::config;
        match self {
            Self::Easy => config::AI_EASY_DEPTH,
            Self::Medium => config::AI_MEDIUM_DEPTH,
            Self::Hard => config::AI_HARD_DEPTH,
            Self::Expert => config::AI_EXPERT_DEPTH,
            Self::Grandmaster => config::AI_GRANDMASTER_DEPTH,
        }
    }

    /// Returns a 0-based index for the difficulty level.
    ///
    /// This is useful for UI menu selections and mapping to other data structures.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Easy => 0,
            Self::Medium => 1,
            Self::Hard => 2,
            Self::Expert => 3,
            Self::Grandmaster => 4,
        }
    }
}

/// Represents the dimensions of the game board.
#[derive(PartialEq, Clone, Copy)]
pub struct BoardSize {
    /// The number of columns in the board.
    pub cols: u32,
    /// The number of rows in the board.
    pub rows: u32,
}

/// Represents the available game modes.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameMode {
    /// Single player mode against the AI.
    Single,
    /// Local two-player mode on the same machine.
    LocalTwo,
    /// Remote two-player mode over a network.
    Remote,
}

impl GameMode {
    /// An array containing all possible game mode variants.
    pub const ALL: [Self; 3] = [Self::Single, Self::LocalTwo, Self::Remote];

    /// Returns a 0-based index for the game mode.
    ///
    /// This is useful for UI menu selections.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Single => 0,
            Self::LocalTwo => 1,
            Self::Remote => 2,
        }
    }
}

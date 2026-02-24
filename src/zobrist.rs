//! # Zobrist Hashing
//!
//! Provides deterministic 64-bit Zobrist hashing for Connect 4 board states.
//! Zobrist hashing is an efficient way to represent a board state as a single
//! 64-bit integer, which is used for indexing into Transposition Tables.
//!
//! The hash can be updated incrementally in O(1) time when a piece is dropped
//! or removed, making it ideal for use during search.

use crate::game::BoardState;
use crate::types::{Bitboard, Player};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::sync::OnceLock;

/// Number of bits available in our bitboard representation.
/// We use 128 to accommodate the largest supported board (Giant 9x7).
const MAX_BIT_POSITIONS: usize = 128;

/// Global Zobrist constants for players and bit positions.
/// We use `OnceLock` to ensure deterministic initialization on first access.
static ZOBRIST: OnceLock<ZobristTable> = OnceLock::new();

/// Zobrist constants for pieces.
/// This table stores a unique 64-bit random number for each possible piece position
/// (bit index) for each player.
pub struct ZobristTable {
    /// Random constants for pieces: `[PlayerIndex][BitPosition]`
    /// Player index 0 is Red, 1 is Yellow.
    pub pieces: [[u64; MAX_BIT_POSITIONS]; 2],
}

/// Initializes the global Zobrist constants using a fixed-seed PRNG.
/// This ensures deterministic hashing across different runs and sessions.
///
/// The seed is "Connect4" in hexadecimal (`0x436F_6E6E_6563_7434`).
fn get_table() -> &'static ZobristTable {
    ZOBRIST.get_or_init(|| {
        tracing::debug!("Initializing global Zobrist constants");
        let mut rng = SmallRng::seed_from_u64(0x436F_6E6E_6563_7434); // Seed: "Connect4" in hex
        let mut pieces = [[0u64; MAX_BIT_POSITIONS]; 2];

        for player_pieces in &mut pieces {
            for bit_constant in player_pieces.iter_mut() {
                *bit_constant = rng.random();
            }
        }

        ZobristTable { pieces }
    })
}

/// Computes the full 64-bit Zobrist hash of a given board state from scratch.
///
/// This involves iterating over all set bits for both players. While correct,
/// it is slower than incremental updates. Typically used at the start of a
/// search (root node) or for verification.
#[must_use]
pub fn compute_hash<T: Bitboard>(state: &BoardState<T>) -> u64 {
    let table = get_table();
    let mut hash = 0u64;

    // XOR Red pieces
    let mut red = state.bits(Player::Red);
    while red != T::zero() {
        let bit_index = red.trailing_zeros();
        hash ^= table.pieces[0][bit_index as usize];
        red &= red.wrapping_sub(T::one()); // Clear lowest set bit
    }

    // XOR Yellow pieces
    let mut yellow = state.bits(Player::Yellow);
    while yellow != T::zero() {
        let bit_index = yellow.trailing_zeros();
        hash ^= table.pieces[1][bit_index as usize];
        yellow &= yellow.wrapping_sub(T::one()); // Clear lowest set bit
    }

    hash
}

/// Performs an incremental XOR update to a hash for a single move.
///
/// This is an O(1) operation, ideal for use inside the search recursion.
/// It XORs the existing hash with the constant for the given player at the
/// specified bit index. Since XOR is its own inverse, calling this twice with
/// the same parameters will effectively "undo" the move in the hash.
#[must_use]
pub fn apply_move(hash: u64, player: Player, bit_index: u32) -> u64 {
    let table = get_table();
    let player_idx = match player {
        Player::Red => 0,
        Player::Yellow => 1,
    };

    hash ^ table.pieces[player_idx][bit_index as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::BoardGeometry;

    #[test]
    fn test_zobrist_determinism() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut state = BoardState::<u64>::default();
        state = state.drop_piece(0, Player::Red, &geo).unwrap();
        state = state.drop_piece(1, Player::Yellow, &geo).unwrap();
        let h1 = compute_hash(&state);
        let h2 = compute_hash(&state);
        assert_eq!(h1, h2, "Hash must be deterministic for the same state");
    }

    #[test]
    fn test_zobrist_incremental_update() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut state = BoardState::<u64>::default();
        let mut hash = compute_hash(&state);
        assert_eq!(hash, 0, "Empty board hash should be 0");

        // Simulate a sequence of moves
        let moves = [(3, Player::Red), (4, Player::Yellow), (3, Player::Red)];

        for (col, player) in moves {
            // Get the bit mask of where the piece will land
            let next_bit = state.get_next_bit(col, &geo).expect("Move should be valid");
            let bit_index = next_bit.trailing_zeros();

            // Perform incremental update
            hash = apply_move(hash, player, bit_index);

            // Apply move to state for verification
            state = state
                .drop_piece(col, player, &geo)
                .expect("Move should be valid");

            // Verify that incremental hash matches full recomputation
            assert_eq!(
                hash,
                compute_hash(&state),
                "Incremental hash must match full recomputation for move in col {col}"
            );
        }
    }

    #[test]
    fn test_zobrist_exhaustive_random_play() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut rng = SmallRng::seed_from_u64(0xDEAD_C0DE);

        // Run 100 random games to completion
        for _ in 0..100 {
            let mut state = BoardState::<u64>::default();
            let mut hash = compute_hash(&state);
            let mut current_player = Player::Red;

            // Up to 42 moves in a standard game
            for _ in 0..42 {
                // Find all legal moves
                let mut legal_moves = Vec::new();
                for col in 0..geo.columns {
                    if state.get_next_bit(col, &geo).is_some() {
                        legal_moves.push(col);
                    }
                }

                if legal_moves.is_empty() {
                    break;
                }

                // Pick a random legal move
                let col = legal_moves[rng.random_range(0..legal_moves.len())];

                // 1. Calculate next bit and index
                let next_bit = state.get_next_bit(col, &geo).unwrap();
                let bit_index = next_bit.trailing_zeros();

                // 2. Incremental update
                hash = apply_move(hash, current_player, bit_index);

                // 3. Apply to state
                state = state.drop_piece(col, current_player, &geo).unwrap();

                // 4. Verify
                assert_eq!(
                    hash,
                    compute_hash(&state),
                    "Incremental hash mismatch at move {col} for state: {state:?}"
                );

                current_player = current_player.other();

                if state.has_won(Player::Red, &geo) || state.has_won(Player::Yellow, &geo) {
                    break;
                }
            }
        }
    }

    #[test]
    fn test_zobrist_no_collisions_simple() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let s1 = BoardState::<u64>::default()
            .drop_piece(3, Player::Red, &geo)
            .unwrap();
        let s2 = BoardState::<u64>::default()
            .drop_piece(4, Player::Red, &geo)
            .unwrap();

        assert_ne!(
            compute_hash(&s1),
            compute_hash(&s2),
            "Different boards should have different hashes (collision detected)"
        );
    }
}

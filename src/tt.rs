//! This module provides a high-performance, two-tier transposition table (TT) for caching search results
//! in the Connect 4 engine.
//!
//! The transposition table is a key component in game AI that avoids redundant computations by
//! storing the evaluation of previously searched positions.
//!
//! # Replacement Strategy
//! The table uses a "two-tier" or "two-slot" replacement strategy:
//! 1. **Depth-Preferred Slot**: Stores the entry with the highest search depth for a given hash index.
//!    This ensures that expensive, high-quality searches are preserved.
//! 2. **Always-Replace Slot**: Stores the most recent entry for a given hash index.
//!    This provides temporal locality, ensuring that the table stays relevant to the current search path.
//!
//! # Collision Protection
//! To prevent "aliasing" (where two different positions share the same hash and lead to incorrect
//! evaluations), each entry stores the full `BoardState`. This guarantees 100% correctness at the
//! cost of a slightly larger memory footprint.
//!
//! # Threading and Pooling
//! Since large tables are expensive to allocate and clear, this module provides thread-local
//! pooling through the `TTManaged` trait and `TTGuard` RAII wrapper. This allows for zero-allocation
//! reuse of tables across multiple search sessions in the same thread.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::game::BoardState;
use crate::types::Bitboard;

/// Flags representing the nature of the score stored in the Transposition Table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    /// The score is an exact evaluation of the position at the given depth.
    Exact,
    /// The score is an upper bound (the true score is less than or equal to this).
    /// Occurs when a search failed high (beta cut-off).
    LowerBound,
    /// The score is a lower bound (the true score is greater than or equal to this).
    /// Occurs when a search failed low (all moves were worse than alpha).
    UpperBound,
}

/// Trait bridging generic transposition table logic with type-specific thread-local pools.
///
/// This trait is required because Rust's `thread_local!` statics cannot be generic over types
/// with different memory layouts. Implementing this trait for `u64` and `u128` allows the engine
/// to manage pools for both standard and large board sizes.
pub trait TTManaged: Bitboard {
    /// Returns a transposition table to the type-specific thread-local pool.
    ///
    /// # Arguments
    /// * `size_mb` - The capacity of the table in megabytes, used for segregation in the pool.
    /// * `tt` - The table instance to return to the pool.
    fn pool_push(size_mb: usize, tt: TranspositionTable<Self>);

    /// Attempts to retrieve a transposition table from the type-specific thread-local pool.
    ///
    /// # Arguments
    /// * `size_mb` - The desired capacity of the table in megabytes.
    ///
    /// # Returns
    /// `Some(TranspositionTable)` if a matching table exists in the pool, `None` otherwise.
    fn pool_take(size_mb: usize) -> Option<TranspositionTable<Self>>;
}

thread_local! {
    /// Thread-local pool for transposition tables using `u64` bitboards.
    static TT_POOL_U64: RefCell<Vec<(usize, TranspositionTable<u64>)>> = const { RefCell::new(Vec::new()) };

    /// Thread-local pool for transposition tables using `u128` bitboards.
    static TT_POOL_U128: RefCell<Vec<(usize, TranspositionTable<u128>)>> = const { RefCell::new(Vec::new()) };
}

impl TTManaged for u64 {
    /// Pushes a `u64` table into the thread-local pool.
    fn pool_push(size_mb: usize, tt: TranspositionTable<Self>) {
        TT_POOL_U64.with(|pool| {
            pool.borrow_mut().push((size_mb, tt));
        });
    }

    /// Takes a `u64` table from the thread-local pool if one with matching size exists.
    fn pool_take(size_mb: usize) -> Option<TranspositionTable<Self>> {
        TT_POOL_U64.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.iter()
                .position(|(s, _)| *s == size_mb)
                .map(|idx| pool.remove(idx).1)
        })
    }
}

impl TTManaged for u128 {
    /// Pushes a `u128` table into the thread-local pool.
    fn pool_push(size_mb: usize, tt: TranspositionTable<Self>) {
        TT_POOL_U128.with(|pool| {
            pool.borrow_mut().push((size_mb, tt));
        });
    }

    /// Takes a `u128` table from the thread-local pool if one with matching size exists.
    fn pool_take(size_mb: usize) -> Option<TranspositionTable<Self>> {
        TT_POOL_U128.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.iter()
                .position(|(s, _)| *s == size_mb)
                .map(|idx| pool.remove(idx).1)
        })
    }
}

/// A single entry in the Transposition Table.
///
/// Optimized to be compact while ensuring 100% collision protection by storing the full `BoardState`.
#[derive(Clone, Copy, Debug)]
pub struct TTEntry<T: Bitboard> {
    /// The full board state to guarantee no hash collisions (aliasing protection).
    pub state: BoardState<T>,
    /// The evaluation score for this position in centipawns or mate-distance.
    pub score: i32,
    /// The depth to which this position was searched (remaining plies to leaf).
    pub depth: u32,
    /// Whether the score is Exact, an Alpha bound (`LowerBound`), or a Beta bound (`UpperBound`).
    pub flag: TTFlag,
    /// The best move found at this node, used to seed move ordering in deeper searches.
    pub best_move: Option<u32>,
    /// The generation when this entry was stored. Used for O(1) table clearing.
    pub generation: u32,
}

impl<T: Bitboard> Default for TTEntry<T> {
    /// Creates a default, empty entry.
    /// Note: `generation` is 0, which is always considered empty/stale.
    fn default() -> Self {
        Self {
            state: BoardState::default(),
            score: 0,
            depth: 0,
            flag: TTFlag::Exact,
            best_move: None,
            generation: 0, // Generation 0 is always considered empty
        }
    }
}

/// A bucket in the Transposition Table containing two slots for replacement strategy.
#[derive(Clone, Copy, Debug, Default)]
struct TTBucket<T: Bitboard> {
    /// Slot 1: Depth-Preferred (stores the most expensive search for this index).
    slot_deep: TTEntry<T>,
    /// Slot 2: Always-Replace (stores the most recent search for this index).
    slot_recent: TTEntry<T>,
}

/// Statistics for the Transposition Table to monitor search efficiency and health.
#[derive(Clone, Copy, Debug, Default)]
pub struct TTStats {
    /// Total number of lookup attempts.
    pub lookups: u64,
    /// Number of hits in the "Deep" (preferred) slot.
    pub hits_deep: u64,
    /// Number of hits in the "Recent" (replacement) slot.
    pub hits_recent: u64,
    /// Total number of store attempts.
    pub stores: u64,
    /// Number of times an existing entry from the current generation was overwritten.
    pub overwrites: u64,
}

/// Thread-safe atomic counters for gathering TT statistics during search.
#[derive(Default)]
struct TTStatsAtomics {
    /// Atomic counter for total lookups.
    lookups: AtomicU64,
    /// Atomic counter for hits in the deep slot.
    hits_deep: AtomicU64,
    /// Atomic counter for hits in the recent slot.
    hits_recent: AtomicU64,
    /// Atomic counter for total stores.
    stores: AtomicU64,
    /// Atomic counter for overwrites.
    overwrites: AtomicU64,
}

/// A fixed-size Two-Tier Transposition Table (TT) for caching search results.
///
/// Uses a power-of-two bucket count for fast masking and a Boxed Slice for lean metadata.
/// The table implements a two-tier replacement policy to balance search quality (depth)
/// with temporal locality (recency).
pub struct TranspositionTable<T: Bitboard> {
    /// The underlying storage for the table entries.
    table: Box<[TTBucket<T>]>,
    /// Mask used to map hashes to bucket indices (capacity - 1).
    mask: usize,
    /// The current generation ID. Entries with a lower generation are treated as empty.
    current_generation: u32,
    /// Atomic statistics for monitoring search performance.
    stats: TTStatsAtomics,
}

/// A RAII guard that returns a Transposition Table to the thread-local pool when dropped.
///
/// This allows the engine to acquire a large TT at the start of a move search and ensure
/// it is returned to the pool for reuse by future searches in the same thread.
pub struct TTGuard<T: TTManaged> {
    /// The wrapped transposition table.
    table: Option<TranspositionTable<T>>,
    /// The size in megabytes, used to return the table to the correct pool bin.
    size_mb: usize,
}

impl<T: TTManaged> Deref for TTGuard<T> {
    type Target = TranspositionTable<T>;

    /// Dereferences the guard to access the underlying table.
    fn deref(&self) -> &Self::Target {
        self.table.as_ref().expect("TTGuard used after drop")
    }
}

impl<T: TTManaged> DerefMut for TTGuard<T> {
    /// Mutably dereferences the guard to access the underlying table.
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.table.as_mut().expect("TTGuard used after drop")
    }
}

impl<T: TTManaged> Drop for TTGuard<T> {
    /// Automatically returns the table to the thread-local pool when the guard is dropped.
    fn drop(&mut self) {
        if let Some(tt) = self.table.take() {
            T::pool_push(self.size_mb, tt);
        }
    }
}

impl<T: Bitboard> TranspositionTable<T> {
    /// Creates a new Two-Tier Transposition Table with approximately the requested size in Megabytes.
    ///
    /// The actual size will be the nearest power of two that fits within or slightly exceeds
    /// the requested size to maximize indexing performance via bitwise masking.
    ///
    /// # Arguments
    /// * `size_mb` - The requested capacity in megabytes.
    #[must_use]
    pub fn new(size_mb: usize) -> Self {
        let bucket_size = std::mem::size_of::<TTBucket<T>>();
        let desired_count = (size_mb * 1024 * 1024) / bucket_size;

        // Ensure count is a power of two for bitwise masking.
        // next_power_of_two() for 0 returns 1.
        let count = desired_count.next_power_of_two();
        let mask = count - 1;

        tracing::info!(
            "Initializing Transposition Table: requested={}MB, actual_entries={}, bucket_size={} bytes",
            size_mb,
            count * 2,
            bucket_size
        );

        Self {
            table: vec![TTBucket::default(); count].into_boxed_slice(),
            mask,
            current_generation: 1, // Start at 1, as 0 is reserved for empty
            stats: TTStatsAtomics::default(),
        }
    }

    /// Clears all entries in the table by incrementing the generation counter and resetting stats.
    ///
    /// This is an O(1) operation because entries from old generations are ignored during lookups.
    /// It is typically called between games to prevent cross-contamination. If the generation
    /// counter overflows, a full O(N) clear of the underlying memory is performed.
    pub fn reset(&mut self) {
        tracing::debug!(
            "Resetting Transposition Table (generation {} -> {})",
            self.current_generation,
            self.current_generation.wrapping_add(1)
        );
        self.reset_stats();
        if self.current_generation == u32::MAX {
            // Rare overflow: reset everything
            self.table.fill(TTBucket::default());
            self.current_generation = 1;
        } else {
            self.current_generation += 1;
        }
    }

    /// Retrieves an entry from the table if it matches the provided board state and current generation.
    ///
    /// Checks both the "Deep" and "Recent" slots in the bucket indexed by the hash.
    ///
    /// # Arguments
    /// * `state` - The current board state to match against.
    /// * `hash` - The pre-calculated Zobrist hash for fast indexing.
    ///
    /// # Returns
    /// `Some(TTEntry)` if a valid match is found, `None` otherwise.
    #[must_use]
    pub fn lookup(&self, state: &BoardState<T>, hash: u64) -> Option<TTEntry<T>> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        let index = (hash as usize) & self.mask;
        let bucket = &self.table[index];

        // Check the Deep slot first (more likely to have useful cutoff data)
        if bucket.slot_deep.generation == self.current_generation
            && bucket.slot_deep.state == *state
        {
            self.stats.hits_deep.fetch_add(1, Ordering::Relaxed);
            return Some(bucket.slot_deep);
        }

        // Check the Recent slot
        if bucket.slot_recent.generation == self.current_generation
            && bucket.slot_recent.state == *state
        {
            self.stats.hits_recent.fetch_add(1, Ordering::Relaxed);
            return Some(bucket.slot_recent);
        }

        None
    }

    /// Stores a new entry in the table using a Two-Tier replacement strategy.
    ///
    /// # Replacement Logic
    /// 1. **State Match**: If the state exists in `slot_deep`, update it if `new_depth >= old_depth`.
    /// 2. **State Match**: If the state exists in `slot_recent`, update it if `new_depth >= old_depth`.
    ///    After updating, if the entry is now deeper than the `slot_deep` entry, they are swapped.
    /// 3. **Replacement**: If no match, store in `slot_deep` if the new search is strictly deeper than
    ///    the existing deep search (or if the existing search is from an old generation), demoting
    ///    the old deep entry to the `slot_recent` slot.
    /// 4. **Replacement**: Otherwise, store in the `slot_recent` slot (Always-Replace policy).
    ///
    /// # Arguments
    /// * `state` - The board state being stored.
    /// * `hash` - The Zobrist hash of the state.
    /// * `score` - The evaluation score.
    /// * `depth` - The depth of the search.
    /// * `flag` - The bound type (`Exact`, `LowerBound`, `UpperBound`).
    /// * `best_move` - The best move found at this node, if any.
    ///
    /// # Panics
    /// Panics if internal invariants are violated, specifically if `slot_recent` is populated
    /// with a current generation entry but `slot_deep` is not.
    pub fn store(
        &mut self,
        state: BoardState<T>,
        hash: u64,
        score: i32,
        depth: u32,
        flag: TTFlag,
        best_move: Option<u32>,
    ) {
        self.stats.stores.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        let index = (hash as usize) & self.mask;
        let bucket = &mut self.table[index];
        let new_entry = TTEntry {
            state,
            score,
            depth,
            flag,
            best_move,
            generation: self.current_generation,
        };

        // 1. Check for match in the Deep slot
        if bucket.slot_deep.generation == self.current_generation && bucket.slot_deep.state == state
        {
            // Refresh if depth is >=. Newer searches at the same depth might have better bounds.
            if depth >= bucket.slot_deep.depth {
                bucket.slot_deep = new_entry;
            }
            return;
        }

        // 2. Check for match in the Recent slot
        if bucket.slot_recent.generation == self.current_generation
            && bucket.slot_recent.state == state
        {
            if depth >= bucket.slot_recent.depth {
                bucket.slot_recent = new_entry;
            }
            // Promotion: if recent is now deeper than deep, swap them.
            // slot_deep is guaranteed to be valid since it is always filled
            // before slot_recent
            if bucket.slot_recent.depth > bucket.slot_deep.depth {
                assert_eq!(bucket.slot_deep.generation, self.current_generation);
                std::mem::swap(&mut bucket.slot_deep, &mut bucket.slot_recent);
            }
            return;
        }

        // 3. No state match: Traditional Two-Tier Replacement
        // If slot_deep is from an old generation, overwrite it immediately.
        if bucket.slot_deep.generation != self.current_generation {
            // Table was empty (or logically cleared) at this index, use the primary slot.
            bucket.slot_deep = new_entry;
        } else if depth > bucket.slot_deep.depth {
            // New deepest search: demote current deep entry to recent slot
            if bucket.slot_recent.generation == self.current_generation {
                self.stats.overwrites.fetch_add(1, Ordering::Relaxed);
            }
            bucket.slot_recent = bucket.slot_deep;
            bucket.slot_deep = new_entry;
        } else {
            // Otherwise, always replace the recent slot (Temporal Locality).
            if bucket.slot_recent.generation == self.current_generation {
                self.stats.overwrites.fetch_add(1, Ordering::Relaxed);
            }
            bucket.slot_recent = new_entry;
        }
    }

    /// Returns the total number of entry slots available in the table.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.table.len() * 2
    }

    /// Returns a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> TTStats {
        TTStats {
            lookups: self.stats.lookups.load(Ordering::Relaxed),
            hits_deep: self.stats.hits_deep.load(Ordering::Relaxed),
            hits_recent: self.stats.hits_recent.load(Ordering::Relaxed),
            stores: self.stats.stores.load(Ordering::Relaxed),
            overwrites: self.stats.overwrites.load(Ordering::Relaxed),
        }
    }

    /// Resets all statistics to zero.
    fn reset_stats(&self) {
        self.stats.lookups.store(0, Ordering::Relaxed);
        self.stats.hits_deep.store(0, Ordering::Relaxed);
        self.stats.hits_recent.store(0, Ordering::Relaxed);
        self.stats.stores.store(0, Ordering::Relaxed);
        self.stats.overwrites.store(0, Ordering::Relaxed);
    }
}

impl<T: TTManaged> TranspositionTable<T> {
    /// Gets a pooled `TranspositionTable` of the requested size.
    ///
    /// If no table is available in the pool, a new one is allocated.
    /// The table is automatically cleared (via `reset()`) before being returned to ensure
    /// a fresh state for the caller.
    ///
    /// # Arguments
    /// * `size_mb` - The desired capacity in megabytes.
    ///
    /// # Returns
    /// A `TTGuard` wrapping the table.
    #[must_use]
    pub fn get_pooled(size_mb: usize) -> TTGuard<T> {
        let mut tt = T::pool_take(size_mb).unwrap_or_else(|| TranspositionTable::new(size_mb));
        tt.reset();
        TTGuard {
            table: Some(tt),
            size_mb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{BoardGeometry, BoardState};
    use crate::types::Player;
    use crate::zobrist;

    /// Returns 3 distinct `BoardStates` for deterministic testing.
    fn get_states() -> (BoardState<u64>, BoardState<u64>, BoardState<u64>) {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let s0 = BoardState::<u64>::default();
        let s1 = s0.drop_piece(0, Player::Red, &geo).unwrap();
        let s2 = s0.drop_piece(1, Player::Red, &geo).unwrap();
        let s3 = s0.drop_piece(2, Player::Red, &geo).unwrap();
        (s1, s2, s3)
    }

    #[test]
    fn test_tt_basic_two_tier() {
        // Use a 0MB table to force a size of 1 bucket (all entries collide)
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);

        // Store S1 at depth 10 (goes to Deep as the bucket is empty)
        tt.store(s1, h1, 100, 10, TTFlag::Exact, None);
        // Store S2 at depth 5 (goes to Recent as 5 <= 10)
        tt.store(s2, h2, 50, 5, TTFlag::Exact, None);

        assert_eq!(tt.lookup(&s1, h1).unwrap().depth, 10);
        assert_eq!(tt.lookup(&s2, h2).unwrap().depth, 5);

        // Verify internal bucket placement
        let bucket = tt.table[0];
        assert_eq!(bucket.slot_deep.state, s1);
        assert_eq!(bucket.slot_recent.state, s2);
    }

    #[test]
    fn test_tt_demotion_logic() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        tt.store(s1, h1, 100, 10, TTFlag::Exact, None); // Deep: S1(10)
        tt.store(s2, h2, 50, 20, TTFlag::Exact, None); // New deeper S2(20) demotes S1(10) to Recent

        let bucket = tt.table[0];
        assert_eq!(bucket.slot_deep.state, s2);
        assert_eq!(bucket.slot_deep.depth, 20);
        assert_eq!(bucket.slot_recent.state, s1);
        assert_eq!(bucket.slot_recent.depth, 10);

        // Store S3 at depth 15. It's not deeper than S2(20), so it should replace S1(10) in Recent.
        tt.store(s3, h3, 75, 15, TTFlag::Exact, None);
        let bucket2 = tt.table[0];
        assert_eq!(bucket2.slot_deep.state, s2);
        assert_eq!(bucket2.slot_recent.state, s3);
    }

    #[test]
    fn test_tt_promotion_on_match() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);

        tt.store(s1, h1, 100, 10, TTFlag::Exact, None); // Deep: S1(10)
        tt.store(s2, h2, 50, 5, TTFlag::Exact, None); // Recent: S2(5)

        // Update S2 to depth 15. Should be promoted to Deep, demoting S1.
        tt.store(s2, h2, 60, 15, TTFlag::Exact, None);

        let bucket = tt.table[0];
        assert_eq!(bucket.slot_deep.state, s2);
        assert_eq!(bucket.slot_deep.depth, 15);
        assert_eq!(bucket.slot_recent.state, s1);
    }

    #[test]
    fn test_tt_bucket_saturation() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        // 1. Fill bucket: Deep=S2(20), Recent=S1(10)
        tt.store(s1, h1, 10, 10, TTFlag::Exact, None);
        tt.store(s2, h2, 20, 20, TTFlag::Exact, None);

        // 2. Add S3(15): Should replace S1(10) in Recent slot (Always-Replace policy)
        tt.store(s3, h3, 30, 15, TTFlag::Exact, None);
        assert_eq!(tt.lookup(&s2, h2).unwrap().state, s2);
        assert_eq!(tt.lookup(&s3, h3).unwrap().state, s3);
        assert!(tt.lookup(&s1, h1).is_none());

        // 3. Store S1(25): S1(25) is a "new" state (displaced from bucket earlier).
        // It's deeper than S2(20), so it replaces S2 in Deep and demotes S2 to Recent.
        tt.store(s1, h1, 40, 25, TTFlag::Exact, None);
        let bucket = tt.table[0];
        assert_eq!(bucket.slot_deep.state, s1);
        assert_eq!(bucket.slot_recent.state, s2);
    }

    #[test]
    fn test_tt_same_state_refresh() {
        let mut tt = TranspositionTable::<u64>::new(1);
        let (s1, _, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);

        // Test refresh in Deep slot: Same depth but better evaluation data
        tt.store(s1, h1, 100, 10, TTFlag::LowerBound, None);
        tt.store(s1, h1, 110, 10, TTFlag::Exact, Some(3));
        assert_eq!(tt.lookup(&s1, h1).unwrap().score, 110);
        assert_eq!(tt.lookup(&s1, h1).unwrap().flag, TTFlag::Exact);

        // Test refresh in Recent slot
        let (s1, s2, _) = get_states();
        let h1_2 = zobrist::compute_hash(&s1);
        let h2_2 = zobrist::compute_hash(&s2);
        let mut tt2 = TranspositionTable::<u64>::new(0);
        tt2.store(s2, h2_2, 200, 20, TTFlag::Exact, None); // Deep: S2
        tt2.store(s1, h1_2, 100, 10, TTFlag::LowerBound, None); // Recent: S1

        tt2.store(s1, h1_2, 110, 10, TTFlag::Exact, Some(3));
        assert_eq!(tt2.lookup(&s1, h1_2).unwrap().score, 110);
    }

    #[test]
    fn test_tt_collision_replacement() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        tt.store(s1, h1, 10, 10, TTFlag::Exact, None); // Deep: S1(10)
        tt.store(s2, h2, 20, 20, TTFlag::Exact, None); // Deep: S2(20), Recent: S1(10)

        // S3 at depth 5 should overwrite Recent (S1)
        tt.store(s3, h3, 30, 5, TTFlag::Exact, None);
        assert!(tt.lookup(&s1, h1).is_none());
        assert_eq!(tt.lookup(&s3, h3).unwrap().depth, 5);
        assert_eq!(tt.lookup(&s2, h2).unwrap().depth, 20);
    }

    #[test]
    fn test_tt_exhaustive_edge_cases() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        // Edge Case 1: Refuse Shallower Refresh
        tt.store(s1, h1, 100, 10, TTFlag::Exact, None);
        tt.store(s1, h1, 50, 5, TTFlag::LowerBound, None); // Should be ignored
        let entry = tt.lookup(&s1, h1).unwrap();
        assert_eq!(entry.depth, 10);
        assert_eq!(entry.score, 100);

        // Edge Case 2: Recent Match Update without Promotion
        tt.store(s2, h2, 200, 20, TTFlag::Exact, None); // Deep: S2(20)
        tt.store(s3, h3, 50, 5, TTFlag::Exact, None); // Recent: S3(5)

        tt.store(s3, h3, 75, 10, TTFlag::Exact, None); // Update S3(10)
        let bucket = tt.table[0];
        assert_eq!(bucket.slot_deep.state, s2, "S2 should still be Deep");
        assert_eq!(bucket.slot_recent.state, s3, "S3 should still be Recent");
        assert_eq!(bucket.slot_recent.depth, 10);

        // Edge Case 3: New State same depth as Deep (Tie-breaker)
        // Scenario: Deep is S2(20). We store S1(20).
        // Logic: depth > deep.depth is false, so it should replace Recent (S3).
        tt.store(s1, h1, 300, 20, TTFlag::Exact, None);
        let bucket2 = tt.table[0];
        assert_eq!(
            bucket2.slot_deep.state, s2,
            "Deep S2(20) should be preserved on tie"
        );
        assert_eq!(
            bucket2.slot_recent.state, s1,
            "New S1(20) should replace Recent"
        );

        // Edge Case 4: Zero capacity request
        let tt_zero = TranspositionTable::<u64>::new(0);
        assert_eq!(tt_zero.capacity(), 2); // 1 bucket * 2 slots
    }

    #[test]
    fn test_tt_clear() {
        let mut tt = TranspositionTable::<u64>::new(1);
        let (s1, _, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        tt.store(s1, h1, 100, 5, TTFlag::Exact, None);
        assert!(tt.lookup(&s1, h1).is_some());

        tt.reset();
        assert!(tt.lookup(&s1, h1).is_none());

        // Verify we can store and find again in the new generation
        tt.store(s1, h1, 200, 5, TTFlag::Exact, None);
        let entry = tt.lookup(&s1, h1).unwrap();
        assert_eq!(entry.score, 200);
        assert_eq!(entry.generation, 2);
    }

    #[test]
    fn test_tt_generation_overflow() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, _, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);

        // Force generation to near overflow
        tt.current_generation = u32::MAX;
        tt.store(s1, h1, 100, 5, TTFlag::Exact, None);
        assert!(tt.lookup(&s1, h1).is_some());

        // Trigger overflow clear
        tt.reset();
        assert_eq!(tt.current_generation, 1);
        assert!(tt.lookup(&s1, h1).is_none());

        // Ensure generation 0 was wiped (if any existed)
        // Manual check of bucket state
        assert_eq!(tt.table[0].slot_deep.generation, 0);
        assert_eq!(tt.table[0].slot_recent.generation, 0);
    }

    #[test]
    fn test_tt_stale_entry_overwriting() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);

        // Gen 1: Store S1 deep
        tt.store(s1, h1, 100, 20, TTFlag::Exact, None);
        assert_eq!(tt.table[0].slot_deep.state, s1);

        // Clear to Gen 2
        tt.reset();

        // Store S2 shallow (depth 5). It should overwrite the stale S1 deep (depth 20)
        // because stale entries are treated as empty.
        tt.store(s2, h2, 50, 5, TTFlag::Exact, None);
        assert_eq!(tt.table[0].slot_deep.state, s2);
        assert_eq!(tt.table[0].slot_deep.generation, 2);
        assert_eq!(tt.table[0].slot_deep.depth, 5);
    }

    #[test]
    fn test_tt_raii_pooling() {
        // Ensure pool is empty
        TT_POOL_U64.with(|p| p.borrow_mut().clear());

        {
            let tt_guard = TranspositionTable::<u64>::get_pooled(1);
            assert_eq!(
                tt_guard.capacity(),
                TranspositionTable::<u64>::new(1).capacity()
            );
            // Guard dropped here
        }

        // Verify it's in the pool
        let in_pool = TT_POOL_U64.with(|p| p.borrow().len());
        assert_eq!(in_pool, 1);

        // Borrow again, should reuse
        let _tt_guard2 = TranspositionTable::<u64>::get_pooled(1);
        let in_pool2 = TT_POOL_U64.with(|p| p.borrow().len());
        assert_eq!(in_pool2, 0);
    }

    #[test]
    fn test_tt_pooling_size_segregation() {
        TT_POOL_U64.with(|p| p.borrow_mut().clear());

        {
            let _g1 = TranspositionTable::<u64>::get_pooled(1);
            let _g2 = TranspositionTable::<u64>::get_pooled(2);
        }

        let in_pool = TT_POOL_U64.with(|p| p.borrow().len());
        assert_eq!(in_pool, 2);

        // Requesting size 1 should specifically get the size 1 TT
        let g3 = TranspositionTable::<u64>::get_pooled(1);
        assert!(g3.capacity() < TranspositionTable::<u64>::new(2).capacity());

        let remaining = TT_POOL_U64.with(|p| p.borrow().len());
        assert_eq!(remaining, 1);
    }

    #[test]
    fn test_tt_lookup_stale() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, _, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);

        tt.store(s1, h1, 100, 10, TTFlag::Exact, None);
        tt.reset();

        // Even though state matches, generation doesn't, so lookup should fail
        assert!(tt.lookup(&s1, h1).is_none());
    }

    #[test]
    fn test_tt_store_promotion_logic_detailed() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);

        tt.store(s1, h1, 100, 10, TTFlag::Exact, None); // Deep: S1(10)
        tt.store(s2, h2, 50, 5, TTFlag::Exact, None); // Recent: S2(5)

        // Update S2 to depth 15. Should be promoted to Deep, demoting S1.
        tt.store(s2, h2, 60, 15, TTFlag::Exact, None);

        assert_eq!(tt.table[0].slot_deep.state, s2);
        assert_eq!(tt.table[0].slot_deep.depth, 15);
        assert_eq!(tt.table[0].slot_recent.state, s1);
        assert_eq!(tt.table[0].slot_recent.depth, 10);
    }

    #[test]
    fn test_tt_hash_collision_integrity() {
        let mut tt = TranspositionTable::<u64>::new(0); // 1 bucket
        let (s1, s2, _) = get_states();
        let h1 = 42; // Force identical hash
        let h2 = 42;

        tt.store(s1, h1, 100, 10, TTFlag::Exact, None);

        // Lookup with s2 and same hash should fail because states differ
        assert!(tt.lookup(&s2, h2).is_none());
        // Lookup with s1 and same hash should succeed
        assert!(tt.lookup(&s1, h1).is_some());
    }

    #[test]
    fn test_tt_capacity_scaling() {
        // bucket size is approx 56-64 bytes.
        // 1MB / 64 bytes = ~16384 buckets.
        let tt1 = TranspositionTable::<u64>::new(1);
        assert!(tt1.capacity() >= 16384);
        assert!(tt1.capacity().is_power_of_two());

        let tt2 = TranspositionTable::<u64>::new(4);
        assert_eq!(tt2.capacity(), tt1.capacity() * 4);
    }

    #[test]
    fn test_tt_pool_thread_isolation() {
        TT_POOL_U64.with(|p| p.borrow_mut().clear());

        // Fill pool in main thread
        {
            let _g = TranspositionTable::<u64>::get_pooled(1);
        }
        assert_eq!(TT_POOL_U64.with(|p| p.borrow().len()), 1);

        // Spawn thread and check its pool is empty
        std::thread::spawn(|| {
            assert_eq!(TT_POOL_U64.with(|p| p.borrow().len()), 0);
            let g = TranspositionTable::<u64>::get_pooled(1);
            assert_eq!(TT_POOL_U64.with(|p| p.borrow().len()), 0);
            // Guard dropped here, pool should now have 1
            drop(g);
            assert_eq!(TT_POOL_U64.with(|p| p.borrow().len()), 1);
        })
        .join()
        .unwrap();

        // Main thread pool should still have exactly 1
        assert_eq!(TT_POOL_U64.with(|p| p.borrow().len()), 1);
    }

    #[test]
    fn test_tt_generation_0_is_always_empty() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, _, _) = get_states();
        let h1 = zobrist::compute_hash(&s1);

        // Manually inject an entry with generation 0
        tt.table[0].slot_deep = TTEntry {
            state: s1,
            generation: 0,
            ..Default::default()
        };

        // Lookup should fail even if state matches because generation is 0
        assert!(tt.lookup(&s1, h1).is_none());
    }

    #[test]
    fn test_tt_replacement_with_both_stale() {
        let mut tt = TranspositionTable::<u64>::new(0);
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        // Gen 1: Fill both slots
        tt.store(s1, h1, 100, 10, TTFlag::Exact, None);
        // S2 demotes S1
        tt.store(s2, h2, 200, 20, TTFlag::Exact, None);
        assert_eq!(tt.table[0].slot_deep.state, s2);
        assert_eq!(tt.table[0].slot_recent.state, s1);

        // Gen 2: Clear
        tt.reset();

        // Store S3. It should overwrite Deep (S2) because it is stale.
        tt.store(s3, h3, 300, 5, TTFlag::Exact, None);
        assert_eq!(tt.table[0].slot_deep.state, s3);
        assert_eq!(tt.table[0].slot_deep.generation, 2);

        // slot_recent should still be the stale S1 from Gen 1
        assert_eq!(tt.table[0].slot_recent.state, s1);
        assert_eq!(tt.table[0].slot_recent.generation, 1);

        // Store S1 again at depth 3. It should overwrite stale recent.
        tt.store(s1, h1, 50, 3, TTFlag::Exact, None);
        assert_eq!(tt.table[0].slot_recent.state, s1);
        assert_eq!(tt.table[0].slot_recent.generation, 2);
    }

    #[test]
    fn test_tt_statistics_verification() {
        let mut tt = TranspositionTable::<u64>::new(0); // 1 bucket
        let (s1, s2, s3) = get_states();
        let h1 = zobrist::compute_hash(&s1);
        let h2 = zobrist::compute_hash(&s2);
        let h3 = zobrist::compute_hash(&s3);

        // 1. Verify Lookups and Misses
        let _ = tt.lookup(&s1, h1);
        let _ = tt.lookup(&s2, h2);
        let stats = tt.stats();
        assert_eq!(stats.lookups, 2);
        assert_eq!(stats.hits_deep, 0);
        assert_eq!(stats.hits_recent, 0);

        // 2. Verify Stores
        tt.store(s1, h1, 100, 10, TTFlag::Exact, None); // Deep: S1
        tt.store(s2, h2, 200, 5, TTFlag::Exact, None); // Recent: S2
        let stats = tt.stats();
        assert_eq!(stats.stores, 2);
        assert_eq!(stats.overwrites, 0); // Both slots were empty/stale

        // 3. Verify Hits (Deep and Recent)
        let _ = tt.lookup(&s1, h1); // Hit Deep
        let _ = tt.lookup(&s2, h2); // Hit Recent
        let stats = tt.stats();
        assert_eq!(stats.lookups, 4);
        assert_eq!(stats.hits_deep, 1);
        assert_eq!(stats.hits_recent, 1);

        // 4. Verify Overwrites (Replacement)
        // Store S3 at depth 2. S3 is shallower than Deep S1(10), so it replaces Recent S2(5).
        tt.store(s3, h3, 300, 2, TTFlag::Exact, None);
        let stats = tt.stats();
        assert_eq!(stats.stores, 3);
        assert_eq!(stats.overwrites, 1);

        // 5. Verify Overwrites (Demotion)
        // Store S2 at depth 20. S2 demotes Deep S1(10) to Recent, evicting Recent S3(2).
        tt.store(s2, h2, 400, 20, TTFlag::Exact, None);
        let stats = tt.stats();
        assert_eq!(stats.stores, 4);
        assert_eq!(stats.overwrites, 2); // Demoting deep is also an overwrite of the existing recent

        // 6. Verify Reset
        tt.reset();
        let stats = tt.stats();
        assert_eq!(stats.lookups, 0);
        assert_eq!(stats.stores, 0);
        assert_eq!(stats.hits_deep, 0);
        assert_eq!(stats.hits_recent, 0);
        assert_eq!(stats.overwrites, 0);

        // Verify lookup after reset is a miss and increments lookups
        let _ = tt.lookup(&s1, h1);
        assert_eq!(tt.stats().lookups, 1);
        assert_eq!(tt.stats().hits_deep, 0);
    }
}

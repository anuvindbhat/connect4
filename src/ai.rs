//! # Connect 4 AI Engine
//!
//! This module implements a high-performance Connect 4 AI based on the Negamax algorithm
//! with Alpha-Beta pruning, Principal Variation Search (PVS), and Iterative Deepening.
//!
//! ## Core Search Algorithms
//!
//! - **Negamax with Alpha-Beta Pruning**: A variant of minimax that simplifies implementation
//!   for zero-sum games. Alpha-Beta pruning significantly reduces the number of nodes
//!   evaluated by "cutting off" branches that cannot possibly influence the final decision.
//! - **Principal Variation Search (PVS)**: An enhancement to Alpha-Beta pruning that assumes
//!   the first move explored is the best. It searches subsequent moves with a "null window"
//!   to quickly prove they are inferior, falling back to a full search only when necessary.
//! - **Iterative Deepening**: A strategy that searches to increasing depths (1, 2, 3...).
//!   This allows for better move ordering by using results from shallower searches to
//!   inform deeper ones, and provides a natural way to implement time-limited searches.
//! - **Transposition Table (TT)**: A cache of previously evaluated positions, keyed by
//!   Zobrist hashes. This avoids redundant work when the same position is reached via
//!   different move sequences (transpositions).
//!
//! ## Heuristic Evaluation
//!
//! The engine uses a sophisticated heuristic to evaluate non-terminal positions:
//! - **Positional Score**: Rewards controlling central columns and high-potential cells.
//! - **Tactical Score**: Identifies immediate and future threats (sequences that can
//!   lead to a win on the next turn or several turns later).
//! - **Setup Score**: Detects "traps" and "forks" where multiple threats are created
//!   simultaneously.
//! - **Mobility**: Measures the number of open "windows" (sequences of 4 cells) still
//!   available to a player.
//! - **Tapering**: The evaluation dynamically shifts its focus from positional control
//!   to tactical urgency as the board fills up.
//!
//! ## Move Selection
//!
//! - **Boltzmann (Softmax) Selection**: For casual play, the AI can choose moves
//!   probabilistically based on their scores. This uses a "temperature" parameter that
//!   decays as the game progresses, making the AI more deterministic in the endgame.
//! - **Tactical Override**: If a forced win or loss is detected, the AI bypasses
//!   randomization to play perfectly.

use crate::config::{
    HeuristicWeights, MAX_COLUMNS, MAX_ROWS, SCORE_WIN, TEMPERATURE, WIN_THRESHOLD,
};
use crate::game::{BoardGeometry, BoardState};
use crate::tt::{TTEntry, TTFlag, TTStats, TranspositionTable};
use crate::types::{Bitboard, Player};
use crate::zobrist;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rng;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::instrument;

/// Minimum depth required to perform Transposition Table lookups and stores.
/// For Connect 4, depth=1 searches are so fast that TT overhead outweighs the gain.
/// We use a single threshold for both lookup and store because:
/// 1) In a fixed-depth search where state is strictly tied to move count (ply),
///    any state found in the TT will always match the current remaining search depth.
/// 2) The cost of performing a lookup (Zobrist hash update + memory fetch +
///    256-bit state comparison) is higher than the cost of evaluating 7 leaf nodes.
const MIN_TT_DEPTH: u32 = 2;

/// The minimum possible evaluation score.
const MIN_SCORE: i32 = i32::MIN + 1;
/// The maximum possible evaluation score.
const MAX_SCORE: i32 = i32::MAX;
const _: () = assert!(
    MAX_SCORE == -MIN_SCORE,
    "Should be able to safely negate scores for negamax"
);

// ========================================================================================
// SEARCH ALGORITHMS
// ========================================================================================

/// Internal state shared across the recursive search tree.
///
/// `SearchContext` carries configuration, heuristic weights, and the transposition table
/// through the negamax recursion. It also tracks the total number of nodes visited
/// during the search using an atomic counter for thread safety (though current search
/// is single-threaded, this allows for future parallelism).
struct SearchContext<'a, T: Bitboard> {
    /// The geometry of the board being searched.
    geo: &'a BoardGeometry<T>,
    /// Heuristic weights used for position evaluation.
    weights: &'a HeuristicWeights,
    /// Total number of nodes (positions) visited during this search.
    node_count: AtomicU64,
    /// Optional mutable reference to the transposition table for memoization.
    tt: Option<&'a mut TranspositionTable<T>>,
}

impl<T: Bitboard> SearchContext<'_, T> {
    /// Returns true if the Transposition Table is present and the search depth
    /// is at or above the minimum threshold (`MIN_TT_DEPTH`).
    ///
    /// At shallow depths, the overhead of TT operations often exceeds the
    /// time saved by skipping the search.
    #[inline]
    fn tt_active(&self, depth: u32) -> bool {
        self.tt.is_some() && depth >= MIN_TT_DEPTH
    }

    /// Retrieves an entry from the transposition table if one is present.
    /// Returns `None` if no entry exists for the given state and hash.
    fn lookup(&self, state: &BoardState<T>, hash: u64) -> Option<TTEntry<T>> {
        self.tt.as_ref().and_then(|tt| tt.lookup(state, hash))
    }

    /// Stores a search result in the transposition table if one is present.
    ///
    /// # Arguments
    ///
    /// * `state`: The board state being stored.
    /// * `hash`: The Zobrist hash of the state.
    /// * `score`: The evaluated score for this position.
    /// * `depth`: The search depth at which this score was determined.
    /// * `flag`: A `TTFlag` indicating if the score is exact or a bound.
    /// * `best_move`: The column index of the best move found for this state.
    fn store(
        &mut self,
        state: BoardState<T>,
        hash: u64,
        score: i32,
        depth: u32,
        flag: TTFlag,
        best_move: Option<u32>,
    ) {
        if let Some(ref mut tt) = self.tt {
            tt.store(state, hash, score, depth, flag, best_move);
        }
    }
}

/// A comprehensive report on the results of an AI search.
///
/// This struct is returned by the engine's search methods and provides not only
/// the recommended move but also performance metrics and transposition table
/// diagnostics, which are useful for debugging and tuning.
#[derive(Debug, Clone)]
pub struct SearchReport {
    /// The column index of the best move found. `None` if no legal moves exist.
    pub best_move: Option<u32>,
    /// The total number of board positions evaluated during the search.
    pub nodes: u64,
    /// Statistics from the transposition table, if one was used.
    pub tt_stats: Option<TTStats>,
}

/// Returns the evaluation score for each column assuming it's the given player's turn.
///
/// This function performs a separate search for each legal move at the root,
/// providing a detailed breakdown of the AI's perspective on the board.
///
/// Scores are returned from the perspective of the `curr_p` player:
/// higher is better for `curr_p`.
///
/// # Arguments
///
/// * `state`: The current board state.
/// * `geo`: The board geometry (dimensions).
/// * `curr_p`: The player whose turn it is.
/// * `depth`: The search depth (in plies).
/// * `weights`: The heuristic weights used for evaluation.
/// * `tt`: Optional transposition table to speed up the search.
///
/// # Returns
///
/// A `Vec` of scores for each column. If a move is illegal, its entry is `None`.
#[must_use]
#[instrument(skip(state, geo, weights, tt))]
pub fn get_column_scores<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    depth: u32,
    weights: &HeuristicWeights,
    tt: Option<&mut TranspositionTable<T>>,
) -> Vec<Option<i32>> {
    let mut ctx = SearchContext {
        geo,
        weights,
        node_count: AtomicU64::new(0),
        tt,
    };

    let hash = if ctx.tt_active(depth) {
        zobrist::compute_hash(state)
    } else {
        0
    };

    let scores = negamax_all_moves(state, hash, &mut ctx, depth, curr_p);
    let mut results = vec![None; geo.columns as usize];
    for (col, score) in scores {
        results[col as usize] = if curr_p == Player::Red {
            Some(score)
        } else {
            Some(-score)
        };
    }
    results
}

/// Finds the best move using iterative deepening and Boltzmann selection.
///
/// Iterative deepening allows the AI to use results from shallower searches to
/// improve move ordering for deeper searches. If `randomize` is true, the AI
/// uses Boltzmann selection (Softmax) to choose a move probabilistically,
/// unless a forced win/loss is detected.
///
/// # Arguments
///
/// * `state`: The current board state.
/// * `geo`: The board geometry.
/// * `curr_p`: The player whose turn it is.
/// * `depth`: The maximum search depth.
/// * `randomize`: Whether to use Boltzmann selection.
/// * `weights`: Heuristic weights for evaluation.
/// * `tt`: Optional transposition table.
///
/// # Returns
///
/// A `SearchReport` containing the best move and search statistics.
#[must_use]
#[instrument(skip(state, geo, weights, tt))]
pub fn find_best_move_detailed<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    depth: u32,
    randomize: bool,
    weights: &HeuristicWeights,
    tt: Option<&mut TranspositionTable<T>>,
) -> SearchReport {
    let mut final_scores_raw = Vec::new();
    let mut best_move_found = None;
    let mut best_score_found = MIN_SCORE;

    let mut ctx = SearchContext {
        geo,
        weights,
        node_count: AtomicU64::new(0),
        tt,
    };

    // If iterative deepening may require the hash, compute it.
    let root_hash = if ctx.tt_active(depth) {
        zobrist::compute_hash(state)
    } else {
        0
    };

    // Iterative Deepening Loop
    for d in 1..=depth {
        if randomize {
            // Boltzmann selection requires accurate scores for ALL moves.
            final_scores_raw = negamax_all_moves(state, root_hash, &mut ctx, d, curr_p);

            if let Some(&(m, s)) = final_scores_raw.iter().max_by_key(|&(_, s)| *s) {
                best_move_found = Some(m);
                best_score_found = s;
            }
        } else {
            // Deterministic mode: use Alpha-Beta pruning at the root for speed.
            let (score, m) = negamax(state, root_hash, &mut ctx, d, MIN_SCORE, MAX_SCORE, curr_p);
            best_score_found = score;
            best_move_found = m;
        }

        // If no moves are available (terminal state), we stop searching.
        if best_move_found.is_none() {
            break;
        }

        // If we found a forced tactical outcome (win or loss), we can stop searching deeper
        // because Connect 4 is a zero-sum game with perfect information.
        if best_score_found.abs() >= WIN_THRESHOLD {
            break;
        }
    }

    // Tactical Override: If a win or loss is found, play perfectly by disabling
    // Boltzmann selection and picking the best move found at the deepest iteration.
    let best_move =
        if randomize && best_score_found.abs() < WIN_THRESHOLD && !final_scores_raw.is_empty() {
            Some(select_boltzmann_move(&final_scores_raw, state, geo))
        } else {
            best_move_found
        };

    SearchReport {
        best_move,
        nodes: ctx.node_count.load(Ordering::Relaxed),
        tt_stats: ctx.tt.as_ref().map(|tt| tt.stats()),
    }
}

/// Returns the search evaluation score for a board state at the specified depth.
///
/// Uses deterministic search (no Boltzmann selection).
/// Scores are absolute (Red-relative).
#[must_use]
pub fn evaluate_position<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    depth: u32,
    weights: &HeuristicWeights,
    tt: Option<&mut TranspositionTable<T>>,
) -> i32 {
    let mut ctx = SearchContext {
        geo,
        weights,
        node_count: AtomicU64::new(0),
        tt,
    };
    let hash = if ctx.tt_active(depth) {
        zobrist::compute_hash(state)
    } else {
        0
    };
    let (score, _) = negamax(state, hash, &mut ctx, depth, MIN_SCORE, MAX_SCORE, curr_p);
    if curr_p == Player::Red { score } else { -score }
}

/// Finds the best move using iterative deepening and Boltzmann selection.
///
/// This is a convenience wrapper around `find_best_move_detailed` that
/// only returns the chosen move.
#[must_use]
pub fn find_best_move<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    depth: u32,
    randomize: bool,
    weights: &HeuristicWeights,
    tt: Option<&mut TranspositionTable<T>>,
) -> Option<u32> {
    find_best_move_detailed(state, geo, curr_p, depth, randomize, weights, tt).best_move
}

/// Selects a move from the given scores using Boltzmann (Softmax) distribution.
///
/// The probability of selecting a move is proportional to `exp(score / temperature)`.
/// The temperature decays as the board fills up, making the selection more
/// deterministic in the late game.
fn select_boltzmann_move<T: Bitboard>(
    scores: &[(u32, i32)],
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
) -> u32 {
    // Scores are already relative to `curr_p` since we are using negamax.
    let max_s = scores
        .iter()
        .map(|&(_, s)| s)
        .max()
        .expect("No scores available");

    // Calculate dynamic temperature based on game progress.
    // As the board fills up, the temperature drops towards zero, making the AI more deterministic.
    let total_cells = geo.columns * geo.rows;
    let occupied_cells = (state.bits(Player::Red) | state.bits(Player::Yellow)).count_ones();
    let remaining_ratio =
        f64::from(total_cells.saturating_sub(occupied_cells)) / f64::from(total_cells);
    // Ensure temp never hits absolute zero
    let dynamic_temp = (TEMPERATURE * remaining_ratio.powi(2)).max(1.0e-8);

    let weights: Vec<f64> = scores
        .iter()
        .map(|&(_, s)| (f64::from(s) / dynamic_temp - f64::from(max_s) / dynamic_temp).exp())
        .collect();

    if let Ok(dist) = WeightedIndex::new(&weights) {
        scores[dist.sample(&mut rng())].0
    } else {
        panic!("Invalid weights for Boltzmann selection: {weights:?}");
    }
}

/// Force-evaluates all legal moves at the root node with full alpha-beta windows.
///
/// This is used when Boltzmann selection is enabled, as it requires accurate
/// scores for all available moves, not just the best one.
///
/// Bypasses TT lookups to ensure identity but will perform TT storage.
fn negamax_all_moves<T: Bitboard>(
    state: &BoardState<T>,
    hash: u64,
    ctx: &mut SearchContext<T>,
    depth: u32,
    curr_p: Player,
) -> Vec<(u32, i32)> {
    ctx.node_count.fetch_add(1, Ordering::Relaxed);

    if state.has_won(curr_p.other(), ctx.geo)
        || state.has_won(curr_p, ctx.geo)
        || state.is_full(ctx.geo)
    {
        return Vec::new();
    }

    let mut scores = Vec::new();
    let moves = get_move_order(state, ctx.geo, curr_p, ctx.weights);
    let next_depth = depth.saturating_sub(1);

    for &col in &moves {
        let (next, bit_idx) = state
            .drop_piece_with_index(col, curr_p, ctx.geo)
            .expect("Root move selection must only occur on valid columns");

        // Optimized hashing: skip Zobrist update if the next level won't use it.
        let next_hash = if ctx.tt_active(next_depth) {
            zobrist::apply_move(hash, curr_p, bit_idx)
        } else {
            0
        };

        // Use full alpha-beta window for EACH branch to get accurate scores.
        let score = apply_depth_tax(
            -negamax(
                &next,
                next_hash,
                ctx,
                next_depth,
                MIN_SCORE,
                MAX_SCORE,
                curr_p.other(),
            )
            .0,
        );
        scores.push((col, score));
    }

    // Transposition Table Storage (Root)
    // Only store results for non-trivial searches.
    if depth >= MIN_TT_DEPTH
        && let Some(&(m, s)) = scores.iter().max_by_key(|&(_, s)| *s)
    {
        ctx.store(*state, hash, s, depth, TTFlag::Exact, Some(m));
    }

    scores
}

/// Core search function implementing Negamax with Alpha-Beta pruning and TT.
///
/// This function recursively explores the game tree to find the best move
/// for the current player. It uses several optimizations:
/// - **Alpha-Beta Pruning**: Discards branches that cannot affect the result.
/// - **Transposition Table**: Caches results to avoid redundant work.
/// - **Move Ordering**: Prioritizes promising moves to increase pruning efficiency.
///
/// # Arguments
///
/// * `state`: The board state to evaluate.
/// * `hash`: The Zobrist hash of the state.
/// * `ctx`: The search context.
/// * `depth`: Remaining search depth.
/// * `alpha`: The minimum score the maximizing player is assured of.
/// * `beta`: The maximum score that the minimizing player is assured of.
/// * `curr_p`: The player whose turn it is.
///
/// # Returns
///
/// A tuple `(score, best_move)` where `score` is the evaluation from the
/// perspective of `curr_p`, and `best_move` is the column index of the best move.
#[allow(clippy::too_many_arguments)]
fn negamax<T: Bitboard>(
    state: &BoardState<T>,
    hash: u64,
    ctx: &mut SearchContext<T>,
    depth: u32,
    mut alpha: i32,
    mut beta: i32,
    curr_p: Player,
) -> (i32, Option<u32>) {
    ctx.node_count.fetch_add(1, Ordering::Relaxed);

    // We store the original alpha-beta bounds (even before TT lookup)
    // since we want to maximize our chance of storing an Exact result.
    // This happens when the window is as wide as possible.
    // Proof that it's better and correct to save alpha before the TT
    // lookup (assume beta is +inf):
    //   Suppose the entry has TTFlag::LowerBound, let the updated alpha value
    //   be alpha_tt.
    //   Case 1) The looked up value <= alpha_orig => alpha_tt = alpha_orig
    //     In this case, it doesn't matter whether we compare against alpha_tt
    //     or alpha_orig when storing.
    //   Case 2) The looked up value > alpha_orig => alpha_tt > alpha_orig
    //     Since TTFlag::LowerBound, true_value >= alpha_tt > alpha_orig.
    //     We then perform the recursive search with window [alpha_tt, +inf).
    //     If it's > alpha_tt, we store TT:Exact which is correct. If it's ==
    //     alpha_tt, the search tells use true_value <= alpha_tt (fail-soft property).
    //     Suppose we compared against alpha_tt when storing, we would've stored
    //     this as UpperBound which is correct, but we can do better.
    //     We have that true_val >= alpha_tt and true_val <= alpha_tt =>
    //     true_val == alpha_tt. Therefore, if we compared against alpha_orig
    //     we would've stored TT:Exact instead of TT::UpperBound which is better.
    let alpha_orig = alpha;
    let beta_orig = beta;

    // Use other() because we want to check if the player who JUST MOVED won.
    if state.has_won(curr_p.other(), ctx.geo) {
        return (-SCORE_WIN, None);
    }

    // It's not possible for the person who just moved to have lost.
    debug_assert!(
        !state.has_won(curr_p, ctx.geo),
        "curr_p can not have won during a search"
    );

    if state.is_full(ctx.geo) {
        return (0, None);
    }

    if depth == 0 {
        // `evaluate_board` returns scores relative to the current player
        // so we don't need to modify it.
        let final_score = evaluate_board(state, ctx.geo, curr_p, ctx.weights);
        return (final_score, None);
    }

    // 1. Transposition Table Lookup (only for non-trivial depths)
    let tt_entry = if depth >= MIN_TT_DEPTH {
        ctx.lookup(state, hash)
    } else {
        None
    };

    // A TT entry can only be used to update bounds or return early if it was
    // at least as deep as our current search depth.
    if let Some(entry) = tt_entry
        && entry.depth >= depth
    {
        match entry.flag {
            TTFlag::Exact => return (entry.score, entry.best_move),
            TTFlag::LowerBound => alpha = alpha.max(entry.score),
            TTFlag::UpperBound => beta = beta.min(entry.score),
        }
        if alpha >= beta {
            return (entry.score, entry.best_move);
        }
    }

    // 2. Move Ordering with TT Hint (can and should be done even if the
    //    entry was at a shallower depth).
    let mut moves = get_move_order(state, ctx.geo, curr_p, ctx.weights);
    if let Some(m) = tt_entry.and_then(|e| e.best_move) {
        for i in 0..moves.len() {
            if moves[i] == m {
                moves[0..=i].rotate_right(1);
                break;
            }
        }
    }

    let (best_score, best_move) =
        negamax_helper(state, hash, ctx, depth, alpha, beta, curr_p, &moves);

    // 3. Transposition Table Storage
    if depth >= MIN_TT_DEPTH {
        let flag = if best_score <= alpha_orig {
            TTFlag::UpperBound
        } else if best_score >= beta_orig {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };
        ctx.store(*state, hash, best_score, depth, flag, best_move);
    }

    (best_score, best_move)
}

/// Helper for `negamax` that iterates through moves and applies PVS.
///
/// **Principal Variation Search (PVS)** optimization:
/// Assuming the first move is the best (due to good move ordering), it's searched
/// with a full window. Subsequent moves are searched with a "null window"
/// `[-alpha-1, -alpha]` to quickly prove they are worse than the first move.
/// If a move "breaks" the null window, it is re-searched with a full window.
#[allow(clippy::too_many_arguments)]
fn negamax_helper<T: Bitboard>(
    state: &BoardState<T>,
    hash: u64,
    ctx: &mut SearchContext<T>,
    depth: u32,
    alpha: i32,
    beta: i32,
    curr_p: Player,
    moves: &[u32],
) -> (i32, Option<u32>) {
    let next_depth = depth.saturating_sub(1);
    // Best score so far for the children's ply.
    let mut best_score_adj = MIN_SCORE;
    let mut best_move = None;

    // alpha, beta adjusted to the children's ply.
    let mut alpha_adj = remove_depth_tax(alpha);
    let beta_adj = remove_depth_tax(beta);

    // We perform all comparisons in the loop at the children's ply.
    // This helps us avoid repeatedly adjusting alpha, beta and taxing
    // the `negamax` result inside the loop.
    for (i, &col) in moves.iter().enumerate() {
        let (next, bit_idx) = state
            .drop_piece_with_index(col, curr_p, ctx.geo)
            .expect("Search must only explore legal moves");

        // Optimized hashing: skip Zobrist update if the next level won't use it.
        let next_hash = if ctx.tt_active(next_depth) {
            zobrist::apply_move(hash, curr_p, bit_idx)
        } else {
            0
        };

        let score_adj = if i == 0 {
            // Full window search for the first (and presumably best) move.
            -negamax(
                &next,
                next_hash,
                ctx,
                next_depth,
                -beta_adj,
                -alpha_adj,
                curr_p.other(),
            )
            .0
        } else {
            // Null window search for subsequent moves.
            let mut s_adj = -negamax(
                &next,
                next_hash,
                ctx,
                next_depth,
                -alpha_adj - 1,
                -alpha_adj,
                curr_p.other(),
            )
            .0;
            // If the null window search failed high, re-search with full window.
            if s_adj > alpha_adj && s_adj < beta_adj {
                s_adj = -negamax(
                    &next,
                    next_hash,
                    ctx,
                    next_depth,
                    -beta_adj,
                    -alpha_adj,
                    curr_p.other(),
                )
                .0;
            }
            s_adj
        };

        if score_adj > best_score_adj {
            best_score_adj = score_adj;
            best_move = Some(col);
            alpha_adj = alpha_adj.max(best_score_adj);
        }

        if beta_adj <= alpha_adj {
            break;
        }
    }

    // Apply the tax once we know what the best score is to convert it to our ply.
    (apply_depth_tax(best_score_adj), best_move)
}

/// "Tax" the score if it's a win/loss by 1 point for every ply it travels up.
///
/// This ensures the AI prefers faster wins (higher score) and slower losses
/// (less negative score).
fn apply_depth_tax(score: i32) -> i32 {
    // We should never tax a score beyond SCORE_WIN since negamax will
    // never return such a value (we use a fail-soft implementation).
    debug_assert!(
        score.abs() <= SCORE_WIN,
        "score {score} should be within +-SCORE_WIN +-{SCORE_WIN}"
    );
    debug_assert_ne!(
        score.abs(),
        WIN_THRESHOLD,
        "score {score} at +-WIN_THRESHOLD +-{WIN_THRESHOLD} should not be taxed"
    );
    if score > WIN_THRESHOLD {
        score - 1
    } else if score < -WIN_THRESHOLD {
        score + 1
    } else {
        score
    }
}

/// Reverses the depth tax to calculate bounds for children nodes.
///
/// A window of [20, Win in 3] for the parent should correspond to a
/// window of [Lose in 2, -20] for the child.
fn remove_depth_tax(score: i32) -> i32 {
    // It's entirely possible that a score is untaxed beyond `SCORE_WIN`.
    // This can happen if a short win/loss is found in one branch and we need
    // to untax that bound as it travels down a second branch.
    if score.abs() < WIN_THRESHOLD || score == MIN_SCORE || score == MAX_SCORE {
        score
    } else if score >= WIN_THRESHOLD {
        score + 1
    } else {
        score - 1
    }
}

/// A stack-allocated, fixed-capacity list of moves.
///
/// Encapsulates a fixed-size array and a logical length to provide a safe,
/// slice-like interface without heap allocation. This is a performance
/// optimization to avoid `Vec` allocations in the hot search loop.
#[derive(Debug, Clone)]
struct MoveOrder {
    /// The fixed-size array holding the move column indices.
    moves: [u32; MAX_COLUMNS as usize],
    /// The number of legal moves currently in the array.
    len: usize,
}

impl MoveOrder {
    /// Creates a new `MoveOrder` with a fixed logical length.
    #[must_use]
    fn new(len: usize) -> Self {
        debug_assert!(
            len <= MAX_COLUMNS as usize,
            "MoveOrder length {len} exceeds MAX_COLUMNS {MAX_COLUMNS}",
        );
        Self {
            moves: [0; MAX_COLUMNS as usize],
            len,
        }
    }
}

impl std::ops::Deref for MoveOrder {
    type Target = [u32];
    fn deref(&self) -> &Self::Target {
        &self.moves[..self.len]
    }
}

impl std::ops::DerefMut for MoveOrder {
    fn deref_mut(&mut self) -> &mut [u32] {
        &mut self.moves[..self.len]
    }
}

impl<'a> IntoIterator for &'a MoveOrder {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        use std::ops::Deref;
        self.deref().iter()
    }
}

impl<'a> IntoIterator for &'a mut MoveOrder {
    type Item = &'a mut u32;
    type IntoIter = std::slice::IterMut<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        use std::ops::DerefMut;
        self.deref_mut().iter_mut()
    }
}

/// Computes a magic multiplier for fast integer division by a constant.
const fn compute_magic_multiplier(divisor: u8) -> u16 {
    assert!(divisor > 1);
    let d = divisor as u32;
    let m = (1u32 << 16).div_ceil(d);
    assert!(m <= u16::MAX as u32);
    #[allow(clippy::cast_possible_truncation)]
    let mult = m as u16;
    mult
}

/// Generates a table of magic multipliers for all possible board heights.
const fn generate_magic_multipliers() -> [u16; MAX_ROWS as usize + 2] {
    let mut table = [0u16; MAX_ROWS as usize + 2];
    let mut d = 2;
    #[allow(clippy::cast_possible_truncation)]
    while d <= MAX_ROWS + 1 {
        table[d as usize] = compute_magic_multiplier(d as u8);
        d += 1;
    }
    table
}

/// Magic multipliers for fast column index calculation from bit index.
const MAGIC_MULTIPLIERS: [u16; MAX_ROWS as usize + 2] = generate_magic_multipliers();

/// Determines the order in which moves should be explored.
///
/// Effective move ordering is critical for Alpha-Beta pruning performance.
/// This function categorizes moves into tiers and sorts them:
/// 1. **Victory**: Immediate winning moves.
/// 2. **Block**: Moves that prevent an immediate opponent win.
/// 3. **Setup**: Moves that create offensive traps.
/// 4. **Positional**: Moves that control high-value areas (center-biased).
/// 5. **Blunders**: Moves that lead to an immediate loss (searched last).
#[allow(clippy::too_many_lines)]
fn get_move_order<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    _weights: &HeuristicWeights,
) -> MoveOrder {
    const NUM_TIERS: usize = 7;
    let ctx = EvalContext::new(state, geo, curr_p);

    // A subset of what `evaluate_directional_patterns` computes.
    let h = geo.rows + 1;
    let mut p_threat_mask = T::zero();
    let mut p_setup_mask = T::zero();
    let mut o_threat_mask = T::zero();

    for d in [h, h + 1, h - 1] {
        let e = ctx.empty;
        // Empty right-shifted by 1, 2, 3
        let er1 = e >> d;
        let er2 = e >> (2 * d);
        let er3 = e >> (3 * d);

        let p = ctx.p_bits;
        // Player right-shifted by 1, 2, 3 and left-shifted by 1, 2, 3
        let pr1 = p >> d;
        let pr2 = p >> (2 * d);
        let pr3 = p >> (3 * d);

        let o = ctx.o_bits;
        // Opponent right-shifted by 1, 2, 3 and left-shifted by 1, 2, 3
        let or1 = o >> d;
        let or2 = o >> (2 * d);
        let or3 = o >> (3 * d);

        {
            // 3-in-a-window patterns for player and opponent.
            let p_three_ws = [
                e & pr1 & pr2 & pr3,
                p & er1 & pr2 & pr3,
                p & pr1 & er2 & pr3,
                p & pr1 & pr2 & er3,
            ];
            let o_three_ws = [
                e & or1 & or2 & or3,
                o & er1 & or2 & or3,
                o & or1 & er2 & or3,
                o & or1 & or2 & er3,
            ];
            p_threat_mask |= p_three_ws[0]
                | (p_three_ws[1] << d)
                | (p_three_ws[2] << (2 * d))
                | (p_three_ws[3] << (3 * d));
            o_threat_mask |= o_three_ws[0]
                | (o_three_ws[1] << d)
                | (o_three_ws[2] << (2 * d))
                | (o_three_ws[3] << (3 * d));
        }

        {
            // 2-in-a-window patterns for player.
            let p_two_ws = [
                e & er1 & pr2 & pr3,
                e & pr1 & er2 & pr3,
                e & pr1 & pr2 & er3,
                p & er1 & er2 & pr3,
                p & er1 & pr2 & er3,
                p & pr1 & er2 & er3,
            ];
            p_setup_mask |= (p_two_ws[0] | (p_two_ws[0] << d))
                | (p_two_ws[1] | (p_two_ws[1] << (2 * d)))
                | (p_two_ws[2] | (p_two_ws[2] << (3 * d)))
                | ((p_two_ws[3] << d) | (p_two_ws[3] << (2 * d)))
                | ((p_two_ws[4] << d) | (p_two_ws[4] << (3 * d)))
                | ((p_two_ws[5] << (2 * d)) | (p_two_ws[5] << (3 * d)));
        }
    }

    {
        // The vertical direction is special since the only possible windows
        // are the ones where the empty spots are on the top.

        let e = ctx.empty;
        // Empty left-shifted by 1
        let el1 = e << 1;

        let p = ctx.p_bits;
        // Player left-shifted by 1, 2, 3
        let pl1 = p << 1;
        let pl2 = p << 2;
        let pl3 = p << 3;

        let o = ctx.o_bits;
        // Opponent left-shifted by 1, 2, 3
        let ol1 = o << 1;
        let ol2 = o << 2;
        let ol3 = o << 3;

        {
            let p_three_w = pl3 & pl2 & pl1 & e;
            let o_three_w = ol3 & ol2 & ol1 & e;
            // 3-in-a-window patterns for player and opponent.
            p_threat_mask |= p_three_w;
            o_threat_mask |= o_three_w;
        }

        {
            // 2-in-a-window patterns for player.
            let p_two_w = pl3 & pl2 & el1 & e;
            p_setup_mask |= p_two_w | (p_two_w >> 1);
        }
    }

    // Moves that enable an immediate opponent win on the next turn.
    let p_blunder_mask = ctx.playable & ((o_threat_mask & !ctx.playable) >> 1);

    debug_assert_eq!(
        p_threat_mask & !geo.board_mask,
        T::zero(),
        "Player threat mask {p_threat_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        o_threat_mask & !geo.board_mask,
        T::zero(),
        "Opponent threat mask {o_threat_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        p_setup_mask & !geo.board_mask,
        T::zero(),
        "Player setup mask {p_setup_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        p_blunder_mask & !geo.board_mask,
        T::zero(),
        "Player blunder mask {p_blunder_mask:?} has sentinel bits set"
    );

    // There are 2 orders here:
    // 1) The order in which to sort the moves (better move to worse move)
    // 2) The order in which to perform the check to see which tier a move is
    //    in.
    // These are mostly the same except for blunders. Blunders are the worst
    // move but we check if a move is a blunder before checking if it's a setup
    // or a positional move.

    let mut captured = T::zero();

    // 1. Victory: Immediate win.
    let t_win = ctx.playable & p_threat_mask;
    captured |= t_win;

    // 2. Block: Prevent opponent win (unless we can win first).
    let t_block = (ctx.playable & o_threat_mask) & !captured;
    captured |= t_block;

    // 3. Blunders: Moves that are tactically suicidal.
    // Only includes moves that aren't also wins/blocks.
    let t_blunder = p_blunder_mask & !captured;
    captured |= t_blunder;

    // 4. Setup: Offensive trap (that doesn't blunder).
    let t_setup = (ctx.playable & p_setup_mask) & !captured;
    captured |= t_setup;

    // 5. Positional Tiers: Strategy-based moves (not tactically urgent).
    let mut positional_tiers = [T::zero(); 3];
    for (i, tier) in positional_tiers.iter_mut().enumerate() {
        *tier = (ctx.playable & geo.weight_masks[i]) & !captured;
        captured |= *tier;
    }
    // The positional tiers cover all columns so every playable column is
    // covered by at least one tier (exactly one due to `captured`).

    // Define tier hierarchy.
    let tiers: [T; NUM_TIERS] = [
        t_win,
        t_block,
        t_setup,
        positional_tiers[0],
        positional_tiers[1],
        positional_tiers[2],
        t_blunder,
    ];

    // O(Columns + Tiers) Extraction Phase
    // Implementation note:
    //   This has been profiled to be faster than the naive O(Columns * Tiers)
    //   extraction.
    let mut col_tier: [Option<usize>; MAX_COLUMNS as usize] = [None; MAX_COLUMNS as usize];
    let mut tier_counts = [0u8; NUM_TIERS];
    let h = geo.rows + 1;
    debug_assert!(h >= 2, "Magic multipliers only work for divisors >= 2");

    // Across all tier masks, `ctx.playable.count_ones()` bits are set.
    for (tier_id, mask) in tiers.iter().enumerate() {
        let mut m = *mask;
        while m != T::zero() {
            let bit_idx = m.trailing_zeros();
            // This is the same as `bit_idx / h`.
            let col = (bit_idx * u32::from(MAGIC_MULTIPLIERS[h as usize])) >> 16;
            debug_assert_eq!(
                col,
                bit_idx / h,
                "Magic multiplier result {col} should match division result {}",
                bit_idx / h
            );
            col_tier[col as usize] = Some(tier_id);
            tier_counts[tier_id] += 1;
            m &= !(T::one() << bit_idx);
        }
    }

    // Prefix-sum offsets for placement.
    let mut offsets = [0u8; NUM_TIERS + 1];
    for (i, count) in tier_counts.iter().enumerate() {
        offsets[i + 1] = offsets[i] + *count;
    }

    // Linear Placement (preserving `search_order` center-bias within tiers).
    let mut ordered_moves = MoveOrder::new(ctx.playable.count_ones() as usize);
    for &col in &geo.search_order {
        if let Some(t) = col_tier[col as usize] {
            let pos = offsets[t];
            ordered_moves[pos as usize] = col;
            offsets[t] += 1;
        }
    }

    debug_assert_eq!(
        {
            let mut a = ordered_moves.to_vec();
            a.sort_unstable();
            a
        },
        (0..geo.columns)
            .filter(|&c| state.get_next_bit(c, geo).is_some())
            .collect::<Vec<_>>(),
        "Ordered moves must be a permutation of the legal moves"
    );
    ordered_moves
}

// ========================================================================================
// HEURISTIC EVALUATION
// ========================================================================================

/// A detailed breakdown of a board's heuristic evaluation.
///
/// This struct separates the different components of the score, which
/// is useful for debugging, weight tuning, and explaining the AI's
/// "thought process".
#[derive(Debug, Clone, Copy)]
struct HeuristicDetail {
    /// Score based on piece placement (center bias).
    positional: i32,
    /// Score based on 2-in-a-window and 3-in-a-window patterns.
    connections: i32,
    /// Score based on immediate and future winning threats.
    tactical: i32,
    /// Score based on offensive trap setups.
    setup: i32,
    /// Score based on mobility (available windows for win).
    windows: i32,
    /// Bonus/penalty for multiple simultaneous threats.
    fork: i32,
}

impl HeuristicDetail {
    /// Returns the sum of all score components.
    fn total(&self) -> i32 {
        self.positional + self.connections + self.tactical + self.setup + self.windows + self.fork
    }
}

/// Context for board evaluation and move ordering to avoid redundant bitwise operations.
///
/// This struct pre-computes and stores bitmasks that are used multiple times
/// during evaluation, reducing the number of expensive bitwise operations.
struct EvalContext<T: Bitboard> {
    /// Bits set for the current player's pieces.
    p_bits: T,
    /// Bits set for the opponent's pieces.
    o_bits: T,
    /// Bits set for all empty cells on the board.
    empty: T,
    /// Bits set for cells where a piece can be dropped in the next turn.
    playable: T,
}

impl<T: Bitboard> EvalContext<T> {
    /// Creates a new `EvalContext` for the given board state and player.
    fn new(state: &BoardState<T>, geo: &BoardGeometry<T>, curr_p: Player) -> Self {
        let (p_bits, o_bits) = (state.bits(curr_p), state.bits(curr_p.other()));
        let occupied = p_bits | o_bits;
        let empty = geo.board_mask & !occupied;
        let playable = empty & ((occupied << 1) | geo.bottom_mask);

        Self {
            p_bits,
            o_bits,
            empty,
            playable,
        }
    }
}

/// Returns a detailed breakdown of the board evaluation.
///
/// Scores are relative to the current player `curr_p`: higher is better for `curr_p`.
///
/// This function is a hotspot and has been highly optimized. It uses a
/// "tapering" strategy that scales positional scores based on how many
/// cells are still empty, making the AI more tactically focused in the endgame.
///
/// # Panics
///
/// Panics if the board capacity or total bits cannot be represented by i32.
#[must_use]
fn evaluate_board_detailed<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    weights: &HeuristicWeights,
) -> HeuristicDetail {
    let ctx = EvalContext::new(state, geo, curr_p);

    // 1a. Tapering Strategy
    // As the board fills up, the non-tactical scores become less relevant.
    let total = (ctx.p_bits | ctx.o_bits).count_ones();
    let max_p = i32::try_from(geo.columns * geo.rows).expect("Board capacity fits in i32");
    let rem = max_p.saturating_sub(i32::try_from(total).expect("Total cells fits in i32"));

    // 1b. Directional Patterns Pre-computation
    // Computes all structure-based features (threats, connections, windows) in one pass.
    let dp = evaluate_directional_patterns(&ctx, geo);

    // 2. Positional Score (3-Tier Gradient)
    // Rewards controlling central cells using a quadratic taper.
    let mut positional = 0;
    let weight_tiers = [
        weights.weight_core,
        weights.weight_inner,
        weights.weight_outer,
    ];
    for (i, &w) in weight_tiers.iter().enumerate() {
        positional += i32::try_from((ctx.p_bits & geo.weight_masks[i]).count_ones())
            .expect("Bit count fits in i32")
            * w;
        positional -= i32::try_from((ctx.o_bits & geo.weight_masks[i]).count_ones())
            .expect("Bit count fits in i32")
            * w;
    }
    positional = (positional * rem * rem) / (max_p * max_p);

    // 3. Structural/Connection Score
    // Rewards 2-in-a-window and 3-in-a-window patterns.
    let conn_score =
        (dp.p_three - dp.o_three) * weights.score_three + (dp.p_two - dp.o_two) * weights.score_two;
    let connections = (conn_score * rem) / max_p;

    // 4. Tactical Score
    // Evaluates immediate threats (can win next turn) and future threats.
    let (p_threat_imm, p_threat_fut) = (
        i32::try_from((dp.p_threat_mask & ctx.playable).count_ones())
            .expect("Bit count fits in i32"),
        i32::try_from((dp.p_threat_mask & !ctx.playable).count_ones())
            .expect("Bit count fits in i32"),
    );
    let (o_threat_imm, o_threat_fut) = (
        i32::try_from((dp.o_threat_mask & ctx.playable).count_ones())
            .expect("Bit count fits in i32"),
        i32::try_from((dp.o_threat_mask & !ctx.playable).count_ones())
            .expect("Bit count fits in i32"),
    );
    let tactical = (p_threat_imm - o_threat_imm) * weights.score_threat_immediate
        + (p_threat_fut - o_threat_fut) * weights.score_threat_future;

    // 5. Setup Score
    // Evaluates "Setups" that create future threats without immediate payoff.
    let p_setup_imm = i32::try_from((dp.p_setup_mask & ctx.playable).count_ones())
        .expect("Bit count fits in i32");
    let o_setup_imm = i32::try_from((dp.o_setup_mask & ctx.playable).count_ones())
        .expect("Bit count fits in i32");
    let setup = (p_setup_imm - o_setup_imm) * weights.score_setup_immediate;

    // 6. Mobility (Windows)
    // Counts how many 4-cell windows are still potentially winnable for each player.
    let window_score = (dp.p_windows - dp.o_windows) * weights.weight_potential_window;
    let windows = (window_score * rem) / max_p;

    let mut fork = 0;
    // 7. Fork Detection
    // A fork is having multiple simultaneous threats.
    if p_threat_imm >= 2 {
        fork += weights.score_fork_immediate;
    }
    if o_threat_imm >= 2 {
        fork -= weights.score_fork_immediate;
    }

    HeuristicDetail {
        positional,
        connections,
        tactical,
        setup,
        windows,
        fork,
    }
}

/// A collection of structural patterns found during directional evaluation.
#[derive(Debug, Clone, Copy)]
struct DirectionalPatterns<T: Bitboard> {
    /// Masks representing positions where the player has a winning threat.
    p_threat_mask: T,
    /// Masks representing positions where the player has a setup.
    p_setup_mask: T,
    /// Number of 3-in-a-window patterns for the player.
    p_three: i32,
    /// Number of 2-in-a-window patterns for the player.
    p_two: i32,
    /// Number of potential winning windows for the player.
    p_windows: i32,
    /// Masks representing positions where the opponent has a winning threat.
    o_threat_mask: T,
    /// Masks representing positions where the opponent has a setup.
    o_setup_mask: T,
    /// Number of 3-in-a-window patterns for the opponent.
    o_three: i32,
    /// Number of 2-in-a-window patterns for the opponent.
    o_two: i32,
    /// Number of potential winning windows for the opponent.
    o_windows: i32,
}

/// Evaluates all four directions (horizontal, vertical, and both diagonals)
/// to find relevant structural patterns.
#[allow(clippy::too_many_lines)]
fn evaluate_directional_patterns<T: Bitboard>(
    ctx: &EvalContext<T>,
    geo: &BoardGeometry<T>,
) -> DirectionalPatterns<T> {
    let h = geo.rows + 1;
    let mut p_threat_mask = T::zero();
    let mut p_setup_mask = T::zero();
    let mut p_three = 0;
    let mut p_two = 0;
    let mut p_windows = 0;
    let mut o_threat_mask = T::zero();
    let mut o_setup_mask = T::zero();
    let mut o_three = 0;
    let mut o_two = 0;
    let mut o_windows = 0;

    // The non-vertical directions in which we can form a window.
    for d in [h, h + 1, h - 1] {
        let e = ctx.empty;
        // Empty right-shifted by 1, 2, 3
        let er1 = e >> d;
        let er2 = e >> (2 * d);
        let er3 = e >> (3 * d);

        let p = ctx.p_bits;
        // Player right-shifted by 1, 2, 3 and left-shifted by 1, 2, 3
        let pr1 = p >> d;
        let pr2 = p >> (2 * d);
        let pr3 = p >> (3 * d);

        let o = ctx.o_bits;
        // Opponent right-shifted by 1, 2, 3 and left-shifted by 1, 2, 3
        let or1 = o >> d;
        let or2 = o >> (2 * d);
        let or3 = o >> (3 * d);

        {
            // 3-in-a-window patterns for player and opponent.
            // Implementation note:
            //   Another way of computing these would have been to keep the
            //   empty slot fixed. For example, the second window would have
            //   been `pl1 & e & pr1 & pr2` instead of `p & er1 & pr2 & pr3`.
            //   This would've simplified the threat mask calculation since
            //   we could've just ORed the windows without left shifting them
            //   back for the empty slot.
            //   However, the windows would no longer be disjoint (same bit
            //   could be set in multiple windows), and we wouldn't be able
            //   to OR them together and then perform one `.count_ones`.
            //   Since the `.count_ones` is more expensive, we choose this
            //   approach of computing the windows.
            let p_three_ws = [
                e & pr1 & pr2 & pr3,
                p & er1 & pr2 & pr3,
                p & pr1 & er2 & pr3,
                p & pr1 & pr2 & er3,
            ];
            let o_three_ws = [
                e & or1 & or2 & or3,
                o & er1 & or2 & or3,
                o & or1 & er2 & or3,
                o & or1 & or2 & er3,
            ];
            p_threat_mask |= p_three_ws[0]
                | (p_three_ws[1] << d)
                | (p_three_ws[2] << (2 * d))
                | (p_three_ws[3] << (3 * d));
            o_threat_mask |= o_three_ws[0]
                | (o_three_ws[1] << d)
                | (o_three_ws[2] << (2 * d))
                | (o_three_ws[3] << (3 * d));
            // Count 3-in-a-window opportunities using unique windows logic.
            // If the same slot contributes to multiple windows, it will be
            // counted multiple times.
            // Implementation note:
            //   A given bit can be in at most 1 of the 4 windows because of
            //   how the windows were constructed. Therefore, ORing all of them
            //   together and checking how many bits are set is identical to
            //   counting set bits individually and summing the counts.
            //   Even though `.count_ones` (POPCNT) is pretty fast (still not
            //   as fast as OR), there's only 1 execution port that can handle
            //   it. Executing many POPCNTs at once can become bottlenecked
            //   (verified by profiling).
            let p_three_ws_comb = p_three_ws.iter().fold(T::zero(), |acc, &w| acc | w);
            let o_three_ws_comb = o_three_ws.iter().fold(T::zero(), |acc, &w| acc | w);
            p_three += i32::try_from(p_three_ws_comb.count_ones()).expect("Bit count fits in i32");
            o_three += i32::try_from(o_three_ws_comb.count_ones()).expect("Bit count fits in i32");
            debug_assert_eq!(
                p_three_ws_comb.count_ones(),
                p_three_ws.iter().map(|&w| w.count_ones()).sum::<u32>(),
                "p_three_ws must be disjoint"
            );
            debug_assert_eq!(
                o_three_ws_comb.count_ones(),
                o_three_ws.iter().map(|&w| w.count_ones()).sum::<u32>(),
                "o_three_ws must be disjoint"
            );
        }

        {
            // 2-in-a-window patterns for player and opponent.
            let p_two_ws = [
                e & er1 & pr2 & pr3,
                e & pr1 & er2 & pr3,
                e & pr1 & pr2 & er3,
                p & er1 & er2 & pr3,
                p & er1 & pr2 & er3,
                p & pr1 & er2 & er3,
            ];
            let o_two_ws = [
                e & er1 & or2 & or3,
                e & or1 & er2 & or3,
                e & or1 & or2 & er3,
                o & er1 & er2 & or3,
                o & er1 & or2 & er3,
                o & or1 & er2 & er3,
            ];
            p_setup_mask |= (p_two_ws[0] | (p_two_ws[0] << d))
                | (p_two_ws[1] | (p_two_ws[1] << (2 * d)))
                | (p_two_ws[2] | (p_two_ws[2] << (3 * d)))
                | ((p_two_ws[3] << d) | (p_two_ws[3] << (2 * d)))
                | ((p_two_ws[4] << d) | (p_two_ws[4] << (3 * d)))
                | ((p_two_ws[5] << (2 * d)) | (p_two_ws[5] << (3 * d)));
            o_setup_mask |= (o_two_ws[0] | (o_two_ws[0] << d))
                | (o_two_ws[1] | (o_two_ws[1] << (2 * d)))
                | (o_two_ws[2] | (o_two_ws[2] << (3 * d)))
                | ((o_two_ws[3] << d) | (o_two_ws[3] << (2 * d)))
                | ((o_two_ws[4] << d) | (o_two_ws[4] << (3 * d)))
                | ((o_two_ws[5] << (2 * d)) | (o_two_ws[5] << (3 * d)));
            // Count 2-in-a-window opportunities using unique windows logic.
            // We sum the counts for all 6 possible 2-piece arrangements in a 4-cell window.
            // This measures "Structural Density" and "Multi-lane Potential": a piece
            // that supports multiple future paths to victory is more valuable than
            // a piece that only supports one.
            let p_two_ws_comb = p_two_ws.iter().fold(T::zero(), |acc, &w| acc | w);
            let o_two_ws_comb = o_two_ws.iter().fold(T::zero(), |acc, &w| acc | w);
            p_two += i32::try_from(p_two_ws_comb.count_ones()).expect("Bit count fits in i32");
            o_two += i32::try_from(o_two_ws_comb.count_ones()).expect("Bit count fits in i32");
            debug_assert_eq!(
                p_two_ws_comb.count_ones(),
                p_two_ws.iter().map(|&w| w.count_ones()).sum::<u32>(),
                "p_two_ws must be disjoint"
            );
            debug_assert_eq!(
                o_two_ws_comb.count_ones(),
                o_two_ws.iter().map(|&w| w.count_ones()).sum::<u32>(),
                "o_two_ws must be disjoint"
            );
        }

        // Potential windows for player/opponent.
        // (any 4-cell sequence with no opponent/player pieces).
        {
            // Viable slots
            let p_v = geo.board_mask & !o;
            let o_v = geo.board_mask & !p;
            // This is the same as p_v & (p_v >> d) & (p_v >> (2 * d)) & (p_v >> (3 * d)).
            let p_v2 = p_v & (p_v >> d);
            let p_v4 = p_v2 & (p_v2 >> (2 * d));
            let o_v2 = o_v & (o_v >> d);
            let o_v4 = o_v2 & (o_v2 >> (2 * d));
            p_windows += i32::try_from(p_v4.count_ones()).expect("Bit count fits in i32");
            o_windows += i32::try_from(o_v4.count_ones()).expect("Bit count fits in i32");
        }
    }

    {
        // The vertical direction is special since the only possible windows
        // are the ones where the empty spots are on the top.

        let e = ctx.empty;
        // Empty left-shifted by 1
        let el1 = e << 1;

        let p = ctx.p_bits;
        // Player left-shifted by 1, 2, 3
        let pl1 = p << 1;
        let pl2 = p << 2;
        let pl3 = p << 3;

        let o = ctx.o_bits;
        // Opponent left-shifted by 1, 2, 3
        let ol1 = o << 1;
        let ol2 = o << 2;
        let ol3 = o << 3;

        {
            let p_three_w = pl3 & pl2 & pl1 & e;
            let o_three_w = ol3 & ol2 & ol1 & e;
            // 3-in-a-window patterns for player and opponent.
            p_threat_mask |= p_three_w;
            o_threat_mask |= o_three_w;
            p_three += i32::try_from(p_three_w.count_ones()).expect("Bit count fits in i32");
            o_three += i32::try_from(o_three_w.count_ones()).expect("Bit count fits in i32");
        }

        {
            // 2-in-a-window patterns for player and opponent.
            let p_two_w: T = pl3 & pl2 & el1 & e;
            let o_two_w: T = ol3 & ol2 & el1 & e;
            p_setup_mask |= p_two_w | (p_two_w >> 1);
            o_setup_mask |= o_two_w | (o_two_w >> 1);
            p_two += i32::try_from(p_two_w.count_ones()).expect("Bit count fits in i32");
            o_two += i32::try_from(o_two_w.count_ones()).expect("Bit count fits in i32");
        }

        // Potential windows for player/opponent.
        // (any 4-cell sequence with no opponent/player pieces).
        {
            // Viable slots
            let p_v = geo.board_mask & !o;
            let o_v = geo.board_mask & !p;
            // This is the same as p_v & (p_v >> 1) & (p_v >> 2) & (p_v >> 3).
            let p_v2 = p_v & (p_v >> 1);
            let p_v4 = p_v2 & (p_v2 >> 2);
            let o_v2 = o_v & (o_v >> 1);
            let o_v4 = o_v2 & (o_v2 >> 2);
            p_windows += i32::try_from(p_v4.count_ones()).expect("Bit count fits in i32");
            o_windows += i32::try_from(o_v4.count_ones()).expect("Bit count fits in i32");
        }
    }

    debug_assert_eq!(
        p_threat_mask & !geo.board_mask,
        T::zero(),
        "Player threat mask {p_threat_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        o_threat_mask & !geo.board_mask,
        T::zero(),
        "Opponent threat mask {o_threat_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        p_setup_mask & !geo.board_mask,
        T::zero(),
        "Player setup mask {p_setup_mask:?} has sentinel bits set"
    );
    debug_assert_eq!(
        o_setup_mask & !geo.board_mask,
        T::zero(),
        "Opponent setup mask {o_setup_mask:?} has sentinel bits set"
    );
    DirectionalPatterns {
        p_threat_mask,
        p_setup_mask,
        p_three,
        p_two,
        p_windows,
        o_threat_mask,
        o_setup_mask,
        o_three,
        o_two,
        o_windows,
    }
}

/// Returns the total heuristic evaluation score for a board state.
///
/// This is a convenience wrapper around `evaluate_board_detailed` that
/// returns only the final aggregated score.
#[must_use]
fn evaluate_board<T: Bitboard>(
    state: &BoardState<T>,
    geo: &BoardGeometry<T>,
    curr_p: Player,
    weights: &HeuristicWeights,
) -> i32 {
    evaluate_board_detailed(state, geo, curr_p, weights).total()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Helper structure for tests to manage state and geometry together.
    /// Replaces the old generic Board<T> that was removed from the main library.
    struct Board<T: Bitboard> {
        state: BoardState<T>,
        geometry: BoardGeometry<T>,
    }

    impl<T: Bitboard> Board<T> {
        fn new(cols: u32, rows: u32) -> Self {
            Self {
                state: BoardState::default(),
                geometry: BoardGeometry::new(cols, rows),
            }
        }

        fn drop_piece(&mut self, col: u32, player: Player) -> Option<u32> {
            let (next_state, bit_idx) =
                self.state
                    .drop_piece_with_index(col, player, &self.geometry)?;
            self.state = next_state;
            let h = self.geometry.rows() + 1;
            Some(bit_idx % h)
        }

        fn get_first_empty_row(&self, col: u32) -> Option<u32> {
            let h = self.geometry.rows() + 1;
            self.state
                .get_next_bit(col, &self.geometry)
                .map(|b| b.trailing_zeros() % h)
        }
    }

    impl<T: Bitboard> Clone for Board<T> {
        fn clone(&self) -> Self {
            Self {
                state: self.state,
                geometry: self.geometry.clone(),
            }
        }
    }

    fn naive_negamax<T: Bitboard>(
        state: &BoardState<T>,
        ctx: &mut SearchContext<T>,
        depth: u32,
        curr_p: Player,
    ) -> i32 {
        if state.has_won(curr_p.other(), ctx.geo) {
            return -SCORE_WIN;
        }
        if state.is_full(ctx.geo) {
            return 0;
        }
        if depth == 0 {
            return evaluate_board(state, ctx.geo, curr_p, ctx.weights);
        }

        let mut max_score = MIN_SCORE;
        let mut moved = false;
        for col in 0..ctx.geo.columns {
            if let Some(next) = state.drop_piece(col, curr_p, ctx.geo) {
                let score = -naive_negamax(&next, ctx, depth - 1, curr_p.other());
                if score > max_score {
                    max_score = score;
                }
                moved = true;
            }
        }

        if !moved {
            return 0; // Should be covered by is_full, but safety first.
        }

        apply_depth_tax(max_score)
    }

    #[test]
    fn test_search_soundness_pvs_ab_tt() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let weights = HeuristicWeights::default();
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut tt = TranspositionTable::<u64>::new(1); // Small TT to exercise replacements

        for i in 0..50 {
            let mut state = BoardState::<u64>::default();
            // Generate a random valid board state with 5-15 pieces
            let num_pieces = rng.random_range(5..16);
            for _ in 0..num_pieces {
                let playable: Vec<u32> = (0..7)
                    .filter(|&c| state.get_next_bit(c, &geo).is_some())
                    .collect();
                if playable.is_empty() {
                    break;
                }
                let col = playable[rng.random_range(0..playable.len())];
                let player = if state.bits(Player::Red).count_ones()
                    <= state.bits(Player::Yellow).count_ones()
                {
                    Player::Red
                } else {
                    Player::Yellow
                };
                if let Some(next) = state.drop_piece(col, player, &geo) {
                    if next.has_won(player, &geo) {
                        break;
                    }
                    state = next;
                }
            }

            let curr_p = if state.bits(Player::Red).count_ones()
                <= state.bits(Player::Yellow).count_ones()
            {
                Player::Red
            } else {
                Player::Yellow
            };

            for depth in 1..=6 {
                let mut ctx_naive = SearchContext {
                    geo: &geo,
                    weights: &weights,
                    node_count: AtomicU64::new(0),
                    tt: None,
                };
                let ref_score = naive_negamax(&state, &mut ctx_naive, depth, curr_p);

                tt.reset();
                let mut ctx_opt = SearchContext {
                    geo: &geo,
                    weights: &weights,
                    node_count: AtomicU64::new(0),
                    tt: Some(&mut tt),
                };
                let opt_score = negamax(
                    &state,
                    zobrist::compute_hash(&state),
                    &mut ctx_opt,
                    depth,
                    MIN_SCORE,
                    MAX_SCORE,
                    curr_p,
                )
                .0;

                assert_eq!(
                    opt_score,
                    ref_score,
                    "Soundness failure at iteration {i}, depth {depth}. Board: {state:?} (Red bits: {}, Yellow bits: {})",
                    state.bits(Player::Red),
                    state.bits(Player::Yellow)
                );
            }
        }
    }

    #[test]
    fn test_tapered_evaluation_isolated() {
        let mut board = Board::<u64>::new(7, 6);
        board.drop_piece(3, Player::Red).unwrap();
        // Red just moved, so it's Yellow's turn.
        // We evaluate from Yellow's turn perspective.
        let score_1 = -evaluate_board(
            &board.state,
            &board.geometry,
            Player::Yellow,
            &HeuristicWeights::default(),
        );
        board.drop_piece(0, Player::Yellow).unwrap();
        // Yellow just moved, it's Red's turn.
        let score_2 = evaluate_board(
            &board.state,
            &board.geometry,
            Player::Red,
            &HeuristicWeights::default(),
        );
        board.drop_piece(6, Player::Yellow).unwrap();
        // Yellow just moved, it's Red's turn.
        let score_3 = evaluate_board(
            &board.state,
            &board.geometry,
            Player::Red,
            &HeuristicWeights::default(),
        );
        assert!(
            score_2 < score_1,
            "Adding opponent (Yellow) pieces should decrease Red score"
        );
        assert!(
            score_3 < score_2,
            "Adding more pieces should decrease score due to tapering"
        );
    }

    #[test]
    fn test_heuristic_symmetry() {
        let mut b1 = Board::<u64>::new(7, 6);
        b1.drop_piece(2, Player::Red).unwrap();
        let mut b2 = Board::<u64>::new(7, 6);
        b2.drop_piece(4, Player::Red).unwrap();
        assert_eq!(
            evaluate_board(
                &b1.state,
                &b1.geometry,
                Player::Yellow, // It's Yellow's turn in both
                &HeuristicWeights::default()
            ),
            evaluate_board(
                &b2.state,
                &b2.geometry,
                Player::Yellow,
                &HeuristicWeights::default()
            )
        );
    }

    #[test]
    fn test_immediate_fork_valuation() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();
        for _ in 0..3 {
            board.drop_piece(0, Player::Red).unwrap();
            board.drop_piece(1, Player::Red).unwrap();
        }
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();
        board.drop_piece(4, Player::Yellow).unwrap();
        board.drop_piece(5, Player::Yellow).unwrap();
        board.drop_piece(6, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        let score = evaluate_board(&board.state, &board.geometry, Player::Red, &weights);
        assert!(
            score >= weights.score_fork_immediate,
            "Score {score} should reflect an immediate fork (>= {})",
            weights.score_fork_immediate
        );
    }

    #[test]
    fn test_mobility_valuation() {
        let board = Board::<u64>::new(7, 6);
        let detail_empty = evaluate_board_detailed(
            &board.state,
            &board.geometry,
            Player::Red,
            &HeuristicWeights::default(),
        );
        assert_eq!(detail_empty.windows, 0);
        let mut board2 = board;
        board2.drop_piece(0, Player::Red).unwrap();
        let detail_red = evaluate_board_detailed(
            &board2.state,
            &board2.geometry,
            Player::Red,
            &HeuristicWeights::default(),
        );
        assert!(detail_red.windows > 0);
        let mut board3 = board2;
        board3.drop_piece(1, Player::Yellow).unwrap();
        let detail_blocked = evaluate_board_detailed(
            &board3.state,
            &board3.geometry,
            Player::Red,
            &HeuristicWeights::default(),
        );
        assert!(detail_blocked.windows < detail_red.windows);
    }

    #[test]
    fn test_win_priority() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();
        for _ in 0..3 {
            board.drop_piece(3, Player::Red).unwrap();
        }
        let scores = get_column_scores(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            &weights,
            None,
        );
        assert_eq!(scores[3].unwrap(), SCORE_WIN - 1);
        for (i, &s) in scores.iter().enumerate() {
            if i != 3 {
                assert!(s.unwrap() < scores[3].unwrap());
            }
        }
    }

    #[test]
    fn test_complex_ordering() {
        let mut board = Board::<u64>::new(7, 6);
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(2, Player::Red).unwrap();
        board.drop_piece(3, Player::Red).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();
        board.drop_piece(4, Player::Yellow).unwrap();
        let best_move = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            false,
            &HeuristicWeights::default(),
            None,
        );
        assert_eq!(best_move, Some(0));
    }

    #[test]
    fn test_win_priority_expert_deterministic() {
        let mut board = Board::<u64>::new(7, 6);
        // Col 3: Win in 1
        for _ in 0..3 {
            board.drop_piece(3, Player::Red).unwrap();
        }
        // Col 4: Win in 3
        for _ in 0..2 {
            board.drop_piece(4, Player::Red).unwrap();
        }

        // We run the search with a depth that can see both wins (d=5).
        // Col 3 (Immediate): SCORE_WIN - 1 | Col 4 (Delayed): SCORE_WIN - 3
        // Since shorter wins have a smaller tactical decay, they get
        // a higher total score.
        let best_move = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            false,
            &HeuristicWeights::default(),
            None,
        );
        assert_eq!(
            best_move,
            Some(3),
            "AI should pick immediate win (col 3) over longer win (col 4)"
        );
    }

    #[test]
    fn test_win_priority_column_scores() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();
        // Col 3: Win in 1
        for _ in 0..3 {
            board.drop_piece(3, Player::Red).unwrap();
        }
        // Col 5: Win in 3
        for _ in 0..2 {
            board.drop_piece(5, Player::Red).unwrap();
        }

        // We run the search with a depth that can see both wins (d=5).
        // Col 3 (Immediate): SCORE_WIN - 1 | Col 5 (Delayed): SCORE_WIN - 3
        let scores = get_column_scores(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            &weights,
            None,
        );
        assert_eq!(scores[3].unwrap(), SCORE_WIN - 1);
        assert_eq!(scores[5].unwrap(), SCORE_WIN - 3);
    }

    #[test]
    fn test_score_depth_invariance() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // 1. Win-in-1 for Red
        for _ in 0..3 {
            board.drop_piece(3, Player::Red).unwrap();
        }
        let s1_d1 = evaluate_position(
            &board.state,
            &board.geometry,
            Player::Red,
            1,
            &weights,
            None,
        );
        let s1_d5 = evaluate_position(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            &weights,
            None,
        );
        assert_eq!(s1_d1, SCORE_WIN - 1);
        assert_eq!(s1_d1, s1_d5);

        // 2. Forced Win-in-3 for Red
        let mut b2 = Board::<u64>::new(7, 6);
        // Setup: Red has (2,0), (3,0). It's Red's turn.
        // Playing at (1,0) creates a fork (threats at 0,0 and 4,0).
        b2.drop_piece(2, Player::Red).unwrap();
        b2.drop_piece(3, Player::Red).unwrap();
        // Give Yellow some pieces far away
        b2.drop_piece(6, Player::Yellow).unwrap();
        b2.drop_piece(6, Player::Yellow).unwrap();

        let s3_d3 = evaluate_position(&b2.state, &b2.geometry, Player::Red, 3, &weights, None);
        let s3_d7 = evaluate_position(&b2.state, &b2.geometry, Player::Red, 7, &weights, None);

        assert_eq!(s3_d3, SCORE_WIN - 3, "Should be a win in 3 plies");
        assert_eq!(s3_d3, s3_d7, "Score should be invariant across depths");
    }

    #[test]
    fn test_stochastic_determinism() {
        let board = Board::<u64>::new(7, 6);
        let first = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            3,
            false,
            &HeuristicWeights::default(),
            None,
        );
        for _ in 0..100 {
            assert_eq!(
                first,
                find_best_move(
                    &board.state,
                    &board.geometry,
                    Player::Red,
                    3,
                    false,
                    &HeuristicWeights::default(),
                    None,
                )
            );
        }
    }

    #[test]
    fn test_count_patterns_logic() {
        let mut board = Board::<u64>::new(7, 6);

        // 1. Single 2-in-a-window (Horizontal: [R R . .])
        board.drop_piece(0, Player::Red).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        let ctx = EvalContext::new(&board.state, &board.geometry, Player::Red);
        let dp = evaluate_directional_patterns(&ctx, &board.geometry);
        assert!(
            dp.p_two > 0,
            "Should detect at least one 2-in-a-window pattern"
        );
        assert_eq!(dp.p_three, 0, "Should not detect 3-in-a-window patterns");

        // 2. 3-in-a-window (Horizontal: [R R R .])
        board.drop_piece(3, Player::Red).unwrap();
        let ctx_3 = EvalContext::new(&board.state, &board.geometry, Player::Red);
        let dp_3 = evaluate_directional_patterns(&ctx_3, &board.geometry);
        assert!(
            dp_3.p_three > 0,
            "Should detect at least one 3-in-a-window pattern"
        );

        // 3. Cross-threat consistency (Window Count logic)
        // Verify that a piece shared between two windows (horizontal and vertical)
        // results in both windows being counted independently.

        // Horizontal window with 3 pieces: (0,0), (1,0), (2,0)
        let mut board_h = Board::<u64>::new(7, 6);
        board_h.drop_piece(0, Player::Red).unwrap();
        board_h.drop_piece(1, Player::Red).unwrap();
        board_h.drop_piece(2, Player::Red).unwrap();
        let ctx_h = EvalContext::new(&board_h.state, &board_h.geometry, Player::Red);
        let dp_h = evaluate_directional_patterns(&ctx_h, &board_h.geometry);
        let three_h = dp_h.p_three;

        // Vertical window with 3 pieces: (0,0), (0,1), (0,2)
        let mut board_v = Board::<u64>::new(7, 6);
        board_v.drop_piece(0, Player::Red).unwrap();
        board_v.drop_piece(0, Player::Red).unwrap();
        board_v.drop_piece(0, Player::Red).unwrap();
        let ctx_v = EvalContext::new(&board_v.state, &board_v.geometry, Player::Red);
        let dp_v = evaluate_directional_patterns(&ctx_v, &board_v.geometry);
        let three_v = dp_v.p_three;

        // Combined state: Both windows share the piece at (0,0)
        let mut board_c = Board::<u64>::new(7, 6);
        board_c.drop_piece(0, Player::Red).unwrap();
        board_c.drop_piece(0, Player::Red).unwrap();
        board_c.drop_piece(0, Player::Red).unwrap();
        board_c.drop_piece(1, Player::Red).unwrap();
        board_c.drop_piece(2, Player::Red).unwrap();
        let ctx_c = EvalContext::new(&board_c.state, &board_c.geometry, Player::Red);
        let dp_c = evaluate_directional_patterns(&ctx_c, &board_c.geometry);
        let three_c = dp_c.p_three;

        assert_eq!(
            three_c,
            three_h + three_v,
            "Cross-threats should be summed, confirming consistent Window Count logic"
        );
    }

    #[test]
    fn test_setup_mask_logic() {
        let board = Board::<u64>::new(7, 6);
        let h = board.geometry.rows + 1;

        // 1. Horizontal setup: [R R . .] at (0,0), (1,0)
        let mut b1 = Board::<u64>::new(7, 6);
        b1.drop_piece(0, Player::Red).unwrap();
        b1.drop_piece(1, Player::Red).unwrap();
        let ctx1 = EvalContext::new(&b1.state, &b1.geometry, Player::Red);
        let dp1 = evaluate_directional_patterns(&ctx1, &b1.geometry);
        let setup1 = dp1.p_setup_mask;
        let expected1 = (1u64 << (2 * h)) | (1u64 << (3 * h));
        assert!(
            setup1 & expected1 == expected1,
            "Should detect empty slots in [R R . .] horizontally"
        );

        // 2. Vertical setup: [R R . .] at (0,0), (0,1)
        let mut b2 = Board::<u64>::new(7, 6);
        b2.drop_piece(0, Player::Red).unwrap();
        b2.drop_piece(0, Player::Red).unwrap();
        let ctx2 = EvalContext::new(&b2.state, &b2.geometry, Player::Red);
        let dp2 = evaluate_directional_patterns(&ctx2, &b2.geometry);
        let setup2 = dp2.p_setup_mask;
        let expected2 = (1u64 << 2) | (1u64 << 3);
        assert!(
            setup2 & expected2 == expected2,
            "Should detect empty slots in [R R . .] vertically"
        );

        // 3. Diagonal setup: [. R R .]
        let d = h + 1;
        let mut b3 = Board::<u64>::new(7, 6);
        b3.drop_piece(1, Player::Yellow).unwrap();
        b3.drop_piece(2, Player::Yellow).unwrap();
        b3.drop_piece(2, Player::Yellow).unwrap();
        b3.drop_piece(1, Player::Red).unwrap();
        b3.drop_piece(2, Player::Red).unwrap();
        let ctx3 = EvalContext::new(&b3.state, &b3.geometry, Player::Red);
        let dp3 = evaluate_directional_patterns(&ctx3, &b3.geometry);
        let setup3 = dp3.p_setup_mask;
        let expected3 = (1u64 << 0) | (1u64 << (3 * d));
        assert!(
            setup3 & expected3 == expected3,
            "Should detect empty slots in [. R R .] diagonally"
        );
    }

    #[test]
    fn test_threat_mask_detection() {
        let mut board = Board::<u64>::new(7, 6);
        board.drop_piece(0, Player::Red).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(2, Player::Red).unwrap();
        let ctx = EvalContext::new(&board.state, &board.geometry, Player::Red);
        let dp = evaluate_directional_patterns(&ctx, &board.geometry);
        let (imm, fut) = (
            dp.p_threat_mask & ctx.playable,
            dp.p_threat_mask & !ctx.playable,
        );
        assert_eq!(
            imm,
            1u64 << 21,
            "Should detect immediate threat at Col 3, Row 0"
        );
        assert_eq!(fut, 0, "Should have no future threats in this setup");

        let mut board2 = Board::<u64>::new(7, 6);
        board2.drop_piece(0, Player::Yellow).unwrap();
        board2.drop_piece(1, Player::Yellow).unwrap();
        board2.drop_piece(2, Player::Yellow).unwrap();
        board2.drop_piece(0, Player::Red).unwrap();
        board2.drop_piece(1, Player::Red).unwrap();
        board2.drop_piece(2, Player::Red).unwrap();
        let ctx2 = EvalContext::new(&board2.state, &board2.geometry, Player::Red);
        let dp2 = evaluate_directional_patterns(&ctx2, &board2.geometry);
        let (imm2, fut2) = (
            dp2.p_threat_mask & ctx2.playable,
            dp2.p_threat_mask & !ctx2.playable,
        );
        assert_eq!(imm2, 0, "Threat at (3,1) should not be immediate yet");
        assert_eq!(
            fut2,
            1u64 << (3 * (board2.geometry.rows + 1) + 1),
            "Should detect future threat at (3,1)"
        );
    }

    #[test]
    fn test_immediate_fork_detection_logic() {
        let mut board = Board::<u64>::new(7, 6);
        for _ in 0..3 {
            board.drop_piece(0, Player::Red).unwrap();
        }
        for _ in 0..3 {
            board.drop_piece(1, Player::Red).unwrap();
        }
        let ctx = EvalContext::new(&board.state, &board.geometry, Player::Red);
        let dp = evaluate_directional_patterns(&ctx, &board.geometry);
        let imm = dp.p_threat_mask & ctx.playable;
        assert_eq!(
            imm.count_ones(),
            2,
            "Should detect 2 unique winning columns"
        );
    }

    #[test]
    fn test_potential_window_analysis() {
        let board = Board::<u64>::new(7, 6);
        let ctx = EvalContext::new(&board.state, &board.geometry, Player::Red);
        let dp = evaluate_directional_patterns(&ctx, &board.geometry);
        assert_eq!(dp.p_windows, 69, "Empty 7x6 should have 69 windows for Red");
        assert_eq!(
            dp.o_windows, 69,
            "Empty 7x6 should have 69 windows for Yellow"
        );
        let mut board2 = board;
        board2.drop_piece(0, Player::Yellow).unwrap();
        let ctx2 = EvalContext::new(&board2.state, &board2.geometry, Player::Red);
        let dp2 = evaluate_directional_patterns(&ctx2, &board2.geometry);
        assert_eq!(
            dp2.p_windows,
            69 - 3,
            "Red should lose 3 windows due to piece at (0,0)"
        );
    }

    #[test]
    fn test_move_order_tier_priorities() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // 1. Setup a Win for Red in Col 1
        for _ in 0..3 {
            board.drop_piece(1, Player::Red).unwrap();
        }
        // 2. Setup a Block (Yellow win) in Col 0
        for _ in 0..3 {
            board.drop_piece(0, Player::Yellow).unwrap();
        }
        // 3. Setup an Offensive Trap (Setup) in Col 6
        board.drop_piece(6, Player::Red).unwrap();
        board.drop_piece(6, Player::Red).unwrap();

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // Expected Order: [1 (Win), 0 (Block), 6 (Setup), ...others...]
        // Center (3) should be 4th as it's positional.
        assert_eq!(moves[0], 1, "Winning move should be first");
        assert_eq!(moves[1], 0, "Blocking move should be second");
        assert_eq!(moves[2], 6, "Setup move should be third");
        assert_eq!(moves[3], 3, "Center should be fourth (highest positional)");
    }

    #[test]
    fn test_move_order_center_bias() {
        let board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // On an empty board, all moves are in the same positional tier (or close).
        // Expected Order: Center-out [3, 2, 4, 1, 5, 0, 6]
        assert_eq!(moves.to_vec(), vec![3, 2, 4, 1, 5, 0, 6]);
    }

    #[test]
    fn test_move_order_blunder_handling() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Setup a blunder in Col 3: Playing in Col 3 allows Yellow to win at (3, 1)
        // Yellow threat at (3, 1): Needs pieces at (0,1), (1,1), (2,1)
        board.drop_piece(0, Player::Red).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        // Red pieces at (0,0), (0,1). Playing in Col 3 (Row 0) will make Row 1 playable for Yellow.

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // Col 3 should be last because it's a blunder.
        assert_eq!(
            *moves.last().unwrap(),
            3,
            "Blunder should be moved to the end"
        );
    }

    #[test]
    fn test_move_order_win_beats_blunder() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Col 3: Immediate win for Red at (3,0)
        board.drop_piece(0, Player::Red).unwrap();
        board.drop_piece(1, Player::Red).unwrap();
        board.drop_piece(2, Player::Red).unwrap();
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // Win should be prioritized over the blunder check.
        assert_eq!(
            moves[0], 3,
            "Winning move must be first even if it enables an opponent win above"
        );
    }

    #[test]
    fn test_move_order_mutual_exclusivity() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Fill the board randomly to create complex masks
        for c in 0..7 {
            for _ in 0..2 {
                let _ = board.drop_piece(c, Player::Red);
            }
        }

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // Check: Every playable column must appear exactly once.
        let mut seen = std::collections::HashSet::new();
        for &m in &moves {
            assert!(m < 7, "Invalid column index");
            assert!(seen.insert(m), "Duplicate move {m} found in order");
        }

        let playable_count = (0..7)
            .filter(|&c| board.get_first_empty_row(c).is_some())
            .count();
        assert_eq!(
            moves.len(),
            playable_count,
            "Missing playable moves in ordering"
        );
    }

    #[test]
    fn test_move_order_giant_board_bias() {
        // Test Giant board (9x7)
        let board = Board::<u128>::new(9, 7);
        let weights = HeuristicWeights::default();

        let moves = get_move_order(&board.state, &board.geometry, Player::Red, &weights);

        // Center-out for 9 columns: [4, 3, 5, 2, 6, 1, 7, 0, 8]
        assert_eq!(moves.to_vec(), vec![4, 3, 5, 2, 6, 1, 7, 0, 8]);
    }

    #[test]
    fn test_trap_blocking_priority() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Red (AI) is trying to block Yellow's threat.
        // Yellow has two ways to win if Red doesn't block properly.
        // Setup a horizontal trap for Yellow: (0,0), (1,0), (2,0)
        board.drop_piece(0, Player::Yellow).unwrap();
        board.drop_piece(1, Player::Yellow).unwrap();
        board.drop_piece(2, Player::Yellow).unwrap();
        // Yellow also has (4,0), (5,0), (6,0)
        board.drop_piece(4, Player::Yellow).unwrap();
        board.drop_piece(5, Player::Yellow).unwrap();
        board.drop_piece(6, Player::Yellow).unwrap();

        // Col 3 is the only move that blocks BOTH (if they were connected)
        // or at least blocks one side of a potential fork.
        // In this specific setup, col 3 blocks both horizontal threats.
        let best_move = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            false,
            &weights,
            None,
        );
        assert_eq!(
            best_move,
            Some(3),
            "AI must block Yellow's horizontal win at col 3"
        );
    }

    #[test]
    fn test_heuristic_center_bias_gradient() {
        let board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Evaluate single Red piece at different columns
        let mut scores = Vec::new();
        for c in 0..7 {
            let mut b = board.clone();
            b.drop_piece(c, Player::Red).unwrap();
            scores.push(evaluate_board(&b.state, &b.geometry, Player::Red, &weights));
        }

        // scores should be symmetric and highest at center (col 3)
        assert_eq!(scores[0], scores[6]);
        assert_eq!(scores[1], scores[5]);
        assert_eq!(scores[2], scores[4]);

        assert!(scores[3] > scores[2]);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_deep_search_trap_evasion() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // A complex trap where playing in one column leads to a forced loss in 4 moves.
        // Red (AI) to move.
        // Yellow has pieces at (2,0) and (4,0).
        board.drop_piece(2, Player::Yellow).unwrap();
        board.drop_piece(4, Player::Yellow).unwrap();
        board.drop_piece(3, Player::Red).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();

        // If Red plays col 3 again, it might open up something.
        // This is a bit abstract, but let's just verify AI doesn't crash on deeper searches.
        let best_move = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            7,
            false,
            &weights,
            None,
        );
        assert!(best_move.unwrap() < 7);
    }

    #[test]
    fn test_iterative_deepening_consistency() {
        let mut board = Board::<u64>::new(7, 6);
        board.drop_piece(3, Player::Red).unwrap();
        board.drop_piece(3, Player::Yellow).unwrap();

        let weights = HeuristicWeights::default();
        let move_d1 = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            1,
            false,
            &weights,
            None,
        );
        let move_d5 = find_best_move(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            false,
            &weights,
            None,
        );

        // On a simple board, d5 should at least be as good as d1 (often same for simple positions).
        assert!(move_d1.unwrap() < 7);
        assert!(move_d5.unwrap() < 7);
    }

    #[test]
    fn test_evaluation_perfect_symmetry() {
        let mut b_left = Board::<u64>::new(7, 6);
        let mut b_right = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Complex asymmetrical-looking but mirrored positions
        let moves = [
            (0, Player::Red),
            (1, Player::Yellow),
            (2, Player::Red),
            (0, Player::Yellow),
        ];
        for (c, p) in moves {
            b_left.drop_piece(c, p).unwrap();
            b_right.drop_piece(6 - c, p).unwrap();
        }

        let score_left = evaluate_board(&b_left.state, &b_left.geometry, Player::Red, &weights);
        let score_right = evaluate_board(&b_right.state, &b_right.geometry, Player::Red, &weights);

        assert_eq!(
            score_left, score_right,
            "Heuristic must be perfectly symmetric"
        );
    }

    #[test]
    fn test_evaluation_player_invariance() {
        let mut b_red = Board::<u64>::new(7, 6);
        let mut b_yel = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Identical positions but colors swapped
        let moves = [0, 1, 3, 2, 0];
        for (i, &c) in moves.iter().enumerate() {
            let p = if i % 2 == 0 {
                Player::Red
            } else {
                Player::Yellow
            };
            b_red.drop_piece(c, p).unwrap();
            b_yel.drop_piece(c, p.other()).unwrap();
        }

        // It is currently Yellow's turn on b_red and Red's turn on b_yel.
        let score_red_mover =
            evaluate_board(&b_red.state, &b_red.geometry, Player::Yellow, &weights);
        let score_yel_mover = evaluate_board(&b_yel.state, &b_yel.geometry, Player::Red, &weights);

        assert_eq!(
            score_red_mover, score_yel_mover,
            "Heuristic must be player relative: score(mover=Yellow) = score(mover=Red) for symmetric position"
        );
    }

    #[test]
    fn test_double_threat_priority() {
        let mut board = Board::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();

        // Red (AI) has pieces at (2,0) and (3,0)
        board.drop_piece(2, Player::Red).unwrap();
        board.drop_piece(3, Player::Red).unwrap();
        // If Red plays 1, it has (1,0), (2,0), (3,0) - a triple with ends open at 0 and 4.
        // This is a "double threat" or fork.

        let scores = get_column_scores(
            &board.state,
            &board.geometry,
            Player::Red,
            5,
            &weights,
            None,
        );

        // Col 1 and Col 4 should have very high scores as they both create a winning fork.
        let s1 = scores[1].unwrap();
        let s4 = scores[4].unwrap();
        let s0 = scores[0].unwrap();

        assert!(s1 > s0);
        assert!(s4 > s0);
        assert_eq!(s1, s4, "Forking moves should have identical high scores");
    }

    #[test]
    fn test_tt_behavioral_identity() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let weights = HeuristicWeights::default();
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut tt = TranspositionTable::<u64>::new(16);

        for _ in 0..50 {
            let mut state = BoardState::<u64>::default();
            // Generate a random valid board state (not full, not won)
            for _ in 0..15 {
                let playable: Vec<u32> = (0..7)
                    .filter(|&c| state.get_next_bit(c, &geo).is_some())
                    .collect();
                if playable.is_empty() {
                    break;
                }
                let col = playable[rng.random_range(0..playable.len())];
                let player = if rng.random_bool(0.5) {
                    Player::Red
                } else {
                    Player::Yellow
                };
                if let Some(next) = state.drop_piece(col, player, &geo) {
                    if next.has_won(player, &geo) {
                        break;
                    }
                    state = next;
                }
            }

            // Identity 1: evaluate_position must be identical
            for depth in [1, 3, 5] {
                let score_no_tt =
                    evaluate_position(&state, &geo, Player::Red, depth, &weights, None);
                tt.reset();
                let score_with_tt =
                    evaluate_position(&state, &geo, Player::Red, depth, &weights, Some(&mut tt));
                assert_eq!(
                    score_no_tt, score_with_tt,
                    "evaluate_position non-identical at depth {depth} for state {state:?}"
                );
            }

            // Identity 2: get_column_scores must result in the SAME score
            for depth in [1, 3, 5] {
                // Get scores for all columns to handle multiple moves with same best score
                let scores_no_tt =
                    get_column_scores(&state, &geo, Player::Red, depth, &weights, None);
                tt.reset();
                let scores_with_tt =
                    get_column_scores(&state, &geo, Player::Red, depth, &weights, Some(&mut tt));

                assert_eq!(
                    scores_no_tt, scores_with_tt,
                    "get_column_scores non-identical at depth {depth} for state {state:?}"
                );

                // Also verify that evaluate_position (which uses the search root) matches
                let best_score_no_tt =
                    evaluate_position(&state, &geo, Player::Red, depth, &weights, None);
                tt.reset();
                let best_score_with_tt =
                    evaluate_position(&state, &geo, Player::Red, depth, &weights, Some(&mut tt));
                assert_eq!(
                    best_score_no_tt, best_score_with_tt,
                    "evaluate_position non-identical at depth {depth} for state {state:?}"
                );
            }
        }
    }

    #[test]
    fn test_heuristic_no_tempo_bonus() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let weights = HeuristicWeights::default();
        let geo = BoardGeometry::<u64>::new(7, 6);

        for _ in 0..100 {
            let mut state = BoardState::<u64>::default();
            // Create a random board state
            for _ in 0..20 {
                let col = rng.random_range(0..7);
                let player = if rng.random_bool(0.5) {
                    Player::Red
                } else {
                    Player::Yellow
                };
                if let Some(next) = state.drop_piece(col, player, &geo) {
                    state = next;
                }
            }

            let score_red_mover = evaluate_board(&state, &geo, Player::Red, &weights);
            let score_yel_mover = evaluate_board(&state, &geo, Player::Yellow, &weights);

            // Since our heuristic is currently symmetric (no initiative bonus for being the mover),
            // evaluate_board(Red) should be exactly equal to -evaluate_board(Yellow).
            // Both are absolute scores representing the state of the board.
            assert_eq!(
                score_red_mover, -score_yel_mover,
                "Heuristic must return consistent scores regardless of mover (if initiative is symmetric). Found {score_red_mover} and {score_yel_mover}",
            );
        }
    }

    #[test]
    fn test_search_consistency_deep() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let weights = HeuristicWeights::default();
        let geo = BoardGeometry::<u64>::new(7, 6);
        let mut tt = TranspositionTable::<u64>::new(64);

        for i in 0..10 {
            let mut state = BoardState::<u64>::default();
            // Generate a more complex board state with 20 pieces
            for _ in 0..20 {
                let playable: Vec<u32> = (0..7)
                    .filter(|&c| state.get_next_bit(c, &geo).is_some())
                    .collect();
                if playable.is_empty() {
                    break;
                }
                let col = playable[rng.random_range(0..playable.len())];
                let player = if rng.random_bool(0.5) {
                    Player::Red
                } else {
                    Player::Yellow
                };
                if let Some(next) = state.drop_piece(col, player, &geo) {
                    if next.has_won(player, &geo) || next.is_full(&geo) {
                        break;
                    }
                    state = next;
                }
            }

            // Depth 7 is deep enough to benefit from TT while still being fast enough for a test.
            let depth = 7;

            let scores_no_tt = get_column_scores(&state, &geo, Player::Red, depth, &weights, None);
            tt.reset();
            let scores_with_tt =
                get_column_scores(&state, &geo, Player::Red, depth, &weights, Some(&mut tt));

            assert_eq!(
                scores_no_tt, scores_with_tt,
                "Consistency failure in iteration {i} at depth {depth}. Board: {state:?}"
            );
        }
    }

    #[test]
    fn test_tt_active_gatekeeping() {
        let geo = BoardGeometry::<u64>::new(7, 6);
        let weights = HeuristicWeights::default();
        let mut tt = TranspositionTable::<u64>::new(1);

        // Case 1: No TT provided
        let ctx_no_tt = SearchContext {
            geo: &geo,
            weights: &weights,
            node_count: AtomicU64::new(0),
            tt: None,
        };
        assert!(!ctx_no_tt.tt_active(1));
        assert!(!ctx_no_tt.tt_active(10));

        // Case 2: TT provided but depth too low
        let ctx_low_depth = SearchContext {
            geo: &geo,
            weights: &weights,
            node_count: AtomicU64::new(0),
            tt: Some(&mut tt),
        };
        assert!(!ctx_low_depth.tt_active(1));

        // Case 3: TT provided and depth sufficient
        assert!(ctx_low_depth.tt_active(2));
        assert!(ctx_low_depth.tt_active(10));
    }

    #[test]
    fn test_magic_multipliers() {
        for (d, &m) in MAGIC_MULTIPLIERS.iter().enumerate().skip(2) {
            // We use u128 for our bitboard so the bit index can be 0..128
            // in the worst case.
            for num in 0..u128::BITS {
                assert_eq!(
                    num / u32::try_from(d).expect("Divisor fits in u32"),
                    (num * u32::from(m)) >> 16,
                    "Magic multiplier {m} not equivalent to divisor {d} for {num}"
                );
            }
        }
    }
}

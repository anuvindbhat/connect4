//! # Connect 4 AI Simulation and Tournament Tool
//!
//! This tool runs a tournament between different AI configurations to estimate their
//! relative Elo ratings across various board geometries. It uses the Bradley-Terry model
//! to calculate Elo ratings from match results and provides confidence intervals.

use connect4::config::HeuristicWeights;
use connect4::engine::DynamicEngine;
use connect4::game::{DynamicBoardGeometry, DynamicBoardState};
use connect4::types::Player;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

// ========================================================================================
// TOURNAMENT CONFIGURATION
// ========================================================================================

/// Minimum number of game pairs required before a matchup can be terminated early due to high confidence.
const MIN_PAIRS: usize = 20;
/// Maximum number of game pairs to play for any single matchup.
const MAX_PAIRS: usize = 500;
/// Number of game pairs to process in parallel in each iteration.
const BATCH_SIZE: usize = 8;
/// Directory where simulation logs are stored.
const LOG_DIR: &str = "logs";

/// The Elo rating assigned to the anchor competitor (usually Depth 1).
const ANCHOR_ELO: f64 = 1000.0;

// DEF (Dynamic Evaluative Filtering) constants
/// Depth used for evaluating the quality of randomly generated openings.
const DEF_EVAL_DEPTH: u32 = 7;
/// Maximum number of attempts to generate a balanced opening before settling for the best one found.
const DEF_MAX_RETRIES: usize = 30;
/// Minimum number of moves in a generated opening.
const OPENING_MIN_MOVES: usize = 0;
/// Maximum number of moves in a generated opening.
const OPENING_MAX_MOVES: usize = 6;

/// Board geometries (columns, rows) used in the tournament.
const GEOMETRIES: [(u32, u32); 3] = [(7, 6), (8, 7), (9, 7)];

/// Default size for Transposition Tables used in competitors' searches.
const SIM_TT_SIZE_MB: usize = 64;

/// Size for the temporary Transposition Table used during opening generation.
const OPENING_TT_SIZE_MB: usize = 64;

// ========================================================================================
// DATA MODELS
// ========================================================================================

/// Represents a specific AI configuration participating in the tournament.
#[derive(Clone, Debug)]
struct Competitor {
    /// Display name of the competitor.
    name: String,
    /// Search depth limit for the AI.
    depth: u32,
    /// Heuristic weights used by the AI's evaluation function.
    weights: HeuristicWeights,
}

/// Represents a game opening sequence.
#[derive(Clone, Debug, Default)]
struct Opening {
    /// Sequence of column indices representing the moves in the opening.
    moves: Vec<u32>,
}

/// Tracks the results of a matchup between two competitors.
/// Results are stored in game pairs (where colors are swapped) to mitigate first-move advantage.
#[derive(Clone, Debug, Default)]
struct MatchHistory {
    /// Counts of results for game pairs, indexed by the pentanomial distribution:
    /// 0: LL (Both lost), 1: LD (Loss/Draw), 2: DD/WL (Double Draw or Win/Loss), 3: WD (Win/Draw), 4: WW (Both won).
    pair_counts: [usize; 5],
    /// Total number of game pairs played.
    game_pairs: usize,
}

/// Stores Elo results for a specific board geometry.
struct GeometryResult {
    /// Estimated Elo ratings for each competitor.
    elos: Vec<f64>,
    /// 95% confidence interval margins for each Elo rating.
    margins: Vec<f64>,
}

impl MatchHistory {
    /// Updates the match history with the results of a game pair.
    /// `r1` is the result for Player 1 playing as Red.
    /// `r2` is the result for Player 1 playing as Yellow.
    /// Result values: 1.0 for win, 0.5 for draw, 0.0 for loss.
    fn update_pair(&mut self, r1: f32, r2: f32) {
        let pair_score = r1 + r2;
        let bucket = if pair_score < 0.25 {
            0 // LL
        } else if pair_score < 0.75 {
            1 // LD
        } else if pair_score < 1.25 {
            2 // DD/WL
        } else if pair_score < 1.75 {
            3 // WD
        } else {
            4 // WW
        };
        self.pair_counts[bucket] += 1;
        self.game_pairs += 1;
    }

    /// Merges another `MatchHistory` into this one.
    fn add(&mut self, other: &Self) {
        for i in 0..5 {
            self.pair_counts[i] += other.pair_counts[i];
        }
        self.game_pairs += other.game_pairs;
    }

    /// Calculates the total score for this competitor.
    /// Scores: LL=0.0, LD=0.5, DD/WL=1.0, WD=1.5, WW=2.0.
    #[allow(clippy::cast_precision_loss)]
    fn score(&self) -> f32 {
        self.pair_counts[1] as f32 * 0.5
            + self.pair_counts[2] as f32 * 1.0
            + self.pair_counts[3] as f32 * 1.5
            + self.pair_counts[4] as f32 * 2.0
    }

    /// Calculates the overall win rate (between 0.0 and 1.0).
    #[allow(clippy::cast_precision_loss)]
    fn win_rate(&self) -> f64 {
        if self.game_pairs == 0 {
            return 0.5;
        }
        f64::from(self.score()) / (self.game_pairs as f64 * 2.0)
    }

    /// Returns (wins, total trials) for use in the Bradley-Terry model.
    /// Each game pair is treated as 2 independent trials.
    #[allow(clippy::cast_precision_loss)]
    fn to_bt_stats(&self) -> (f64, f64) {
        (f64::from(self.score()), self.game_pairs as f64 * 2.0)
    }

    /// Calculates the estimated Elo difference and 95% confidence interval margin.
    /// Uses the standard error of the mean for the pentanomial distribution of game pair results.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_elo_diff(&self) -> (f64, f64) {
        let n = self.game_pairs as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }

        let mu = self.win_rate();
        // Pentanomial variance calculation
        let mut sum_sq = 0.0;
        let weights: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        for (i, &w) in weights.iter().enumerate() {
            sum_sq += self.pair_counts[i] as f64 * w.powi(2);
        }
        let sigma2 = (sum_sq / n - mu.powi(2)).max(0.01);
        let stderr = (sigma2 / n).sqrt();

        // 95% confidence interval (1.96 standard errors)
        let ci_wr = 1.96 * stderr;

        let wr_to_elo = |wr: f64| -400.0 * (1.0 / wr.clamp(0.001, 0.999) - 1.0).ln() / 10.0f64.ln();

        let elo_diff = wr_to_elo(mu);
        let elo_high = wr_to_elo(mu + ci_wr);
        let elo_margin = (elo_high - elo_diff).abs();

        (elo_diff, elo_margin)
    }
}

// ========================================================================================
// BRADLEY-TERRY SOLVER (Zero-Dependency)
// ========================================================================================

/// Results from the Bradley-Terry solver.
struct BtSolverResult {
    /// Estimated Elo ratings for each competitor.
    elos: Vec<f64>,
    /// 95% confidence interval margins for each Elo rating.
    margins: Vec<f64>,
}

/// Simple matrix inversion using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular or near-singular.
fn invert_matrix(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut aug: Vec<Vec<f64>> = vec![vec![0.0; 2 * n]; n];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            aug[i][j] = val;
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot = i;
        for j in (i + 1)..n {
            if aug[j][i].abs() > aug[pivot][i].abs() {
                pivot = j;
            }
        }
        aug.swap(i, pivot);

        let factor = aug[i][i];
        if factor.abs() < 1e-12 {
            return None;
        }

        for val in &mut aug[i] {
            *val /= factor;
        }

        for k in 0..n {
            if k != i {
                let f = aug[k][i];
                let (row_k, row_i) = if k < i {
                    let (left, right) = aug.split_at_mut(i);
                    (&mut left[k], &right[0])
                } else {
                    let (left, right) = aug.split_at_mut(k);
                    (&mut right[0], &left[i])
                };
                for (v_k, &v_i) in row_k.iter_mut().zip(row_i.iter()) {
                    *v_k -= f * v_i;
                }
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

/// Solves the Bradley-Terry model using the Minorization-Maximization (MM) algorithm.
///
/// This algorithm estimates the relative strengths of competitors based on win/loss
/// data. It includes Laplace smoothing for stability and calculates uncertainty
/// margins by inverting the Fisher Information matrix.
fn solve_bradley_terry(
    n_competitors: usize,
    results: &HashMap<(usize, usize), MatchHistory>,
) -> BtSolverResult {
    // 1. Build Win and Total Matrices with pseudo-counts for stability (0.01 wins / 0.02 games)
    let mut wins = vec![vec![0.01; n_competitors]; n_competitors];
    let mut games = vec![vec![0.02; n_competitors]; n_competitors];

    for (&(i, j), history) in results {
        let (w_i, n_ij) = history.to_bt_stats();
        wins[i][j] += w_i;
        games[i][j] += n_ij;
        wins[j][i] += n_ij - w_i;
        games[j][i] += n_ij;
    }

    let total_wins: Vec<f64> = (0..n_competitors).map(|i| wins[i].iter().sum()).collect();

    // 2. Iterative MM algorithm
    let mut gamma = vec![1.0; n_competitors];
    for _ in 0..1000 {
        let mut next_gamma = vec![0.0; n_competitors];
        for i in 0..n_competitors {
            let mut denominator = 0.0;
            for j in 0..n_competitors {
                if i == j {
                    continue;
                }
                denominator += games[i][j] / (gamma[i] + gamma[j]);
            }
            next_gamma[i] = total_wins[i] / denominator;
        }

        // Normalize gamma so the anchor (index 0) stays at 1.0
        let scale = 1.0 / next_gamma[0];
        for g in &mut next_gamma {
            *g *= scale;
        }

        let mut max_diff: f64 = 0.0;
        for i in 0..n_competitors {
            max_diff = max_diff.max((next_gamma[i] - gamma[i]).abs());
        }
        gamma = next_gamma;
        if max_diff < 1e-9 {
            break;
        }
    }

    // 3. Compute Elos
    let elos: Vec<f64> = gamma
        .iter()
        .map(|&g| ANCHOR_ELO + 400.0 * g.ln() / 10.0f64.ln())
        .collect();

    // 4. Compute Uncertainty (Fisher Information)
    let mut hessian = vec![vec![0.0; n_competitors]; n_competitors];
    for i in 0..n_competitors {
        for j in 0..n_competitors {
            if i == j {
                continue;
            }
            let val = games[i][j] * gamma[i] * gamma[j] / (gamma[i] + gamma[j]).powi(2);
            hessian[i][j] = val;
            hessian[i][i] -= val;
        }
    }

    // Fixed-anchor Hessian inversion: remove row/col 0
    let mut sub_hessian = vec![vec![0.0; n_competitors - 1]; n_competitors - 1];
    for i in 1..n_competitors {
        for j in 1..n_competitors {
            sub_hessian[i - 1][j - 1] = -hessian[i][j];
        }
    }
    // Correct the diagonal for the sub-matrix (it must be positive definite)
    for (i, row) in sub_hessian.iter_mut().enumerate() {
        let mut row_sum = 0.0;
        for j in 0..n_competitors {
            if (i + 1) == j {
                continue;
            }
            row_sum +=
                games[i + 1][j] * gamma[i + 1] * gamma[j] / (gamma[i + 1] + gamma[j]).powi(2);
        }
        row[i] = row_sum;
    }

    let margins = match invert_matrix(&sub_hessian) {
        Some(inv) => {
            let mut m = vec![0.0];
            for (i, row) in inv.iter().enumerate().take(n_competitors - 1) {
                let se_beta = row[i].sqrt();
                let elo_margin = 1.96 * se_beta * (400.0 / 10.0f64.ln());
                m.push(elo_margin);
            }
            m
        }
        None => vec![50.0; n_competitors],
    };

    BtSolverResult { elos, margins }
}

// ========================================================================================
// ENGINE ORCHESTRATION
// ========================================================================================

/// Generates a single balanced opening for a given board geometry.
///
/// It uses "Dynamic Evaluative Filtering" (DEF) to find positions that are
/// not immediately won or lost and are relatively balanced (score close to 0).
fn generate_single_opening(geo: &DynamicBoardGeometry, weights: &HeuristicWeights) -> Opening {
    let mut rng = rand::rng();
    let threshold = weights.score_threat_immediate / 5;
    let mut best_opening = Opening::default();
    let mut best_score = i32::MAX;
    let mut engine = DynamicEngine::new(geo.clone(), *weights, Some(OPENING_TT_SIZE_MB));

    for _ in 0..DEF_MAX_RETRIES {
        let moves = rand::RngExt::random_range(&mut rng, OPENING_MIN_MOVES..=OPENING_MAX_MOVES);
        let mut state = DynamicBoardState::new(geo);
        let mut curr_moves = Vec::new();
        let mut curr_p = Player::Red;
        let mut valid = true;
        for _ in 0..moves {
            let playable: Vec<u32> = (0..geo.columns())
                .filter(|&c| state.get_next_bit_index(c, geo).is_some())
                .collect();
            if playable.is_empty() {
                valid = false;
                break;
            }
            let col = playable[rand::RngExt::random_range(&mut rng, 0..playable.len())];
            state = state.drop_piece(col, curr_p, geo).unwrap();
            if state.has_won(curr_p, geo) || state.is_full(geo) {
                valid = false;
                break;
            }
            curr_moves.push(col);
            curr_p = curr_p.other();
        }
        if valid {
            let score = engine.evaluate_position(&state, curr_p, DEF_EVAL_DEPTH);
            if score.abs() < threshold {
                return Opening { moves: curr_moves };
            }
            if score.abs() < best_score {
                best_score = score.abs();
                best_opening = Opening { moves: curr_moves };
            }
        }
    }
    best_opening
}

/// Plays a single game between two competitors from a given opening.
/// Returns 1.0 if `p1` wins, 0.5 for a draw, and 0.0 if `p2` wins.
fn play_single_game(
    p1: &Competitor,
    p2: &Competitor,
    p1_color: Player,
    geo: &DynamicBoardGeometry,
    opening: &Opening,
) -> f32 {
    let mut state = DynamicBoardState::new(geo);
    let mut curr_p = Player::Red;
    for &col in &opening.moves {
        state = state.drop_piece(col, curr_p, geo).unwrap();
        curr_p = curr_p.other();
    }

    let mut engine1 = DynamicEngine::new(geo.clone(), p1.weights, Some(SIM_TT_SIZE_MB));
    let mut engine2 = DynamicEngine::new(geo.clone(), p2.weights, Some(SIM_TT_SIZE_MB));

    while !state.is_full(geo) {
        let (engine, depth) = if curr_p == p1_color {
            (&mut engine1, p1.depth)
        } else {
            (&mut engine2, p2.depth)
        };
        let col = engine.find_best_move(&state, curr_p, depth, false);
        // We unwrap since the simulation loop ensures the game is not terminal.
        state = state
            .drop_piece(col.expect("No moves available"), curr_p, geo)
            .unwrap();
        if state.has_won(curr_p, geo) {
            return if curr_p == p1_color { 1.0 } else { 0.0 };
        }
        curr_p = curr_p.other();
    }
    0.5
}

/// Runs a full matchup between two competitors on a specific geometry.
///
/// Matches are played in pairs (swapping colors) until `MAX_PAIRS` is reached
/// or until the confidence interval for the Elo difference is sufficiently small.
fn run_matchup(c1: &Competitor, c2: &Competitor, geo: &DynamicBoardGeometry) -> MatchHistory {
    let mut history = MatchHistory::default();

    while history.game_pairs < MAX_PAIRS {
        let batch_results: MatchHistory = (0..BATCH_SIZE)
            .into_par_iter()
            .map(|_| {
                let opening = generate_single_opening(geo, &c1.weights);
                let mut batch_history = MatchHistory::default();
                let r1 = play_single_game(c1, c2, Player::Red, geo, &opening);
                let r2 = play_single_game(c1, c2, Player::Yellow, geo, &opening);
                batch_history.update_pair(r1, r2);
                batch_history
            })
            .reduce(MatchHistory::default, |mut a, b| {
                a.add(&b);
                a
            });

        history.add(&batch_results);

        if history.game_pairs >= MIN_PAIRS {
            let (_, margin) = history.estimate_elo_diff();
            if margin < 25.0 {
                break;
            }
        }
    }
    history
}

/// Prints a formatted Markdown table of the tournament results.
fn print_elo_report(
    competitors: &[Competitor],
    results_table: &[GeometryResult],
    global: &GeometryResult,
) {
    println!("\n## AI ELO RATING REPORT");
    let g0_cols = GEOMETRIES[0].0;
    let g0_rows = GEOMETRIES[0].1;
    let g1_cols = GEOMETRIES[1].0;
    let g1_rows = GEOMETRIES[1].1;
    let g2_cols = GEOMETRIES[2].0;
    let g2_rows = GEOMETRIES[2].1;

    println!(
        "| Rank | Competitor         | {g0_cols}x{g0_rows} Elo      | {g1_cols}x{g1_rows} Elo      | {g2_cols}x{g2_rows} Elo      | Global Avg       |"
    );
    println!(
        "|:----:|:-------------------|:-------------|:-------------|:-------------|:-----------------|"
    );

    let mut indices: Vec<usize> = (0..competitors.len()).collect();
    indices.sort_by(|&a, &b| global.elos[b].partial_cmp(&global.elos[a]).unwrap());

    for (rank_idx, &i) in indices.iter().enumerate() {
        let rank = rank_idx + 1;
        let name = &competitors[i].name;
        let mut row = format!("| {rank:^4} | {name:<18} ");

        for geo_res in results_table {
            let elo = geo_res.elos[i];
            let margin = geo_res.margins[i];
            let cell = format!("{elo:4.0} ± {margin:<3.0}");
            let _ = write!(row, "| {cell:<12} ");
        }

        let g_elo = global.elos[i];
        let g_margin = global.margins[i];
        let g_cell = format!("{g_elo:4.0} ± {g_margin:<3.0}");
        let _ = write!(row, "| **{g_cell:<12}** |");
        println!("{row}");
    }
}

// ========================================================================================
// MAIN ENTRY
// ========================================================================================

/// Initializes the tracing subscriber for logging to both file and stdout.
fn init_tracing() -> tracing_appender::non_blocking::WorkerGuard {
    let _ = std::fs::create_dir_all(LOG_DIR);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let log_file_name = format!("sim_ai_{timestamp}.log");
    let file_appender = tracing_appender::rolling::never(LOG_DIR, log_file_name);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout))
        .init();

    guard
}

/// Runs all matchups for a specific board geometry.
fn run_geometry(
    geo: &DynamicBoardGeometry,
    competitors: &[Competitor],
    global_results: &mut HashMap<(usize, usize), MatchHistory>,
) -> GeometryResult {
    let cols = geo.columns();
    let rows = geo.rows();
    tracing::info!("--- GEOMETRY: {cols}x{rows} ---");

    let mut geo_matchups: HashMap<(usize, usize), MatchHistory> = HashMap::new();
    let mut pairs = Vec::new();
    for i in 0..competitors.len() {
        for j in (i + 1)..competitors.len() {
            pairs.push((i, j));
        }
    }

    let results: Vec<((usize, usize), MatchHistory)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            tracing::info!(
                "Matchup: {} vs {}",
                competitors[i].name,
                competitors[j].name
            );
            let history = run_matchup(&competitors[i], &competitors[j], geo);
            ((i, j), history)
        })
        .collect();

    for (pair, history) in results {
        geo_matchups.insert(pair, history.clone());
        global_results
            .entry(pair)
            .and_modify(|h| h.add(&history))
            .or_insert(history);
    }

    let solver_res = solve_bradley_terry(competitors.len(), &geo_matchups);
    for (i, competitor) in competitors.iter().enumerate() {
        let name = &competitor.name;
        let elo = solver_res.elos[i];
        let margin = solver_res.margins[i];
        tracing::info!("{name:<20} : {elo:>7.1} ± {margin:<5.1} Elo");
    }

    GeometryResult {
        elos: solver_res.elos,
        margins: solver_res.margins,
    }
}

/// Entry point for the AI tournament simulation.
/// Configures competitors and runs the tournament across multiple geometries.
fn main() {
    let _guard = init_tracing();

    let weights = HeuristicWeights::default();
    let competitors = vec![
        Competitor {
            name: "Depth 1 (Anchor)".to_string(),
            depth: 1,
            weights,
        },
        Competitor {
            name: "Depth 3".to_string(),
            depth: 3,
            weights,
        },
        Competitor {
            name: "Depth 5".to_string(),
            depth: 5,
            weights,
        },
        Competitor {
            name: "Depth 7".to_string(),
            depth: 7,
            weights,
        },
        Competitor {
            name: "Depth 9".to_string(),
            depth: 9,
            weights,
        },
    ];

    tracing::info!("============================================================");
    tracing::info!("CONNECT 4 AI TOURNAMENT - GENERALIZED ELO ESTIMATION");
    tracing::info!(
        "Competitors: {} | Geometries: {}",
        competitors.len(),
        GEOMETRIES.len()
    );
    tracing::info!("============================================================\n");

    let mut results_table = Vec::new();
    let mut global_results: HashMap<(usize, usize), MatchHistory> = HashMap::new();

    for &(cols, rows) in &GEOMETRIES {
        let geo = DynamicBoardGeometry::new(cols, rows);
        let geo_res = run_geometry(&geo, &competitors, &mut global_results);
        results_table.push(geo_res);
        println!();
    }

    let global_solver_res = solve_bradley_terry(competitors.len(), &global_results);
    let global_res = GeometryResult {
        elos: global_solver_res.elos,
        margins: global_solver_res.margins,
    };

    print_elo_report(&competitors, &results_table, &global_res);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elo_math_sanity() {
        let mut history = MatchHistory::default();
        for _ in 0..10 {
            history.update_pair(1.0, 1.0);
        }
        let (diff, _) = history.estimate_elo_diff();
        assert!(diff > 300.0);

        let mut h2 = MatchHistory::default();
        for _ in 0..10 {
            h2.update_pair(1.0, 0.0);
        }
        let (diff2, _) = h2.estimate_elo_diff();
        assert!(diff2.abs() < 1.0);
    }

    #[test]
    fn test_matrix_inversion() {
        let matrix = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_matrix(&matrix).unwrap();
        // Expecting [[0.6, -0.7], [-0.2, 0.4]]
        assert!((inv[0][0] - 0.6).abs() < 1e-6);
        assert!((inv[0][1] + 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_bt_solver_logic() {
        let mut results = HashMap::new();

        // Scenario: 3 competitors
        // C0 (Anchor) vs C1: C1 wins 75% (30/40 games)
        let mut h01 = MatchHistory::default();
        for _ in 0..20 {
            h01.update_pair(0.0, 0.5);
        } // C0 scores 10, C1 scores 30
        results.insert((0, 1), h01);

        // C1 vs C2: C2 wins 75%
        let mut h12 = MatchHistory::default();
        for _ in 0..20 {
            h12.update_pair(0.0, 0.5);
        } // C1 scores 10, C2 scores 30
        results.insert((1, 2), h12);

        let solver_res = solve_bradley_terry(3, &results);

        // C0 is anchor (index 0)
        assert!((solver_res.elos[0] - ANCHOR_ELO).abs() < 0.1);
        // C1 should be significantly higher than C0
        assert!(solver_res.elos[1] > solver_res.elos[0]);
        // C2 should be significantly higher than C1
        assert!(solver_res.elos[2] > solver_res.elos[1]);

        // Check transitivity: The difference (C2-C1) should be similar to (C1-C0)
        let diff1 = solver_res.elos[1] - solver_res.elos[0];
        let diff2 = solver_res.elos[2] - solver_res.elos[1];
        assert!((diff1 - diff2).abs() < 1.0);
    }
}

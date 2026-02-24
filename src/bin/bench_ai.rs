//! # Connect 4 AI Benchmarking Tool
//!
//! This tool provides comprehensive benchmarks for the Connect 4 AI engine, including
//! search efficiency (nodes per second, branching factor) and tactical accuracy
//! through a suite of predefined puzzles. It evaluates the impact of the
//! Transposition Table on search performance across different board geometries.

use connect4::config::HeuristicWeights;
use connect4::engine::DynamicEngine;
use connect4::game::{DynamicBoardGeometry, DynamicBoardState};
use connect4::tt::TTStats;
use connect4::types::Player;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

// ========================================================================================
// BENCHMARK CONFIGURATION
// ========================================================================================

/// Search depths to evaluate during the efficiency benchmark.
const BENCH_DEPTHS: [u32; 7] = [1, 3, 5, 7, 9, 11, 13];
/// Search depth used for the tactical puzzle suite.
const TACTICAL_DEPTH: u32 = 7;
/// Number of iterations for lower search depths to ensure stable measurements.
const ITERATIONS_LOW_DEPTH: usize = 10;
/// Number of iterations for higher search depths where searches are more time-consuming.
const ITERATIONS_HIGH_DEPTH: usize = 2;
/// Directory where benchmark logs are stored.
const LOG_DIR: &str = "logs";

/// Board geometries (columns, rows) used in the efficiency benchmark.
const GEOMETRIES: [(u32, u32); 3] = [(7, 6), (8, 7), (9, 7)];

/// Transposition Table size for the benchmark.
const BENCH_TT_SIZE_MB: usize = 128;

// ========================================================================================
// DATA MODELS
// ========================================================================================

/// Represents a tactical puzzle for testing the AI's move selection accuracy.
#[derive(Debug, Clone)]
struct TacticalPuzzle {
    /// Descriptive name of the puzzle.
    name: &'static str,
    /// Number of columns in the puzzle's board geometry.
    cols: u32,
    /// Number of rows in the puzzle's board geometry.
    rows: u32,
    /// Sequence of moves to set up the puzzle position.
    moves: &'static [u32],
    /// List of column indices that are considered correct moves for the current player.
    expected_moves: &'static [u32],
}

/// Collected performance metrics for a specific search depth.
#[derive(Debug, Clone)]
struct BenchMetrics {
    /// The search depth these metrics were collected at.
    depth: u32,
    /// Total nodes visited in the baseline search (No TT).
    nodes_baseline: u64,
    /// Nodes per second in the baseline search.
    nps_baseline: f64,
    /// Effective Branching Factor in the baseline search.
    ebf_baseline: f64,
    /// Average time taken for the baseline search.
    time_baseline: Duration,
    /// Total nodes visited with the Transposition Table enabled.
    nodes_tt: u64,
    /// Nodes per second with the Transposition Table enabled.
    nps_tt: f64,
    /// Effective Branching Factor with the Transposition Table enabled.
    ebf_tt: f64,
    /// Average time taken with the Transposition Table enabled.
    time_tt: Duration,
    /// Speedup factor achieved by using the Transposition Table.
    speedup: f64,
    /// Percentage reduction in nodes visited due to the Transposition Table.
    reduction: f64,
    /// Statistics from the Transposition Table during the search.
    tt_stats: TTStats,
}

impl Default for BenchMetrics {
    /// Returns a `BenchMetrics` instance with all values initialized to zero or defaults.
    fn default() -> Self {
        Self {
            depth: 0,
            nodes_baseline: 0,
            nps_baseline: 0.0,
            ebf_baseline: 0.0,
            time_baseline: Duration::ZERO,
            nodes_tt: 0,
            nps_tt: 0.0,
            ebf_tt: 0.0,
            time_tt: Duration::ZERO,
            speedup: 0.0,
            reduction: 0.0,
            tt_stats: TTStats::default(),
        }
    }
}

impl BenchMetrics {
    /// Calculates performance metrics based on raw search data.
    ///
    /// This function computes derived metrics like Nodes Per Second (NPS),
    /// Effective Branching Factor (EBF), speedup, and node reduction percentages.
    #[allow(clippy::cast_precision_loss)]
    fn calculate(
        depth: u32,
        nodes_baseline: u64,
        duration_baseline: Duration,
        nodes_tt: u64,
        duration_tt: Duration,
        iterations: usize,
        tt_stats: TTStats,
    ) -> Self {
        let n_b = nodes_baseline / iterations as u64;
        let n_tt = nodes_tt / iterations as u64;

        let avg_secs_b = duration_baseline.as_secs_f64() / iterations as f64;
        let avg_secs_tt = duration_tt.as_secs_f64() / iterations as f64;

        let nps_b = if avg_secs_b > 0.0 {
            n_b as f64 / avg_secs_b
        } else {
            0.0
        };
        let nps_tt = if avg_secs_tt > 0.0 {
            n_tt as f64 / avg_secs_tt
        } else {
            0.0
        };

        let ebf_b = if depth > 0 {
            (n_b as f64).powf(1.0 / f64::from(depth))
        } else {
            0.0
        };
        let ebf_tt = if depth > 0 {
            (n_tt as f64).powf(1.0 / f64::from(depth))
        } else {
            0.0
        };

        let speedup = duration_baseline.as_secs_f64() / duration_tt.as_secs_f64();
        let reduction = if n_b > 0 {
            (1.0 - (n_tt as f64 / n_b as f64)) * 100.0
        } else {
            0.0
        };

        let iters_u64 = iterations as u64;
        let iters_u32 = u32::try_from(iterations).expect("iterations fits in u32");
        let avg_tt_stats = TTStats {
            lookups: tt_stats.lookups / iters_u64,
            hits_deep: tt_stats.hits_deep / iters_u64,
            hits_recent: tt_stats.hits_recent / iters_u64,
            stores: tt_stats.stores / iters_u64,
            overwrites: tt_stats.overwrites / iters_u64,
        };

        Self {
            depth,
            nodes_baseline: n_b,
            nps_baseline: nps_b,
            ebf_baseline: ebf_b,
            time_baseline: duration_baseline / iters_u32,
            nodes_tt: n_tt,
            nps_tt,
            ebf_tt,
            time_tt: duration_tt / iters_u32,
            speedup,
            reduction,
            tt_stats: avg_tt_stats,
        }
    }
}

/// Formats a `Duration` into a human-readable string (seconds, milliseconds, or microseconds).
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 0.001 {
        format!("{:.1}µs", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.1}ms", secs * 1000.0)
    } else {
        format!("{secs:.2}s")
    }
}

/// Formats a large number with thousand separators.
fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let bytes = s.as_bytes();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

// ========================================================================================
// TACTICAL SUITE
// ========================================================================================

/// A predefined suite of tactical puzzles to test AI move selection accuracy.
const TACTICAL_SUITE: [TacticalPuzzle; 6] = [
    TacticalPuzzle {
        name: "Mate-in-1 (Horizontal)",
        cols: 7,
        rows: 6,
        moves: &[3, 0, 4, 0, 5, 0],
        expected_moves: &[2, 6],
    },
    TacticalPuzzle {
        name: "Mate-in-1 (Vertical)",
        cols: 7,
        rows: 6,
        moves: &[3, 4, 3, 4, 3, 4],
        expected_moves: &[3],
    },
    TacticalPuzzle {
        name: "Mate-in-3 (Fork)",
        cols: 7,
        rows: 6,
        moves: &[2, 0, 3, 6],
        expected_moves: &[1, 4],
    },
    TacticalPuzzle {
        name: "Block Mate-in-1",
        cols: 7,
        rows: 6,
        moves: &[0, 3, 0, 4, 0],
        expected_moves: &[0],
    },
    TacticalPuzzle {
        name: "Diagonal Block",
        cols: 7,
        rows: 6,
        moves: &[0, 1, 2, 2, 3, 5, 3, 3, 4, 6, 4, 0, 4, 6],
        expected_moves: &[4],
    },
    TacticalPuzzle {
        name: "Large Board Fork",
        cols: 8,
        rows: 7,
        moves: &[3, 0, 4, 7], // Red pieces at 3,4. Yellow at 0,7.
        expected_moves: &[2, 5],
    },
];

// ========================================================================================
// ENGINE ORCHESTRATION
// ========================================================================================

/// Runs the search efficiency benchmark for a given board geometry.
///
/// This evaluates search performance at various depths, comparing a baseline
/// search (without Transposition Table) to a search with the Transposition Table enabled.
fn run_efficiency_bench(
    geo: &DynamicBoardGeometry,
    weights: &HeuristicWeights,
) -> Vec<BenchMetrics> {
    tracing::info!(
        "--- EFFICIENCY BENCHMARK ({}x{}) ---",
        geo.columns(),
        geo.rows()
    );
    let mut results = Vec::new();

    // Engines are created outside the iteration loop.
    // engine_no_tt is recreated once per depth for baseline comparison.
    // engine_tt is also created once per depth.

    for &depth in &BENCH_DEPTHS {
        let iterations = if depth > 9 {
            ITERATIONS_HIGH_DEPTH
        } else {
            ITERATIONS_LOW_DEPTH
        };
        let board_state = DynamicBoardState::new(geo);

        let mut total_nodes_baseline = 0;
        let mut total_duration_baseline = Duration::ZERO;
        let mut total_nodes_tt = 0;
        let mut total_duration_tt = Duration::ZERO;
        let mut total_tt_stats = TTStats::default();

        let mut engine_no_tt = DynamicEngine::new(geo.clone(), *weights, None);
        let mut engine_tt = DynamicEngine::new(geo.clone(), *weights, Some(BENCH_TT_SIZE_MB));

        for _ in 0..iterations {
            // Baseline (No TT)
            let start = Instant::now();
            let report =
                engine_no_tt.find_best_move_detailed(&board_state, Player::Red, depth, false);
            total_duration_baseline += start.elapsed();
            total_nodes_baseline += report.nodes;

            // TT Enabled (Fresh start per iteration for fair measurement)
            engine_tt.reset_tt();
            let start = Instant::now();
            let report = engine_tt.find_best_move_detailed(&board_state, Player::Red, depth, false);
            total_duration_tt += start.elapsed();
            total_nodes_tt += report.nodes;

            if let Some(stats) = report.tt_stats {
                total_tt_stats.lookups += stats.lookups;
                total_tt_stats.hits_deep += stats.hits_deep;
                total_tt_stats.hits_recent += stats.hits_recent;
                total_tt_stats.stores += stats.stores;
                total_tt_stats.overwrites += stats.overwrites;
            }
        }

        let metrics = BenchMetrics::calculate(
            depth,
            total_nodes_baseline,
            total_duration_baseline,
            total_nodes_tt,
            total_duration_tt,
            iterations,
            total_tt_stats,
        );

        tracing::info!(
            "Depth {:>2} | Time: {:>8} -> {:>8} | Nodes: {:>10} -> {:>10} ({:>5.1}% Reduc) | Speedup: {:>5.2}x | NPS: {:>10.0} -> {:>10.0} | EBF: {:>5.2} -> {:>5.2}",
            metrics.depth,
            format_duration(metrics.time_baseline),
            format_duration(metrics.time_tt),
            metrics.nodes_baseline,
            metrics.nodes_tt,
            metrics.reduction,
            metrics.speedup,
            metrics.nps_baseline,
            metrics.nps_tt,
            metrics.ebf_baseline,
            metrics.ebf_tt
        );

        #[allow(clippy::cast_precision_loss)]
        let hit_rate = if metrics.tt_stats.lookups > 0 {
            (metrics.tt_stats.hits_deep + metrics.tt_stats.hits_recent) as f64
                / metrics.tt_stats.lookups as f64
                * 100.0
        } else {
            0.0
        };
        tracing::info!(
            "         | TT Avg: Hit Rate: {:>5.1}% (Deep: {}, Recent: {}) | Overwrites: {}",
            hit_rate,
            metrics.tt_stats.hits_deep,
            metrics.tt_stats.hits_recent,
            metrics.tt_stats.overwrites
        );
        results.push(metrics);
    }
    results
}

/// Runs the tactical puzzle suite to evaluate search accuracy.
///
/// Returns a tuple of (puzzles passed, total puzzles).
fn run_tactical_suite(weights: &HeuristicWeights) -> (usize, usize) {
    tracing::info!("\n--- TACTICAL SUITE ---");
    let mut passed = 0;

    for puzzle in &TACTICAL_SUITE {
        let geo = DynamicBoardGeometry::new(puzzle.cols, puzzle.rows);
        let mut state = DynamicBoardState::new(&geo);
        let mut curr_p = Player::Red;
        for &m in puzzle.moves {
            state = state
                .drop_piece(m, curr_p, &geo)
                .expect("Puzzle setup failed: column full");
            curr_p = curr_p.other();
        }

        let mut engine = DynamicEngine::new(geo.clone(), *weights, Some(BENCH_TT_SIZE_MB));
        // No need to reset_tt as it's a new engine each puzzle

        let start = Instant::now();
        let report = engine.find_best_move_detailed(&state, curr_p, TACTICAL_DEPTH, false);
        let duration = start.elapsed();

        let success = if let Some(m) = report.best_move {
            puzzle.expected_moves.contains(&m)
        } else {
            false
        };
        if success {
            passed += 1;
        }

        let status = if success { "✅ PASS" } else { "❌ FAIL" };
        tracing::info!(
            "{:<25} ({:^3}x{:^1}) : {} ({:>7.2}ms) | Best: {:?} | Nodes: {}",
            puzzle.name,
            puzzle.cols,
            puzzle.rows,
            status,
            duration.as_secs_f64() * 1000.0,
            report.best_move,
            report.nodes
        );
    }
    (passed, TACTICAL_SUITE.len())
}

/// Prints a detailed Markdown report of search performance and tactical accuracy.
fn print_markdown_report(
    geo_results: Vec<((u32, u32), Vec<BenchMetrics>)>,
    tactical: (usize, usize),
) {
    println!("\n## SEARCH PERFORMANCE REPORT");

    for ((cols, rows), results) in geo_results {
        println!("\n### GEOMETRY: {cols}x{rows}");
        println!("#### Efficiency Summary");
        println!(
            "| Depth | Time (B) | Time (TT) | Nodes (B)   | Nodes (TT)  | NPS (B)     | NPS (TT)    | EBF (B) | EBF (TT) | Reduc  | Speedup  |"
        );
        println!(
            "|------:|---------:|----------:|------------:|------------:|------------:|------------:|--------:|---------:|-------:|---------:|"
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        for m in &results {
            println!(
                "| {:<5} | {:>8} | {:>9} | {:>11} | {:>11} | {:>11} | {:>11} | {:>7.2} | {:>8.2} | {:>5.1}% | {:>7.2}x |",
                m.depth,
                format_duration(m.time_baseline),
                format_duration(m.time_tt),
                format_thousands(m.nodes_baseline),
                format_thousands(m.nodes_tt),
                format_thousands(m.nps_baseline.round() as u64),
                format_thousands(m.nps_tt.round() as u64),
                m.ebf_baseline,
                m.ebf_tt,
                m.reduction,
                m.speedup
            );
        }

        println!("\n#### Transposition Table Diagnostics");
        println!("| Depth | Hit Rate | Hits (Deep) | Hits (Rec) | Overwrites |");
        println!("|------:|---------:|------------:|-----------:|-----------:|");
        for m in results {
            let total_hits = m.tt_stats.hits_deep + m.tt_stats.hits_recent;
            #[allow(clippy::cast_precision_loss)]
            let hit_rate = if m.tt_stats.lookups > 0 {
                total_hits as f64 / m.tt_stats.lookups as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "| {:<5} | {:>7.1}% | {:>11} | {:>10} | {:>10} |",
                m.depth,
                hit_rate,
                format_thousands(m.tt_stats.hits_deep),
                format_thousands(m.tt_stats.hits_recent),
                format_thousands(m.tt_stats.overwrites)
            );
        }
    }

    println!(
        "\n**Tactical Accuracy:** {}/{} puzzles passed.",
        tactical.0, tactical.1
    );
}

// ========================================================================================
// MAIN ENTRY
// ========================================================================================

/// Entry point for the AI benchmarking tool.
/// Performs a warm-up phase, runs efficiency benchmarks across multiple geometries,
/// and evaluates the tactical puzzle suite.
fn main() {
    let _ = std::fs::create_dir_all(LOG_DIR);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let log_file_name = format!("bench_ai_{timestamp}.log");
    let file_appender = tracing_appender::rolling::never(LOG_DIR, log_file_name);
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
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

    let weights = HeuristicWeights::default();

    tracing::info!("============================================================");
    tracing::info!("CONNECT 4 SEARCH PROFILER - PRECISION MEASUREMENT");
    tracing::info!(
        "Iterations: {}/{}",
        ITERATIONS_LOW_DEPTH,
        ITERATIONS_HIGH_DEPTH
    );
    tracing::info!("============================================================\n");

    // Warm-up phase
    tracing::info!("Warming up CPU caches...");
    let warm_up_geo = DynamicBoardGeometry::new(7, 6);
    let warm_up_state = DynamicBoardState::new(&warm_up_geo);
    let mut warm_up_engine = DynamicEngine::new(warm_up_geo, weights, None);
    for _ in 0..5 {
        let _ = warm_up_engine.find_best_move_detailed(&warm_up_state, Player::Red, 5, false);
    }
    tracing::info!("Warm-up complete.\n");

    let mut geo_results = Vec::new();
    for &(cols, rows) in &GEOMETRIES {
        let geo = DynamicBoardGeometry::new(cols, rows);
        let bench_results = run_efficiency_bench(&geo, &weights);
        geo_results.push(((cols, rows), bench_results));
    }

    let tactical_results = run_tactical_suite(&weights);

    print_markdown_report(geo_results, tactical_results);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        let depth = 5;
        let nodes_baseline = 10000;
        let duration_baseline = Duration::from_secs(2);
        let nodes_tt = 5000;
        let duration_tt = Duration::from_secs(1);
        let iterations = 10;
        let tt_stats = TTStats::default();

        let m = BenchMetrics::calculate(
            depth,
            nodes_baseline,
            duration_baseline,
            nodes_tt,
            duration_tt,
            iterations,
            tt_stats,
        );

        assert_eq!(m.depth, 5);
        assert_eq!(m.nodes_baseline, 1000);
        assert_eq!(m.nodes_tt, 500);
        assert_eq!(m.time_baseline, Duration::from_millis(200));
        assert_eq!(m.time_tt, Duration::from_millis(100));
        assert!((m.speedup - 2.0).abs() < 0.01);
        assert!((m.reduction - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_tactical_suite_integrity() {
        for puzzle in &TACTICAL_SUITE {
            let geo = DynamicBoardGeometry::new(puzzle.cols, puzzle.rows);
            let mut state = DynamicBoardState::new(&geo);
            for &m in puzzle.moves {
                assert!(
                    state.get_next_bit_index(m, &geo).is_some(),
                    "Invalid move in puzzle: {}",
                    puzzle.name
                );
                state = state.drop_piece(m, Player::Red, &geo).unwrap();
            }
        }
    }
}

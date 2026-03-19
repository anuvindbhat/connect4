//! # Connect 4 AI Tuner
//!
//! This tool provides a framework for optimizing the heuristic weights of the Connect 4 engine
//! using a combination of Differential Evolution (jDE), Sequential Probability Ratio Test (SPRT),
//! and Dynamic Evaluative Filtering (DEF).
//!
//! The tuner automates the process of finding optimal parameters for the engine's move selection,
//! ensuring robust performance across various board geometries and against diverse strategic opponents.

use connect4::config::HeuristicWeights;
use connect4::engine::DynamicEngine;
use connect4::game::{DynamicBoardGeometry, DynamicBoardState};
use connect4::types::Player;
use rand::RngExt;
use rand::seq::{IndexedRandom, SliceRandom};
use rayon::prelude::*;
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

// ========================================================================================
// TUNER CONFIGURATION
// ========================================================================================

/// The board geometries used during the tuning process to ensure generalizability.
const GEOMETRIES: [(u32, u32); 3] = [(7, 6), (8, 7), (9, 7)];

/// The initial size of the candidate population.
/// Population size should be around 5d to 10d where d is the number of dimensions
/// (number of parameters being tuned).
const POPULATION_START_SIZE: usize = 128;

/// The final size of the candidate population after Linear Population Size Reduction (LPSR).
const POPULATION_END_SIZE: usize = 64;

/// The total number of generations to run the tuning process.
const TOTAL_GENERATIONS: usize = 1000;

/// The maximum number of candidates allowed in the Hall of Fame (HOF).
const MAX_HOF_SIZE: usize = 10;

/// The number of opponents selected from the Hall of Fame to participate in each generation's gauntlet.
const ACTIVE_CHAMPIONS: usize = 6;

/// The percentage of the population that survives unchanged to the next generation.
const ELITE_PERCENT: usize = 10;

/// The percentage of the population replaced with random new candidates in each generation.
const FRESH_BLOOD_PERCENT: usize = 10;

/// The number of game pairs a candidate plays in Phase 1 at the start of the tuning process.
/// We scale the number of games as the population converges to distinguish
/// between increasingly similar elite candidates.
/// Try to maintain a 1:3 ratio of Phase 1 to Phase 2 games.
/// Note that due to Incremental Evaluation, Phase 2 includes the Phase 1
/// games.
const PHASE1_START_GAMES: usize = 36;

/// The number of game pairs a candidate plays in Phase 1 at the end of the tuning process.
const PHASE1_END_GAMES: usize = 180;

/// The number of game pairs a candidate plays in Phase 2 at the start of the tuning process.
const PHASE2_START_GAMES: usize = 108;

/// The number of game pairs a candidate plays in Phase 2 at the end of the tuning process.
const PHASE2_END_GAMES: usize = 540;

/// The number of heuristic weights that are subject to mutation and optimization.
const WEIGHT_FIELD_COUNT: usize = 8;

/// The minimum variance used in SPRT calculations to prevent numerical instability.
const MIN_VARIANCE: f64 = 0.01;

/// The minimum number of game pairs before SPRT can early accept or reject a candidate.
/// This is the minimum number of pairs before SPRT can early accept/reject.
/// We choose this since after 6 pairs, every champion and every geometry
/// will be tested by latin-square interleaving. Note that, we still
/// need 18 (6 * 3) pairs to cover the full cross-product.
const MIN_SPRT_PAIRS: usize = 6;

/// Elo Resolution Factor: Derived from the inverse square root law.
/// Targets an Elo difference that scales with the noise floor (Standard Error)
/// of the sample size: ELO = 350 / sqrt(N).
const ELO_RESOLUTION_FACTOR: f64 = 350.0;

/// HOF Similarity threshold (Anchored Log-Manhattan Distance).
/// A threshold of 0.50 represents a ~15% cumulative strategic shift across
/// primary tactical parameters, ensuring only functionally unique "species"
/// are preserved in the Hall of Fame.
const HOF_SIMILARITY_THRESHOLD: f32 = 0.50;

/// The maximum number of retries allowed for Dynamic Evaluative Filtering (DEF) when generating a contested opening.
const DEF_MAX_RETRIES: usize = 500;

/// The minimum number of moves in a generated opening.
/// Opening generation (Spectrum Strategy)
/// We start from 0 moves to ensure positional weights (center control, etc.) are tuned.
const OPENING_MIN_MOVES: usize = 0;

/// The maximum number of moves in a generated opening.
/// We cap at 8 moves (~20% fill) to provide complex tactical puzzles without
/// overwhelming the generator with blunders.
const OPENING_MAX_MOVES: usize = 8;

/// The probability of mutating the differential weight 'F' in jDE.
const TAU_F: f32 = 0.1;

/// The probability of mutating the crossover probability 'CR' in jDE.
const TAU_CR: f32 = 0.1;

/// The minimum value for the differential weight 'F' in jDE.
const F_MIN: f32 = 0.1;

/// The maximum value for the differential weight 'F' in jDE.
const F_MAX: f32 = 0.9;

/// The maximum batch size for parallel match processing.
/// Limits for dynamic batch sizing during match evaluation. At most
/// `MAX_BATCH_SIZE` pairs will be processed in parallel. This ensures that
/// at most `MAX_BATCH_SIZE - 1` pairs will be discarded if a candidate is
/// accepted/rejected mid-batch due to SPRT.
const MAX_BATCH_SIZE: usize = 32;

/// The number of game pairs used during the final high-precision validation phase.
const VALIDATION_PAIRS: usize = PHASE2_END_GAMES;

/// The search depth used during the final high-precision validation phase.
const VALIDATION_DEPTH: u32 = 9;

/// The directory where tuner logs are stored.
const LOG_DIR: &str = "logs";

/// The size in megabytes of the Transposition Table (TT) used during candidate evaluation.
const TUNE_TT_SIZE_MB: usize = 64;

/// The size in megabytes of the Transposition Table (TT) used during final validation.
const VALIDATION_TT_SIZE_MB: usize = 64;

/// The size in megabytes of the Transposition Table (TT) used during opening generation.
const OPENING_TT_SIZE_MB: usize = 64;

const _: () = assert!(
    POPULATION_START_SIZE >= POPULATION_END_SIZE,
    "Population size must be non-increasing to support LPSR strategy"
);
const _: () = assert!(
    PHASE1_START_GAMES <= PHASE1_END_GAMES,
    "Phase 1 game counts must be monotonic"
);
const _: () = assert!(
    PHASE2_START_GAMES <= PHASE2_END_GAMES,
    "Phase 2 game counts must be monotonic"
);
const _: () = assert!(
    PHASE1_START_GAMES < PHASE2_START_GAMES,
    "Phase 1 search horizon must be shorter than Phase 2"
);
const _: () = assert!(
    PHASE1_END_GAMES < PHASE2_END_GAMES,
    "Phase 1 search horizon must be shorter than Phase 2"
);

const _: () = assert!(
    PHASE1_START_GAMES >= ACTIVE_CHAMPIONS * GEOMETRIES.len() * 2,
    "Phase 1 must have enough games to cover each champion and geometry at least once",
);

// Ensure SPRT has enough games to reject in Phase 1.
const _: () = assert!(
    PHASE1_START_GAMES > 2 * MIN_SPRT_PAIRS,
    "Initial Phase 1 games must be sufficient for SPRT early accept/reject"
);

// Ensure a candidate plays each champion and each geometry at least once
// before SPRT can accept/reject.
const _: () = assert!(
    MIN_SPRT_PAIRS >= ACTIVE_CHAMPIONS,
    "SPRT pairs must cover all active champions"
);
const _: () = assert!(
    MIN_SPRT_PAIRS >= GEOMETRIES.len(),
    "SPRT pairs must cover all board geometries"
);

// ========================================================================================
// GENERATIONAL SCHEDULES
// ========================================================================================

/// Linear interpolation helper for generational scaling.
///
/// Returns a value between `start` and `end` based on the current `gen_idx`
/// relative to `TOTAL_GENERATIONS`.
#[allow(clippy::cast_precision_loss)]
fn generational_lerp(start: f32, end: f32, gen_idx: usize) -> f32 {
    assert!(
        gen_idx < TOTAL_GENERATIONS,
        "Generation index {gen_idx} out of bounds (max {TOTAL_GENERATIONS})"
    );
    let progress = gen_idx as f32 / (TOTAL_GENERATIONS - 1) as f32;
    start + progress * (end - start)
}

/// Linear Population Size Reduction (LPSR) schedule.
/// Shrinks the population from START to END size across the run to focus
/// compute on refining elite candidates as the search matures.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn get_population_size(gen_idx: usize) -> usize {
    generational_lerp(
        POPULATION_START_SIZE as f32,
        POPULATION_END_SIZE as f32,
        gen_idx,
    )
    .round() as usize
}

/// Tiered Depth Schedule.
/// Increases search depth at specific milestones to prevent weights from
/// overfitting to shallow search horizons.
fn get_depth(gen_idx: usize) -> u32 {
    assert!(
        gen_idx < TOTAL_GENERATIONS,
        "Generation index {gen_idx} out of bounds (max {TOTAL_GENERATIONS})"
    );
    if gen_idx < (TOTAL_GENERATIONS * 10) / 100 {
        3
    } else if gen_idx < (TOTAL_GENERATIONS * 30) / 100 {
        5
    } else if gen_idx < (TOTAL_GENERATIONS * 75) / 100 {
        7
    } else {
        9
    }
}

/// Dynamic Phase Scaling: precision increases as optimization matures.
///
/// Returns the current game limits for Phase 1 and Phase 2 based on generational progress.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn get_phase_limits(gen_idx: usize) -> (usize, usize) {
    let p1 = generational_lerp(PHASE1_START_GAMES as f32, PHASE1_END_GAMES as f32, gen_idx)
        as usize
        & !1;
    let p2 = generational_lerp(PHASE2_START_GAMES as f32, PHASE2_END_GAMES as f32, gen_idx)
        as usize
        & !1;
    (p1, p2)
}

// ========================================================================================
// DATA MODELS
// ========================================================================================

/// A high-precision floating-point representation of the engine's heuristic weights.
///
/// This representation is used during the optimization process to allow for fine-grained
/// adjustments and smooth convergence before being rounded to integer values for engine use.
#[derive(Clone, Debug, Copy)]
struct FloatWeights {
    /// Score for potential future threats.
    score_threat_future: f32,
    /// Score for immediate setups.
    score_setup_immediate: f32,
    /// Score for a line of three pieces.
    score_three: f32,
    /// Score for a line of two pieces.
    score_two: f32,
    /// Weight assigned to the number of potential winning windows passing through a cell.
    weight_potential_window: f32,
    /// Weight assigned to the core (central) columns of the board.
    weight_core: f32,
    /// Weight assigned to the inner columns (adjacent to core).
    weight_inner: f32,
    /// Weight assigned to the outer columns (edges).
    weight_outer: f32,
}

impl FloatWeights {
    /// Returns labels for each heuristic weight field.
    fn labels() -> [&'static str; WEIGHT_FIELD_COUNT] {
        [
            "threat_fut",
            "setup_imm",
            "three",
            "two",
            "mobility",
            "core",
            "inner",
            "outer",
        ]
    }

    /// Creates a `FloatWeights` instance from the standard `HeuristicWeights`.
    #[allow(clippy::cast_precision_loss)]
    fn from_heuristic(w: HeuristicWeights) -> Self {
        Self {
            score_threat_future: w.score_threat_future as f32,
            score_setup_immediate: w.score_setup_immediate as f32,
            score_three: w.score_three as f32,
            score_two: w.score_two as f32,
            weight_potential_window: w.weight_potential_window as f32,
            weight_core: w.weight_core as f32,
            weight_inner: w.weight_inner as f32,
            weight_outer: w.weight_outer as f32,
        }
    }

    /// Converts high-precision float weights to the engine's integer format.
    /// Anchors `score_fork_immediate` and `score_threat_immediate` to the library's defaults to
    /// maintain scale.
    #[allow(clippy::cast_possible_truncation)]
    fn to_heuristic(self) -> HeuristicWeights {
        let defaults = HeuristicWeights::default();
        HeuristicWeights {
            score_fork_immediate: defaults.score_fork_immediate, // Anchored
            score_threat_immediate: defaults.score_threat_immediate, // Anchored
            score_threat_future: self.score_threat_future.round() as i32,
            score_setup_immediate: self.score_setup_immediate.round() as i32,
            score_three: self.score_three.round() as i32,
            score_two: self.score_two.round() as i32,
            weight_potential_window: self.weight_potential_window.round() as i32,
            weight_core: self.weight_core.round() as i32,
            weight_inner: self.weight_inner.round() as i32,
            weight_outer: self.weight_outer.round() as i32,
            score_win: defaults.score_win,
        }
    }

    /// Returns mutable references to fields participating in DE mutation.
    fn fields_mut(&mut self) -> [&mut f32; WEIGHT_FIELD_COUNT] {
        [
            &mut self.score_threat_future,
            &mut self.score_setup_immediate,
            &mut self.score_three,
            &mut self.score_two,
            &mut self.weight_potential_window,
            &mut self.weight_core,
            &mut self.weight_inner,
            &mut self.weight_outer,
        ]
    }

    /// Returns values of fields participating in DE mutation.
    fn fields(self) -> [f32; WEIGHT_FIELD_COUNT] {
        [
            self.score_threat_future,
            self.score_setup_immediate,
            self.score_three,
            self.score_two,
            self.weight_potential_window,
            self.weight_core,
            self.weight_inner,
            self.weight_outer,
        ]
    }

    /// Calculates the Anchored Log-Manhattan (ALM) Distance between two weight sets.
    ///
    /// In a deterministic search engine, absolute values matter less than the
    /// 'Strategic Exchange Rate' relative to the fixed tactical anchors (Triples/Wins).
    /// ALM captures percentage shifts rather than absolute shifts, ensuring that
    /// small positional levers are weighted appropriately against massive tactical ones.
    fn distance(&self, other: &Self) -> f32 {
        // Sensitivity Weights (S_i)
        // Calibrated based on SD/mean of population observed during tuning.
        // Low SD/mean implies more sensitivity.
        // These weights reflect how likely a change in this parameter is to alter
        // the engine's primary move selection.
        let sensitivity: [f32; WEIGHT_FIELD_COUNT] = [
            0.9, // score_threat_future
            1.0, // score_setup_immediate
            0.7, // score_three
            0.6, // score_two
            0.5, // weight_potential_window
            1.1, // weight_core
            0.8, // weight_inner
            0.6, // weight_outer
        ];

        let s_fields = self.fields();
        let o_fields = other.fields();
        let mut sum_dist = 0.0;

        for i in 0..WEIGHT_FIELD_COUNT {
            // Log-transform (L1 Manhattan) focuses on the "Order of Magnitude" shift.
            // We use ln(w + 1.0) to handle weights that can be zero and to treat
            // the transition from 0 -> 1 as a significant behavioral event.
            let v1 = (s_fields[i] + 1.0).ln();
            let v2 = (o_fields[i] + 1.0).ln();
            sum_dist += sensitivity[i] * (v1 - v2).abs();
        }

        sum_dist
    }
}

/// Represents a sequence of moves that define the starting state of a game.
#[derive(Clone, Debug)]
struct Opening {
    /// The sequence of column indices where pieces are dropped.
    moves: Vec<u32>,
}

/// A specific game scenario used to test a candidate against an opponent.
#[derive(Clone, Debug)]
struct Challenge {
    /// The opening sequence of moves.
    opening: Opening,
    /// The index of the opponent in the current gauntlet.
    champ_idx: usize,
    /// The index of the board geometry to use.
    geo_idx: usize,
}

/// The status of a candidate's evaluation in the current gauntlet.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
enum MatchStatus {
    /// Evaluation is ongoing.
    #[default]
    Active,
    /// Candidate has been statistically accepted.
    Accepted,
    /// Candidate has been statistically rejected.
    Rejected,
}

/// Describes the action taken when updating the Hall of Fame with a new candidate.
#[derive(Clone, Debug, Copy)]
enum HofAction {
    /// No change was made to the Hall of Fame.
    None,
    /// The candidate was added as a new member.
    Added,
    /// The candidate replaced an existing member at the specified index.
    Replaced(usize),
}

/// Tracks the results of game pairs played by a candidate.
#[derive(Clone, Debug, Default)]
struct MatchHistory {
    /// Pentanomial buckets for pair outcomes:
    /// 0: LL (0.0), 1: LD (0.5), 2: DD/WL (1.0), 3: WD (1.5), 4: WW (2.0)
    pair_counts: [usize; 5],
    /// The total number of game pairs played.
    game_pairs: usize,
    /// The current statistical status of the match evaluation.
    status: MatchStatus,
}

impl MatchHistory {
    /// Updates history with a game pair result.
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

    /// Total points earned (absolute).
    #[allow(clippy::cast_precision_loss)]
    fn score(&self) -> f32 {
        self.pair_counts[1] as f32 * 0.5
            + self.pair_counts[2] as f32 * 1.0
            + self.pair_counts[3] as f32 * 1.5
            + self.pair_counts[4] as f32 * 2.0
    }
}

/// A candidate set of heuristic weights being evaluated by the tuner.
#[derive(Clone, Debug)]
struct Candidate {
    /// The heuristic weights for this candidate.
    weights: FloatWeights,
    /// The unique index of this candidate in the population.
    index: usize,
    /// Differential Weight (jDE) used for mutation.
    f: f32,
    /// Crossover Probability (jDE) used for mutation.
    cr: f32,
    /// The match history for this candidate in the current generation.
    match_history: MatchHistory,
}

impl Candidate {
    /// Creates a new candidate with randomized weights.
    fn new_random(index: usize) -> Self {
        let mut rng = rand::rng();
        #[allow(clippy::cast_precision_loss)]
        let anchor = HeuristicWeights::default().score_threat_immediate as f32;
        let (threat_fut_lo, threat_fut_hi) = (0.0, anchor);
        let (setup_imm_lo, setup_imm_hi) = (0.0, anchor);
        let (three_lo, three_hi) = (0.0, anchor);
        let (two_lo, two_hi) = (0.0, anchor / 2.0);
        let (mobility_lo, mobility_hi) = (0.0, anchor / 2.0);
        let (core_lo, core_hi) = (0.0, anchor);
        let (inner_lo, inner_hi) = (0.0, anchor / 2.0);
        let (outer_lo, outer_hi) = (0.0, anchor / 5.0);
        let w = FloatWeights {
            score_threat_future: rng.random_range(threat_fut_lo..threat_fut_hi),
            score_setup_immediate: rng.random_range(setup_imm_lo..setup_imm_hi),
            score_three: rng.random_range(three_lo..three_hi),
            score_two: rng.random_range(two_lo..two_hi),
            weight_potential_window: rng.random_range(mobility_lo..mobility_hi),
            weight_core: rng.random_range(core_lo..core_hi),
            weight_inner: rng.random_range(inner_lo..inner_hi),
            weight_outer: rng.random_range(outer_lo..outer_hi),
        };

        let mut c = Self {
            weights: w,
            index,
            f: 0.5,
            cr: 0.9,
            match_history: MatchHistory::default(),
        };
        c.clamp();
        c
    }

    /// Enforces strict logical hierarchies between related weights.
    fn clamp(&mut self) {
        let w = &mut self.weights;
        let defaults = HeuristicWeights::default();
        #[allow(clippy::cast_precision_loss)]
        // `score_threat_immediate` is the anchor to which all other scores
        // are callibrated.
        let threat_imm = defaults.score_threat_immediate as f32;
        // Note: `score_fork_immediate` and `score_threat_immediate` are pinned to defaults in
        // `to_heuristic`.

        // Tactical Hierarchies: Immediate Threat > Future Threat > Immediate Setup
        w.score_threat_future = w.score_threat_future.clamp(0.0, threat_imm);
        w.score_setup_immediate = w.score_setup_immediate.clamp(0.0, w.score_threat_future);

        // Structural Hierarchies: Three > Two
        w.score_three = w.score_three.clamp(0.0, threat_imm);
        w.score_two = w.score_two.clamp(0.0, w.score_three);

        // Mobility
        w.weight_potential_window = w.weight_potential_window.max(0.0);

        // Positional Hierarchies: Center > Inner > Outer
        w.weight_core = w.weight_core.max(0.0);
        w.weight_inner = w.weight_inner.clamp(0.0, w.weight_core);
        w.weight_outer = w.weight_outer.clamp(0.0, w.weight_inner);
    }

    /// Mutates jDE parameters (f and cr) for self-adaptive evolution.
    fn mutate_params(&mut self) {
        let mut rng = rand::rng();
        if rng.random_range(0.0..1.0) < TAU_F {
            self.f = F_MIN + rng.random_range(0.0..1.0) * F_MAX;
        }
        if rng.random_range(0.0..1.0) < TAU_CR {
            self.cr = rng.random_range(0.0..1.0);
        }
    }

    /// Calculates the average points earned per game played.
    #[allow(clippy::cast_precision_loss)]
    fn avg_score(&self) -> f32 {
        if self.match_history.game_pairs == 0 {
            return 0.0;
        }
        // Normalized to per-game score (0.0 to 1.0)
        self.match_history.score() / (self.match_history.game_pairs as f32 * 2.0)
    }

    /// Calculates a hierarchical fitness score for selection.
    /// Priority: Accepted > Active > Rejected.
    /// Within categories, uses `avg_score`.
    fn fitness(&self) -> f32 {
        let base = match self.match_history.status {
            MatchStatus::Accepted => 2.0,
            MatchStatus::Active => 1.0,
            MatchStatus::Rejected => 0.0,
        };
        base + self.avg_score()
    }
}

// ========================================================================================
// STATISTICAL RIGOR (SPRT Engine)
// ========================================================================================

/// Parameters for the Sequential Probability Ratio Test (SPRT).
#[derive(Clone, Copy)]
struct SprtParams {
    /// The probability of a Type I error (false positive).
    alpha: f64,
    /// The probability of a Type II error (false negative).
    beta: f64,
    /// The null hypothesis Elo difference (baseline).
    elo0: f64,
    /// The alternative hypothesis Elo difference (target improvement).
    elo1: f64,
}

/// The Sequential Probability Ratio Test (SPRT) engine.
struct Sprt {
    /// The lower decision boundary.
    la: f64,
    /// The upper decision boundary.
    lb: f64,
    /// The midpoint probability between H0 and H1.
    p_mid: f64,
    /// The difference in probabilities between H1 and H0.
    delta: f64,
}

impl Sprt {
    /// Creates a new SPRT engine with the given parameters.
    fn new(params: SprtParams) -> Self {
        let p = |elo: f64| 1.0 / (1.0 + 10.0f64.powf(-elo / 400.0));
        let p0 = p(params.elo0);
        let p1 = p(params.elo1);

        Self {
            la: (params.beta / (1.0 - params.alpha)).ln(),
            lb: ((1.0 - params.beta) / params.alpha).ln(),
            p_mid: p0.midpoint(p1),
            delta: p1 - p0,
        }
    }

    /// Sequential Probability Ratio Test using Elo-based LLR on Pair Outcomes.
    /// This Gaussian approximation uses variance derived from the Pentanomial distribution.
    #[allow(clippy::cast_precision_loss)]
    fn evaluate(&self, history: &mut MatchHistory) {
        if history.game_pairs < MIN_SPRT_PAIRS || history.status != MatchStatus::Active {
            return;
        }

        let n = history.game_pairs as f64;
        let score = f64::from(history.score());
        let mu = score / (2.0 * n); // Normalized to 0..1 scale for Elo math

        // Variance of normalized pair scores: X' in {0.0, 0.25, 0.5, 0.75, 1.0}
        // We use Bessel-corrected sample variance for small N stability.
        let mut sum_sq_diff = 0.0;
        let outcomes: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        for (i, &val) in outcomes.iter().enumerate() {
            sum_sq_diff += history.pair_counts[i] as f64 * (val - mu).powi(2);
        }
        let sigma2 = (sum_sq_diff / (n - 1.0)).max(MIN_VARIANCE);

        // Log-Likelihood Ratio (Simplified for small Elo differences)
        let llr = n * (mu - self.p_mid) * self.delta / sigma2;

        if llr <= self.la {
            history.status = MatchStatus::Rejected;
        } else if llr >= self.lb {
            history.status = MatchStatus::Accepted;
        }
    }
}

// ========================================================================================
// THE TUNER ENGINE
// ========================================================================================

/// Initializes structured logging with both stdout and file appenders.
///
/// Returns a `WorkerGuard` that must be kept alive to ensure all logs are flushed.
fn init_tracing() -> tracing_appender::non_blocking::WorkerGuard {
    let _ = std::fs::create_dir_all(LOG_DIR);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let log_file_name = format!("tune-ai-{timestamp}.log");
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

/// The core tuning engine that manages the population, Hall of Fame, and generational loop.
struct Tuner {
    /// The board geometries used for evaluation.
    geometries: Vec<DynamicBoardGeometry>,
    /// The Hall of Fame containing the best candidates discovered so far.
    hall_of_fame: Vec<FloatWeights>,
    /// The leader weights from the previous generation for delta tracking.
    prev_leader: Option<FloatWeights>,
}

/// Parameters for population evaluation to avoid deep argument lists.
struct EvalParams<'a> {
    /// The search depth for games.
    depth: u32,
    /// The maximum number of game pairs to play per candidate.
    match_limit: usize,
    /// The parameters for SPRT.
    sprt_params: SprtParams,
    /// The gauntlet of challenges for the current generation.
    challenges: &'a [Challenge],
    /// The set of opponents from the Hall of Fame.
    opponents: &'a [FloatWeights],
}

impl Tuner {
    /// Creates a new `Tuner` with default geometries and the baseline heuristic in the Hall of Fame.
    fn new() -> Self {
        Self {
            geometries: GEOMETRIES
                .iter()
                .map(|&(c, r)| DynamicBoardGeometry::new(c, r))
                .collect(),
            hall_of_fame: vec![FloatWeights::from_heuristic(HeuristicWeights::default())],
            prev_leader: None,
        }
    }

    /// Initializes the starting population with random candidates.
    fn init_population() -> Vec<Candidate> {
        (0..POPULATION_START_SIZE)
            .map(Candidate::new_random)
            .collect()
    }

    /// Executes the full generational tuning loop.
    fn run(&mut self, population: &mut Vec<Candidate>) {
        for gen_idx in 0..TOTAL_GENERATIONS {
            self.run_generation(gen_idx, population);
        }
    }

    /// Performs a high-precision, parallelized cross-geometry validation.
    /// Returns the average win rate against the default heuristic.
    #[allow(clippy::cast_precision_loss)]
    fn final_validation(&self, best: &HeuristicWeights) -> f32 {
        tracing::info!("Performing Final Cross-Geometry Validation...");

        let geometries_results: Vec<f32> = self
            .geometries
            .par_iter()
            .map(|geo| {
                let total_score: f32 = (0..VALIDATION_PAIRS)
                    .into_par_iter()
                    .map(|_| {
                        // Use a neutral opening for a fair comparison
                        let opening = generate_single_opening(
                            geo,
                            &HeuristicWeights::default(),
                            VALIDATION_DEPTH,
                        );

                        // Play Red/Yellow pair in parallel to maximize core utilization
                        let (r1, r2) = rayon::join(
                            || {
                                play_single_game(
                                    *best,
                                    HeuristicWeights::default(),
                                    geo,
                                    VALIDATION_DEPTH,
                                    Player::Red,
                                    false,
                                    &opening,
                                    VALIDATION_TT_SIZE_MB,
                                )
                            },
                            || {
                                play_single_game(
                                    *best,
                                    HeuristicWeights::default(),
                                    geo,
                                    VALIDATION_DEPTH,
                                    Player::Yellow,
                                    false,
                                    &opening,
                                    VALIDATION_TT_SIZE_MB,
                                )
                            },
                        );
                        r1 + r2
                    })
                    .sum();

                (total_score / (VALIDATION_PAIRS * 2) as f32) * 100.0
            })
            .collect();

        geometries_results.iter().sum::<f32>() / self.geometries.len() as f32
    }

    /// Orchestrates a single generational cycle of the optimization algorithm.
    ///
    /// This includes scheduling, opponent selection, challenge generation,
    /// dual-phase evaluation (gauntlet), Hall of Fame updates, and evolution.
    fn run_generation(&mut self, gen_idx: usize, population: &mut Vec<Candidate>) {
        let start = Instant::now();

        // 1. Scheduling: milestone-based depth and limit retrieval.
        let (depth, p1_limit, p2_limit) = Tuner::get_schedule(gen_idx);

        // 2. Opponent Selection: stochastic HOF selection with baseline protection.
        let opponents = self.select_opponents();

        // 3. Challenge Generation: creating a balanced opening gauntlet for this generation.
        let challenges = self.generate_challenges(p2_limit / 2, &opponents, depth);

        // 4. Phase 1: Broad statistical filtering (Dynamic Elo scaling).
        // We test with loose SPRT bounds to quickly accept/reject
        // candidates with the relatively small Phase 1 sample.
        self.run_gauntlet_phase(population, depth, p1_limit, 0.1, &challenges, &opponents);

        // 5. Candidate Promotion: we rank candidates based on their
        // SPRT fitness (highest fitness first).
        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        let survivor_count = population.len() / 2;
        {
            // By utilizing a mutable slice, we modify original candidates directly
            // and eliminate the need for redundant clones or manual synchronization loops.
            let survivors = &mut population[..survivor_count];

            // Reset match status for survivors to allow Phase 2 deep validation.
            // This ensures all survivors are evaluated against the stricter
            // statistical thresholds of Phase 2 (incremental evaluation) while maintaning
            // their match history.
            for s in survivors.iter_mut() {
                s.match_history.status = MatchStatus::Active;
            }

            // 6. Phase 2: Deep statistical validation for the survivors.
            self.run_gauntlet_phase(survivors, depth, p2_limit, 0.05, &challenges, &opponents);
        }

        // Final sort to identify the generation leader and prepare for evolution.
        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        let best = population[0].clone();

        // 7. Update Hall of Fame with generation leader.
        let hof_action = self.update_hof(best.weights);

        // 8. Logging and Reporting.
        // We report BEFORE evolution to ensure statistics (Acc/Rej/Eff)
        // reflect the generation that just completed.
        self.report_generation(
            gen_idx,
            depth,
            p1_limit,
            p2_limit,
            population,
            survivor_count,
            &best,
            start.elapsed(),
            hof_action,
        );

        // 9. Evolution: Generate the next population for the subsequent generation.
        if gen_idx + 1 < TOTAL_GENERATIONS {
            Tuner::evolve_population(population, &best, gen_idx + 1);
        }
    }

    /// Calculates the search depth and precision limits based on generational progress.
    fn get_schedule(gen_idx: usize) -> (u32, usize, usize) {
        let depth = get_depth(gen_idx);
        let (p1, p2) = get_phase_limits(gen_idx);
        (depth, p1, p2)
    }

    /// Selects a diverse set of opponents from the Hall of Fame, always including the baseline gatekeeper.
    fn select_opponents(&self) -> Vec<FloatWeights> {
        let mut rng = rand::rng();
        let mut selected = Vec::with_capacity(ACTIVE_CHAMPIONS);

        // Always include the baseline gatekeeper (HOF[0]).
        selected.push(self.hall_of_fame[0]);

        // Select the remaining without replacement to maximize diversity.
        let mut indices: Vec<usize> = (1..self.hall_of_fame.len()).collect();
        indices.shuffle(&mut rng);
        for &idx in &indices {
            if selected.len() >= ACTIVE_CHAMPIONS {
                break;
            }
            selected.push(self.hall_of_fame[idx]);
        }

        // Fallback if HOF is small (select randomly with replacement).
        while selected.len() < ACTIVE_CHAMPIONS {
            selected.push(*self.hall_of_fame.choose(&mut rng).unwrap());
        }
        selected
    }

    /// Executes a gauntlet evaluation phase with dynamic Elo resolution for the provided candidates.
    fn run_gauntlet_phase(
        &self,
        candidates: &mut [Candidate],
        depth: u32,
        match_limit: usize,
        alpha_beta_prob: f64,
        challenges: &[Challenge],
        opponents: &[FloatWeights],
    ) {
        #[allow(clippy::cast_precision_loss)]
        let elo_res = ELO_RESOLUTION_FACTOR / (match_limit as f64).sqrt();
        let params = SprtParams {
            alpha: alpha_beta_prob,
            beta: alpha_beta_prob,
            elo0: 0.0,
            elo1: elo_res,
        };
        self.evaluate_population(
            candidates,
            depth,
            match_limit,
            params,
            challenges,
            opponents,
        );
    }

    /// Centralizes logging and generation reporting.
    ///
    /// This method generates detailed statistics about the current generation, including
    /// population diversity, SPRT efficiency, and parameter distributions.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::cast_precision_loss
    )]
    fn report_generation(
        &mut self,
        gen_idx: usize,
        depth: u32,
        p1_limit: usize,
        p2_limit: usize,
        population: &[Candidate],
        survivor_count: usize,
        best: &Candidate,
        duration: std::time::Duration,
        hof_action: HofAction,
    ) {
        assert!(
            population
                .windows(2)
                .all(|w| w[0].fitness() >= w[1].fitness()),
            "Population must be sorted by fitness before evolution (descending)"
        );

        let p1_elo = ELO_RESOLUTION_FACTOR / (p1_limit as f64).sqrt();
        let p2_elo = ELO_RESOLUTION_FACTOR / (p2_limit as f64).sqrt();

        // 1. Meta Statistics (Pop, Games, SPRT, jDE)
        let mut acc = 0;
        let mut rej = 0;
        let mut act = 0;
        let mut sum_f = 0.0;
        let mut sum_cr = 0.0;

        for c in population {
            match c.match_history.status {
                MatchStatus::Accepted => acc += 1,
                MatchStatus::Rejected => rej += 1,
                MatchStatus::Active => act += 1,
            }
            sum_f += c.f;
            sum_cr += c.cr;
        }

        let avg_f = sum_f / population.len() as f32;
        let avg_cr = sum_cr / population.len() as f32;

        // 2. Leader & Diversity
        let leader_delta = if let Some(prev) = self.prev_leader {
            best.weights.distance(&prev)
        } else {
            0.0
        };
        self.prev_leader = Some(best.weights);

        let sum_div: f32 = population
            .iter()
            .map(|c| c.weights.distance(&best.weights))
            .sum();
        let avg_div = sum_div / population.len() as f32;

        // 3. HOF Diversity
        let mut hof_dist_sum = 0.0;
        let mut hof_dist_count = 0_usize;
        for i in 0..self.hall_of_fame.len() {
            for j in i + 1..self.hall_of_fame.len() {
                hof_dist_sum += self.hall_of_fame[i].distance(&self.hall_of_fame[j]);
                hof_dist_count += 1;
            }
        }
        let hof_div = if hof_dist_count > 0 {
            hof_dist_sum / hof_dist_count as f32
        } else {
            0.0
        };

        // 4. SPRT Efficiency
        let total_pairs: usize = population.iter().map(|c| c.match_history.game_pairs).sum();
        let theoretical_max_pairs = (population.len() * (p1_limit / 2))
            + (survivor_count * ((p2_limit.saturating_sub(p1_limit)) / 2));
        let reduction = if theoretical_max_pairs > 0 {
            // Efficiency as "Savings": How much compute did we avoid?
            (1.0 - (total_pairs as f64 / theoretical_max_pairs as f64)) * 100.0
        } else {
            0.0
        };

        // 5. Build Parameter Distribution Tables
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let param_tables = [100.0, 100.0 - FRESH_BLOOD_PERCENT as f32, 50.0]
            .iter()
            .map(|&pct| {
                let count = (population.len() as f32 * (pct / 100.0))
                    .round()
                    .clamp(0.0, population.len() as f32) as usize;
                let subset = &population[..count];
                (pct, Self::build_param_table(subset))
            })
            .collect::<Vec<_>>();

        // Formatting Helper
        let status_str = format!("{acc} Acc / {rej} Rej / {act} Act");
        let leader_status = if leader_delta < 0.1 {
            "Stable"
        } else {
            "Active"
        };
        let div_status = if avg_div < 0.2 {
            "Collapsed"
        } else {
            "Healthy"
        };
        let hof_status = if hof_div < 0.5 {
            "Converged"
        } else {
            "Diverse"
        };
        let hof_action_str = match hof_action {
            HofAction::None => "None".to_string(),
            HofAction::Added => "Added".to_string(),
            HofAction::Replaced(idx) => format!("Replaced({idx})"),
        };

        let gi = gen_idx + 1;
        tracing::info!(
            "Gen {gi:>3}/{TOTAL_GENERATIONS} | Depth {depth} | Pop {:<3} | Games {p1_limit}/{p2_limit} | Elo Res {:.1}/{:.1}",
            population.len(),
            p1_elo,
            p2_elo,
        );
        tracing::info!(
            "Leader: Avg Pts {:.3} | Delta {:.2} ({leader_status}) | Div {:.2} ({div_status})",
            best.avg_score(),
            leader_delta,
            avg_div,
        );
        tracing::info!(
            "Stats:  {status_str:<18} | Reduc {reduction:>5.1}% | jDE F={avg_f:.2}, Cr={avg_cr:.2}",
        );
        tracing::info!(
            "HOF:    Size {:<2} | Action {:<11} | Pairwise Div {hof_div:.2} ({hof_status})",
            self.hall_of_fame.len(),
            hof_action_str,
        );
        for (pct, param_table) in param_tables {
            tracing::info!(
                "Top {pct:.0}% Parameter Distribution [Mean \u{00B1} SD (Min..Max)]:\n{param_table}"
            );
        }
        tracing::info!("Time: {duration:?} | Weights: {:?}\n", best.weights);
    }

    /// Builds a formatted table of parameter statistics (mean, std dev, min, max)
    /// for a given population subset.
    fn build_param_table(population: &[Candidate]) -> String {
        let labels = FloatWeights::labels();
        let mut pop_stats = String::new();
        for (i, &label) in labels.iter().enumerate() {
            let vals: Vec<f32> = population.iter().map(|c| c.weights.fields()[i]).collect();
            #[allow(clippy::cast_precision_loss)]
            let n = vals.len() as f32;
            let mean = vals.iter().sum::<f32>() / n;
            let std_dev = (vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
            let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
            let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            if i > 0 && i % 2 == 0 {
                pop_stats.push('\n');
            } else if i > 0 {
                pop_stats.push_str(" | ");
            }
            write!(
                pop_stats,
                "  {label:<11}: {mean:>8.1} \u{00B1} {std_dev:>6.1} ({min:>6.0}..{max:>6.0})"
            )
            .unwrap();
        }
        pop_stats
    }

    /// Triggers the evolutionary step to produce the population for the next generation.
    fn evolve_population(
        population: &mut Vec<Candidate>,
        best_gen: &Candidate,
        next_gen_idx: usize,
    ) {
        let next_pop_size = get_population_size(next_gen_idx);
        Self::evolve(population, best_gen, next_pop_size);
    }

    /// Updates the HOF using similarity-based replacement to maintain gauntlet diversity.
    /// If the HOF is full, it replaces the member most strategistally similar to the
    /// new candidate to ensure we refine established "species" rather than
    /// destructively deleting unique ones.
    fn update_hof(&mut self, best_weights: FloatWeights) -> HofAction {
        let mut closest_idx = None;
        let mut min_dist = f32::MAX;

        // Find the most similar strategist in the HOF (ignoring baseline gatekeeper at 0)
        for (i, member) in self.hall_of_fame.iter().enumerate().skip(1) {
            let dist = best_weights.distance(member);
            if dist < min_dist {
                min_dist = dist;
                closest_idx = Some(i);
            }
        }

        if let Some(idx) = closest_idx {
            if min_dist < HOF_SIMILARITY_THRESHOLD || self.hall_of_fame.len() >= MAX_HOF_SIZE {
                // Refine existing tactical niche
                self.hall_of_fame[idx] = best_weights;
                HofAction::Replaced(idx)
            } else {
                // Add as a new species
                self.hall_of_fame.push(best_weights);
                HofAction::Added
            }
        } else if self.hall_of_fame.len() < MAX_HOF_SIZE {
            // HOF only has gatekeeper, add second member
            self.hall_of_fame.push(best_weights);
            HofAction::Added
        } else {
            HofAction::None
        }
    }

    /// Evaluates a population of candidates against a gauntlet of opponents using SPRT.
    fn evaluate_population(
        &self,
        population: &mut [Candidate],
        depth: u32,
        match_limit: usize,
        sprt_params: SprtParams,
        challenges: &[Challenge],
        opponents: &[FloatWeights],
    ) {
        assert_eq!(
            match_limit % 2,
            0,
            "Match limit {match_limit} must be even to ensure color balance"
        );
        let active_count = AtomicUsize::new(population.len());
        let params = EvalParams {
            depth,
            match_limit,
            sprt_params,
            challenges,
            opponents,
        };

        population.par_iter_mut().for_each(|candidate| {
            self.evaluate_candidate(candidate, &active_count, &params);
        });
    }

    /// Performs the incremental SPRT evaluation for a single candidate.
    fn evaluate_candidate(
        &self,
        candidate: &mut Candidate,
        active_count: &AtomicUsize,
        params: &EvalParams,
    ) {
        let sprt = Sprt::new(params.sprt_params);
        let candidate_weights = candidate.weights.to_heuristic();
        let total_threads = rayon::current_num_threads();

        // Unique fixed offset per candidate to prevent shared-opening bias.
        // Using a multiple of the gauntlet period (Champs x Geos) ensures
        // that every candidate starts their SPRT evaluation on a perfectly
        // balanced block of games.
        // Note that this also handles incremental evaluation from Phase 1
        // to Phase 2 since we correctly resume from where we
        // left off in Phase 1.
        let period = params.opponents.len() * self.geometries.len();
        let challenge_offset = candidate.index * period;

        // Incremental evaluation: picks up where previous phase left off
        while candidate.match_history.game_pairs * 2 < params.match_limit
            && candidate.match_history.status == MatchStatus::Active
        {
            let active = active_count.load(Ordering::Relaxed).max(1);
            // Dynamically scale batch size based on available resources.
            // We target enough pairs to fill the cores assigned to this candidate.
            // We want to keep batch size small when possible to minimize
            // discarded work due to SPRT.
            let batch_size = (total_threads / active).clamp(1, MAX_BATCH_SIZE);

            // Limit batch size to remaining match limit
            let remaining_pairs = params
                .match_limit
                .div_ceil(2)
                .saturating_sub(candidate.match_history.game_pairs);
            let current_batch = batch_size.min(remaining_pairs).max(1);

            // Execute batch of game pairs in parallel.
            let results: Vec<(f32, f32)> = (0..current_batch)
                .into_par_iter()
                .map(|offset| {
                    // Apply seed shuffling offset
                    let pair_idx = candidate.match_history.game_pairs + offset;
                    let challenge_idx = (pair_idx + challenge_offset) % params.challenges.len();
                    let challenge = &params.challenges[challenge_idx];

                    self.play_game_pair(
                        params.depth,
                        &candidate_weights,
                        challenge,
                        params.opponents,
                    )
                })
                .collect();

            // Integrate results sequentially to respect SPRT causality.
            for (r1, r2) in results {
                candidate.match_history.update_pair(r1, r2);
                sprt.evaluate(&mut candidate.match_history);
                if candidate.match_history.game_pairs * 2 >= params.match_limit
                    || candidate.match_history.status != MatchStatus::Active
                {
                    break;
                }
            }
        }
        active_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Plays a Red/Yellow game pair for a specific challenge.
    fn play_game_pair(
        &self,
        depth: u32,
        candidate: &HeuristicWeights,
        challenge: &Challenge,
        opponents: &[FloatWeights],
    ) -> (f32, f32) {
        let opponent_weights = opponents[challenge.champ_idx].to_heuristic();
        let geo = &self.geometries[challenge.geo_idx];

        // Play in pairs to ensure symmetry and cancel opening bias.
        // We use rayon::join to play the Red and Yellow games in parallel.
        rayon::join(
            || {
                play_single_game(
                    *candidate,
                    opponent_weights,
                    geo,
                    depth,
                    Player::Red,
                    false,
                    &challenge.opening,
                    TUNE_TT_SIZE_MB,
                )
            },
            || {
                play_single_game(
                    *candidate,
                    opponent_weights,
                    geo,
                    depth,
                    Player::Yellow,
                    false,
                    &challenge.opening,
                    TUNE_TT_SIZE_MB,
                )
            },
        )
    }

    /// Executes the evolutionary step (Elitism + Mutation) to produce the next population.
    fn evolve(population: &mut Vec<Candidate>, best_gen: &Candidate, next_pop_size: usize) {
        assert!(
            next_pop_size <= population.len(),
            "Next generation population size {next_pop_size} cannot exceed current size {}",
            population.len()
        );
        assert!(
            population
                .windows(2)
                .all(|w| w[0].fitness() >= w[1].fitness()),
            "Population must be sorted by fitness before evolution (descending)"
        );
        let mut rng = rand::rng();
        let mut next_gen = Vec::with_capacity(next_pop_size);

        // 1. Elitism: Top X% survive unchanged.
        // This ensures that "unlucky" high-potential candidates get another chance
        // to prove themselves in the next generation's gauntlet.
        let elite_count = (next_pop_size * ELITE_PERCENT / 100).max(1);

        for (i, candidate) in population.iter().enumerate().take(elite_count) {
            let mut elite = candidate.clone();
            elite.index = i;
            elite.match_history = MatchHistory::default();
            next_gen.push(elite);
        }

        // 2. Shifted Mutation: Fill the remaining slots with mutants.
        // We generate mutants using parents starting from index 0.
        // Logic:
        // - next_gen[0..E] are Clones of population[0..E]
        // - next_gen[E..N] are Mutants of population[0..(N-E)]
        // This means the Elites get TWO slots: one for survival (clone) and one for offspring (mutant).
        // The worst candidates (bottom X%) are dropped entirely and do not reproduce.
        let mutant_slots = next_pop_size - elite_count;
        for i in 0..mutant_slots {
            // Select random agents for difference vector
            let mut r1 = rng.random_range(0..population.len());
            while r1 == i {
                r1 = rng.random_range(0..population.len());
            }
            let mut r2 = rng.random_range(0..population.len());
            while r2 == i || r2 == r1 {
                r2 = rng.random_range(0..population.len());
            }

            let mut trial = population[i].clone();
            trial.mutate_params();
            trial.weights = mutate_de(
                population[i].weights, // Parent is population[i]
                best_gen.weights,      // Attractor is Generation Leader
                population[r1].weights,
                population[r2].weights,
                trial.f,
                trial.cr,
            );
            trial.clamp();
            trial.match_history = MatchHistory::default();
            trial.index = elite_count + i;
            next_gen.push(trial);
        }

        // Fresh blood: X% of the NEXT population size
        let mutant_count = (next_pop_size * FRESH_BLOOD_PERCENT / 100).max(1);
        let start_idx = next_pop_size.saturating_sub(mutant_count);
        // Ensure we never overwrite the elites
        let safe_start_idx = start_idx.max(elite_count);
        for (i, mutant) in next_gen.iter_mut().enumerate().skip(safe_start_idx) {
            *mutant = Candidate::new_random(i);
        }

        *population = next_gen;
    }

    /// Generates a set of balanced challenges for the current generation.
    fn generate_challenges(
        &self,
        count: usize,
        opponents: &[FloatWeights],
        depth: u32,
    ) -> Vec<Challenge> {
        let active_champs = opponents.len();

        (0..count)
            .into_par_iter()
            .map(|i| {
                // Latin Square Interleaving: Ensures every aligned block of games
                // is balanced across both champions and geometries. A full period
                // of (Champs x Geos) covers every unique pairing exactly once.
                let champ_idx = i % active_champs;
                let geo_idx = (i + (i / active_champs)) % self.geometries.len();
                let geo = &self.geometries[geo_idx];

                // Gauntlet-Aware DEF: We filter the opening against the specific
                // opponent assigned to this challenge. This ensures that the match
                // starts in contested territory for this specific strategic matchup.
                let opponent_heuristic = opponents[champ_idx].to_heuristic();

                Challenge {
                    opening: generate_single_opening(geo, &opponent_heuristic, depth),
                    champ_idx,
                    geo_idx,
                }
            })
            .collect()
    }
}

// ========================================================================================
// EVOLUTIONARY MATH
// ========================================================================================

/// Performs Differential Evolution (DE) mutation using the 'DE/current-to-best/1/bin' strategy.
///
/// This strategy balances exploitation (pulling towards the generation leader)
/// with exploration (using the difference vector between two random candidates).
fn mutate_de(
    target: FloatWeights,
    best: FloatWeights,
    r1: FloatWeights,
    r2: FloatWeights,
    f: f32,
    cr: f32,
) -> FloatWeights {
    let mut rng = rand::rng();
    let mut trial = target;
    let forced_idx = rng.random_range(0..WEIGHT_FIELD_COUNT);
    let target_fields = target.fields();
    let best_fields = best.fields();
    let r1_fields = r1.fields();
    let r2_fields = r2.fields();
    let trial_fields = trial.fields_mut();

    for i in 0..WEIGHT_FIELD_COUNT {
        if i == forced_idx || rng.random_range(0.0..1.0) < cr {
            // DE/current-to-best/1/bin Strategy
            // V = X_current + F * (X_best - X_current) + F * (X_r1 - X_r2)
            // This balances exploitation (pull towards best) with exploration (difference vector).
            *trial_fields[i] = target_fields[i]
                + f * (best_fields[i] - target_fields[i])
                + f * (r1_fields[i] - r2_fields[i]);
        }
    }
    trial
}

// ========================================================================================
// MATCHMAKING ENGINE
// ========================================================================================

/// Generates a single contested opening using Dynamic Evaluative Filtering (DEF).
///
/// DEF ensures that matches start in positions where neither player has a decisive
/// advantage, leading to higher quality games for tuning.
fn generate_single_opening(
    geo: &DynamicBoardGeometry,
    evaluator: &HeuristicWeights,
    depth: u32,
) -> Opening {
    let mut rng = rand::rng();
    let threshold = evaluator.score_threat_immediate / 5; // 20% of a threat
    let mut best_opening = Opening { moves: Vec::new() };
    let mut best_score = i32::MAX;
    let mut engine = DynamicEngine::new(geo.clone(), *evaluator, Some(OPENING_TT_SIZE_MB));

    for _ in 0..DEF_MAX_RETRIES {
        // IMPORTANT: The move count is generated INSIDE the loop.
        // This prevents infinite retries on statistically rare high-move counts.
        let moves = rng.random_range(OPENING_MIN_MOVES..=OPENING_MAX_MOVES);
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
            let col = playable[rng.random_range(0..playable.len())];
            state = state.drop_piece(col, curr_p, geo).unwrap();
            if state.has_won(curr_p, geo) || state.is_full(geo) {
                valid = false;
                break;
            }
            curr_moves.push(col);
            curr_p = curr_p.other();
        }
        if valid {
            let score = engine.evaluate_position(&state, curr_p, depth);
            if score.abs() < threshold {
                return Opening { moves: curr_moves };
            }
            if score.abs() < best_score {
                best_score = score.abs();
                best_opening = Opening { moves: curr_moves };
            }
        }
    }
    tracing::warn!(
        "DEF: Max retries ({}) reached. Using best opening found (Score: {}).",
        DEF_MAX_RETRIES,
        best_score
    );
    best_opening
}

/// Simulates a single game between a candidate and an opponent from a given opening.
///
/// Returns 1.0 if the candidate wins, 0.5 for a draw, and 0.0 if the opponent wins.
#[allow(clippy::too_many_arguments)]
fn play_single_game(
    candidate: HeuristicWeights,
    opponent: HeuristicWeights,
    geo: &DynamicBoardGeometry,
    depth: u32,
    candidate_color: Player,
    randomize: bool,
    opening: &Opening,
    tt_size_mb: usize,
) -> f32 {
    let mut state = DynamicBoardState::new(geo);
    let mut curr_p = Player::Red;
    for &col in &opening.moves {
        state = state.drop_piece(col, curr_p, geo).unwrap();
        curr_p = curr_p.other();
    }

    let mut engine1 = DynamicEngine::new(geo.clone(), candidate, Some(tt_size_mb));
    let mut engine2 = DynamicEngine::new(geo.clone(), opponent, Some(tt_size_mb));

    while !state.is_full(geo) {
        let engine = if curr_p == candidate_color {
            &mut engine1
        } else {
            &mut engine2
        };
        let col = engine.find_best_move(&state, curr_p, depth, randomize);
        // We unwrap since the simulation loop ensures the game is not terminal.
        state = state
            .drop_piece(col.expect("No moves available"), curr_p, geo)
            .unwrap();
        if state.has_won(curr_p, geo) {
            return if curr_p == candidate_color { 1.0 } else { 0.0 };
        }
        curr_p = curr_p.other();
    }
    0.5
}

// ========================================================================================
// MAIN ENTRY
// ========================================================================================

/// The main entry point for the Connect 4 AI Tuner.
fn main() {
    let cores = rayon::current_num_threads();

    // Initialize structured logging
    let _guard = init_tracing();

    let mut tuner = Tuner::new();
    let mut population = Tuner::init_population();

    tracing::info!("============================================================");
    tracing::info!("CONNECT 4 AI TUNER");
    tracing::info!(
        "Hardware: {cores} Cores | Population: {POPULATION_START_SIZE} -> {POPULATION_END_SIZE} | Generations: {TOTAL_GENERATIONS}"
    );
    tracing::info!("============================================================\n");

    // Execute the main generational tuning loop
    tuner.run(&mut population);

    // Perform final cross-geometry validation for the optimized elite
    let best_float = population[0].weights;
    let best = best_float.to_heuristic();

    let avg_wr = tuner.final_validation(&best);

    tracing::info!("\n--- OPTIMIZED HEURISTIC WEIGHTS ---");
    tracing::info!("Avg Win Rate vs Default: {avg_wr:.1}%");
    tracing::info!("High-Precision Weights: {best_float:#?}");
    tracing::info!("Final Integer Weights: {best:#?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_history_update_pair() {
        let mut hist = MatchHistory::default();
        // WW
        hist.update_pair(1.0, 1.0);
        assert_eq!(hist.game_pairs, 1);
        assert_eq!(hist.pair_counts[4], 1);
        assert!((hist.score() - 2.0).abs() < f32::EPSILON);

        // LD
        hist.update_pair(0.0, 0.5);
        assert_eq!(hist.game_pairs, 2);
        assert_eq!(hist.pair_counts[1], 1);
        assert!((hist.score() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sprt_evaluation() {
        let params = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            elo0: 0.0,
            elo1: 50.0,
        };
        let sprt = Sprt::new(params);
        let mut hist = MatchHistory::default();

        // Need at least MIN_SPRT_PAIRS
        for _ in 0..MIN_SPRT_PAIRS {
            hist.update_pair(1.0, 1.0); // WW
        }
        sprt.evaluate(&mut hist);
        // With 3 WW and Elo1=50, it should be accepted
        assert_eq!(hist.status, MatchStatus::Accepted);

        let mut hist_fail = MatchHistory::default();
        for _ in 0..MIN_SPRT_PAIRS {
            hist_fail.update_pair(0.0, 0.0); // LL
        }
        sprt.evaluate(&mut hist_fail);
        assert_eq!(hist_fail.status, MatchStatus::Rejected);
    }

    #[test]
    fn test_avg_score_precision() {
        let mut c = Candidate::new_random(0);
        c.match_history = MatchHistory::default();

        // Scenario: 2 WW, 2 LD, 1 DD
        c.match_history.pair_counts[4] = 2; // 4.0 pts
        c.match_history.pair_counts[1] = 2; // 1.0 pts
        c.match_history.pair_counts[2] = 1; // 1.0 pts
        c.match_history.game_pairs = 5; // 10 total games

        // Expected: 6.0 pts / 10 games = 0.6
        assert!((c.avg_score() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sprt_draw_heavy_stability() {
        let params = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            elo0: -10.0,
            elo1: 10.0,
        };
        let sprt = Sprt::new(params);
        let mut hist = MatchHistory::default();

        // Fill with DD (Draw-Draw) or WL (Win-Loss) results (Bucket 2)
        // In a symmetric window (-10 to 10), LLR should remain exactly 0.0.
        for _ in 0..100 {
            hist.update_pair(0.5, 0.5); // DD
            sprt.evaluate(&mut hist);
            assert_eq!(hist.status, MatchStatus::Active);
        }
    }

    #[test]
    fn test_alm_monotonicity() {
        let w1 = FloatWeights::from_heuristic(HeuristicWeights::default());
        let mut w_far = w1;
        let mut last_dist = 0.0;

        // Incrementally drifting a single parameter should result in
        // strictly increasing distance.
        for i in 1..10 {
            #[allow(clippy::cast_precision_loss)]
            {
                w_far.score_three += 100.0 * i as f32;
            }
            let current_dist = w1.distance(&w_far);
            assert!(current_dist > last_dist, "Distance must be monotonic");
            last_dist = current_dist;
        }
    }

    #[test]
    fn test_hof_gatekeeper_protection() {
        let mut tuner = Tuner::new();
        let base = FloatWeights::from_heuristic(HeuristicWeights::default());
        // Fill HOF to capacity with unique species
        for i in 0..MAX_HOF_SIZE {
            let mut w = base;
            #[allow(clippy::cast_precision_loss)]
            {
                w.score_three += (i + 1) as f32 * 1000.0;
            }
            tuner.update_hof(w);
        }

        // Add a model extremely similar to the gatekeeper (at index 0)
        let mut similar_to_gatekeeper = base;
        similar_to_gatekeeper.weight_core += 0.1;

        tuner.update_hof(similar_to_gatekeeper);

        // Verification: Even if it's similar to index 0, the gatekeeper must be PROTECTED.
        // It should have either been added as a new member (if space) or replaced
        // the next closest member (since index 0 is skipped).
        assert!(
            (tuner.hall_of_fame[0].weight_core - base.weight_core).abs() < f32::EPSILON,
            "Gatekeeper at index 0 was corrupted!"
        );
    }

    #[test]
    fn test_hof_full_capacity_replacement() {
        let mut tuner = Tuner::new();
        let base = FloatWeights::from_heuristic(HeuristicWeights::default());

        // 1. Fill HOF with 10 distinct 'tactical niches' (distance > threshold)
        // We use exponential scaling because ALM distance is log-transformed.
        tuner.hall_of_fame = Vec::new();
        for i in 0..MAX_HOF_SIZE {
            let mut w = base;
            w.score_three *= 20.0f32.powi(i32::try_from(i).unwrap());
            tuner.hall_of_fame.push(w);
        }
        assert_eq!(tuner.hall_of_fame.len(), MAX_HOF_SIZE);

        // 2. Create a new species that is unique (dist > threshold from all)
        // BUT make it slightly closer to niche #5 than others.
        let mut new_species = tuner.hall_of_fame[5];
        // Dist to 5: 0.4 * ln(4.0) = 0.554 (Unique!)
        // Dist to 6: 0.4 * (ln(20) - ln(4.0)) = 0.4 * (2.996 - 1.386) = 0.644 (Unique!)
        new_species.score_three *= 4.0;

        // Verify it's unique but closest to 5
        for (i, member) in tuner.hall_of_fame.iter().enumerate().skip(1) {
            let d = new_species.distance(member);
            assert!(
                d > HOF_SIMILARITY_THRESHOLD,
                "Setup error: species not unique at index {i} (Dist: {d})"
            );
        }

        tuner.update_hof(new_species);

        // 3. Verification: HOF size must still be 10, and it should have replaced index 5.
        assert_eq!(tuner.hall_of_fame.len(), MAX_HOF_SIZE);
        assert!(
            (tuner.hall_of_fame[5].score_three - new_species.score_three).abs() < f32::EPSILON,
            "Failed to replace the most similar niche in a full HOF"
        );
    }

    #[test]
    fn test_hof_similarity_precedence() {
        let mut tuner = Tuner::new();
        let base = FloatWeights::from_heuristic(HeuristicWeights::default());
        // Fill HOF with distinct species
        tuner.hall_of_fame = Vec::new();
        for i in 0..MAX_HOF_SIZE - 1 {
            let mut w = base;
            #[allow(clippy::cast_precision_loss)]
            {
                w.score_three += i as f32 * 1000.0;
            }
            tuner.hall_of_fame.push(w);
        }

        // Create a model similar to index 2
        let mut similar_to_2 = tuner.hall_of_fame[2];
        similar_to_2.weight_core += 2.0;

        // Ensure it is closest to index 2 and under the similarity threshold
        assert!(tuner.hall_of_fame[2].distance(&similar_to_2) < HOF_SIMILARITY_THRESHOLD);

        tuner.update_hof(similar_to_2);

        // Verification: Functional replacement should have happened at index 2.
        assert!(
            (tuner.hall_of_fame[2].weight_core - similar_to_2.weight_core).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_seed_shuffling_independence() {
        let tuner = Tuner::new();
        let gatekeeper = HeuristicWeights::default();
        let active_champs = 1;
        let challenges =
            tuner.generate_challenges(100, &[FloatWeights::from_heuristic(gatekeeper)], 1);

        let gauntlet_period = active_champs * tuner.geometries.len();

        // This is a structural verification of the evaluate_population loop:
        let get_idx = |cand_idx: usize, game_pairs: usize, offset: usize| {
            let seed_offset = cand_idx * gauntlet_period;
            (game_pairs + seed_offset + offset) % challenges.len()
        };

        // At game_pairs = 0, offset = 0:
        assert_ne!(
            get_idx(0, 0, 0),
            get_idx(1, 0, 0),
            "Candidates must start with different openings"
        );
        assert_eq!(
            get_idx(0, 0, gauntlet_period),
            get_idx(1, 0, 0),
            "Candidate 2 should be 'ahead' in the sequence"
        );
    }

    #[test]
    fn test_clamp_push_effect() {
        let mut c = Candidate::new_random(0);

        // Set Core (Top tier) to a low value
        c.weights.weight_core = 5.0;
        // Set Inner (Mid tier) to a high value
        c.weights.weight_inner = 100.0;

        c.clamp();

        // Verification: Inner must have been "pushed" down by Core
        assert!((c.weights.weight_inner - 5.0).abs() < f32::EPSILON);

        // Multi-stage push: core -> inner -> outer
        c.weights.weight_core = 2.0;
        c.weights.weight_inner = 5.0;
        c.weights.weight_outer = 10.0;
        c.clamp();
        assert!((c.weights.weight_outer - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mutate_params_bounds() {
        let mut c = Candidate::new_random(0);
        // Run mutation 1000 times to ensure stochastic safety
        for _ in 0..1000 {
            c.mutate_params();
            assert!(c.f >= F_MIN && c.f <= (F_MIN + F_MAX));
            assert!(c.cr >= 0.0 && c.cr <= 1.0);
        }
    }

    #[test]
    fn test_alm_distance_properties() {
        let w1 = FloatWeights::from_heuristic(HeuristicWeights::default());
        let mut w2 = w1;

        // Identity
        assert!(w1.distance(&w2).abs() < f32::EPSILON);

        // Symmetry
        w2.score_three += 500.0;
        assert!((w1.distance(&w2) - w2.distance(&w1)).abs() < f32::EPSILON);

        // Sensitivity Verification:
        // A 10% change in score_three (Sensitivity 0.4) should be > than
        // a 10% change in weight_outer (Sensitivity 0.3).
        let mut w_three = w1;
        w_three.score_three *= 1.1;
        let mut w_outer = w1;
        let w_outer_ref = w_outer;
        w_outer.weight_outer *= 1.1;

        let d_three = w1.distance(&w_three);
        let d_outer = w_outer_ref.distance(&w_outer);
        assert!(
            d_three > d_outer,
            "Tactical shift should outweigh positional shift"
        );
        assert!(d_three > 0.03, "10% Three shift should be significant");
    }

    #[test]
    fn test_candidate_clamp_hierarchies() {
        let mut c = Candidate::new_random(0);
        // Break the hierarchies
        c.weights.weight_core = 5.0;
        c.weights.weight_inner = 10.0; // Inner > Core (Invalid)
        c.weights.weight_outer = 15.0; // Outer > Inner (Invalid)

        c.clamp();

        assert!(c.weights.weight_inner <= c.weights.weight_core);
        assert!(c.weights.weight_outer <= c.weights.weight_inner);
    }

    #[test]
    fn test_de_mutation_determinism() {
        let target = FloatWeights::from_heuristic(HeuristicWeights::default());
        let best = target;
        let mut r1 = target;
        // Modify all fields to ensure any forced_idx produces a diff
        for f in r1.fields_mut() {
            *f += 1000.0;
        }
        let r2 = target;

        // If F=0.5 and CR=0.0 (forced skip), trial should equal target except for one field
        let trial = mutate_de(target, best, r1, r2, 0.5, 0.0);
        let mut diff_count = 0;
        let target_f = target.fields();
        let trial_f = trial.fields();
        for i in 0..WEIGHT_FIELD_COUNT {
            if (target_f[i] - trial_f[i]).abs() > f32::EPSILON {
                diff_count += 1;
            }
        }
        assert_eq!(diff_count, 1, "CR=0 should mutate exactly one forced field");
    }

    #[test]
    fn test_hof_replacement_logic() {
        let mut tuner = Tuner::new();
        let base = FloatWeights::from_heuristic(HeuristicWeights::default());
        tuner.hall_of_fame = vec![base]; // Reset to just gatekeeper

        // 1. Add a unique species
        let mut unique = base;
        // Need multiplier > 3.5 to cross threshold of 0.5 with sensitivity 0.4
        unique.score_three *= 5.0;
        assert!(base.distance(&unique) > HOF_SIMILARITY_THRESHOLD);
        tuner.update_hof(unique);
        assert_eq!(tuner.hall_of_fame.len(), 2);

        // 2. Add a similar species (should replace 'unique')
        let mut similar = unique;
        similar.weight_core += 1.0;
        assert!(unique.distance(&similar) < HOF_SIMILARITY_THRESHOLD);
        tuner.update_hof(similar);
        assert_eq!(tuner.hall_of_fame.len(), 2);
        assert!((tuner.hall_of_fame[1].weight_core - similar.weight_core).abs() < f32::EPSILON);

        // 3. Fill HOF to test FIFO
        // We use massive exponential multipliers to ensure they are distinct species
        for i in 0..12 {
            let mut w = base;
            w.score_three *= 5.0f32.powi(i + 2);
            tuner.update_hof(w);
        }
        assert_eq!(tuner.hall_of_fame.len(), MAX_HOF_SIZE);

        // Ensure Gatekeeper at index 0 was never replaced
        assert!((tuner.hall_of_fame[0].score_three - base.score_three).abs() < f32::EPSILON);
    }

    #[test]
    fn test_opening_validity() {
        let geo = DynamicBoardGeometry::new(7, 6);
        let gatekeeper = HeuristicWeights::default();
        for _ in 0..10 {
            let opening = generate_single_opening(&geo, &gatekeeper, 3);
            let mut state = DynamicBoardState::new(&geo);
            let mut p = Player::Red;
            for &m in &opening.moves {
                state = state
                    .drop_piece(m, p, &geo)
                    .expect("Opening move must be valid");
                assert!(!state.has_won(p, &geo), "Opening cannot be a win");
                p = p.other();
            }
            assert!(!state.is_full(&geo), "Opening cannot be a full board");
        }
    }

    #[test]
    fn test_de_mutation_full_crossover() {
        let target = FloatWeights::from_heuristic(HeuristicWeights::default());
        let best = target;
        let mut r1 = target;
        for f in r1.fields_mut() {
            *f += 1000.0;
        }
        let r2 = target;

        // If CR=1.0, ALL fields should be mutated
        let trial = mutate_de(target, best, r1, r2, 0.5, 1.0);
        let target_f = target.fields();
        let trial_f = trial.fields();
        for i in 0..WEIGHT_FIELD_COUNT {
            assert!(
                (target_f[i] - trial_f[i]).abs() > f32::EPSILON,
                "Field {i} should have mutated"
            );
        }
    }

    #[test]
    fn test_latin_square_coverage() {
        let tuner = Tuner::new();
        let gatekeeper = HeuristicWeights::default();
        let active_champs = 3;
        let count = 15;
        let opponents = vec![FloatWeights::from_heuristic(gatekeeper); active_champs];
        let challenges = tuner.generate_challenges(count, &opponents, 1);

        assert_eq!(challenges.len(), count);

        // Check champion interleaving: 0, 1, 2, 0, 1, 2...
        for (i, c) in challenges.iter().enumerate() {
            assert_eq!(c.champ_idx, i % active_champs);
        }

        // Check geometry coverage: Ensure all 3 geometries are used
        let mut geo_counts = [0; 3];
        for c in &challenges {
            geo_counts[c.geo_idx] += 1;
        }
        for (i, &cnt) in geo_counts.iter().enumerate() {
            assert!(cnt > 0, "Geometry {i} was never used");
        }
    }

    #[test]
    fn test_opening_move_count_bounds() {
        let geo = DynamicBoardGeometry::new(7, 6);
        let gatekeeper = HeuristicWeights::default();
        for _ in 0..100 {
            let opening = generate_single_opening(&geo, &gatekeeper, 3);
            #[allow(clippy::absurd_extreme_comparisons)]
            {
                assert!(opening.moves.len() >= OPENING_MIN_MOVES);
            }
            assert!(opening.moves.len() <= OPENING_MAX_MOVES);
        }
    }

    #[test]
    fn test_population_ranking_and_selection() {
        let mut population: Vec<Candidate> = (0..10).map(Candidate::new_random).collect();

        // Scenario:
        // Candidate 5: Active, Avg Score 0.8 (Fitness 1.8)
        // Candidate 2: Accepted, Avg Score 0.1 (Fitness 2.1)
        // Candidate 2 should rank HIGHER than Candidate 5 despite lower mean.

        population[5].match_history.pair_counts[4] = 80; // 160 pts
        population[5].match_history.game_pairs = 100;
        population[5].match_history.status = MatchStatus::Active;

        population[2].match_history.pair_counts[4] = 5; // 10 pts
        population[2].match_history.game_pairs = 50;
        population[2].match_history.status = MatchStatus::Accepted;

        let mut ranked = population.clone();
        ranked.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        assert_eq!(ranked[0].index, 2, "Accepted candidate must rank first");
        assert_eq!(
            ranked[1].index, 5,
            "High-mean Active candidate must rank second"
        );

        // Survivor logic (top 50%)
        let survivors: Vec<Candidate> = ranked.into_iter().take(5).collect();
        assert_eq!(survivors.len(), 5);
        assert_eq!(survivors[0].index, 2);
    }

    #[test]
    fn test_elo_resolution_scaling() {
        // Targets: ELO = ELO_RESOLUTION_FACTOR / sqrt(N)
        #[allow(clippy::cast_precision_loss)]
        let n_start = PHASE1_START_GAMES as f64;
        #[allow(clippy::cast_precision_loss)]
        let n_end = PHASE2_END_GAMES as f64;

        let elo_start = ELO_RESOLUTION_FACTOR / n_start.sqrt();
        let elo_end = ELO_RESOLUTION_FACTOR / n_end.sqrt();

        // Verification: Precision scaling should follow inverse square root law
        assert!((elo_start / elo_end - (n_end / n_start).sqrt()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_match_history_status_latching() {
        let params = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            elo0: 0.0,
            elo1: 50.0,
        };
        let sprt = Sprt::new(params);
        let mut hist = MatchHistory::default();

        // 1. Force a Rejection
        for _ in 0..10 {
            hist.update_pair(0.0, 0.0); // LL
        }
        sprt.evaluate(&mut hist);
        assert_eq!(hist.status, MatchStatus::Rejected);

        // 2. Attempt to "recover" by adding wins
        for _ in 0..100 {
            hist.update_pair(1.0, 1.0); // WW
        }
        sprt.evaluate(&mut hist);
        // Status must remain Rejected
        assert_eq!(hist.status, MatchStatus::Rejected);
    }

    #[test]
    fn test_float_to_heuristic_fidelity() {
        let hw_custom = HeuristicWeights {
            score_fork_immediate: 99999,   // Non-default
            score_threat_immediate: 88888, // Non-default
            ..HeuristicWeights::default()
        };

        let fw = FloatWeights::from_heuristic(hw_custom);
        let hw = fw.to_heuristic();
        let defaults = HeuristicWeights::default();

        // CRITICAL: `score_fork_immediate` must be PINNED to the library's default
        assert_eq!(hw.score_fork_immediate, defaults.score_fork_immediate);
        // CRITICAL: `score_threat_immediate` must be ANCHORED to the library's default
        assert_eq!(hw.score_threat_immediate, defaults.score_threat_immediate);
    }

    #[test]
    fn test_sprt_variance_clamping() {
        let params = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            elo0: 0.0,
            elo1: 5.0, // Tight window
        };
        let sprt = Sprt::new(params);
        let mut hist = MatchHistory::default();

        // Fill with perfectly uniform results (all DD)
        // Mathematically, variance (sigma2) would be 0.0, causing a divide-by-zero.
        // We test that the tuner correctly clamps this to MIN_VARIANCE.
        for _ in 0..MIN_SPRT_PAIRS {
            hist.update_pair(0.5, 0.5);
        }

        // This should not panic
        sprt.evaluate(&mut hist);
        assert_eq!(hist.status, MatchStatus::Active);
    }

    #[test]
    fn test_matchmaking_symmetry_property() {
        let weights = HeuristicWeights::default();
        let geo = DynamicBoardGeometry::new(7, 6);
        let opening = Opening {
            moves: vec![3, 3, 2, 4],
        }; // Symmetrical-ish opening

        // Identity Property: A player against themselves must result in 1.0 pair score
        // (Either 0.5 + 0.5 or 1.0 + 0.0)
        let r1 = play_single_game(
            weights,
            weights,
            &geo,
            3,
            Player::Red,
            false,
            &opening,
            TUNE_TT_SIZE_MB,
        );
        let r2 = play_single_game(
            weights,
            weights,
            &geo,
            3,
            Player::Yellow,
            false,
            &opening,
            TUNE_TT_SIZE_MB,
        );

        assert!(
            (r1 + r2 - 1.0).abs() < f32::EPSILON,
            "Pair score against self must be 1.0"
        );
    }

    #[test]
    fn test_evolution_best_preservation() {
        let mut population: Vec<Candidate> = (0..POPULATION_START_SIZE)
            .map(Candidate::new_random)
            .collect();
        // Give candidate 5 a great score
        population[5].match_history.pair_counts[4] = 100;
        population[5].match_history.game_pairs = 100;
        let best = population[5].clone();

        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        Tuner::evolve(&mut population, &best, POPULATION_START_SIZE);

        // Verification: The best must be at index 0, and its history must be reset
        assert!(
            (population[0].weights.score_three - best.weights.score_three).abs() < f32::EPSILON
        );
        assert_eq!(population[0].match_history.game_pairs, 0);
        assert_eq!(population[0].index, 0);
    }

    #[test]
    fn test_tuner_schedule_logic() {
        // Test Generation 0
        let (p1_0, p2_0) = get_phase_limits(0);
        assert_eq!(p1_0, PHASE1_START_GAMES);
        assert_eq!(p2_0, PHASE2_START_GAMES);

        // Test Final Generation
        let (p1_final, p2_final) = get_phase_limits(TOTAL_GENERATIONS - 1);
        assert_eq!(p1_final, PHASE1_END_GAMES & !1);
        assert_eq!(p2_final, PHASE2_END_GAMES & !1);
    }

    #[test]
    fn test_sprt_early_exit_invariant() {
        let params = SprtParams {
            alpha: 0.05,
            beta: 0.05,
            elo0: 0.0,
            elo1: 50.0,
        };
        let sprt = Sprt::new(params);
        let mut hist = MatchHistory::default();

        // Scenario: 2 WW results (extremely one-sided)
        // BUT: game_pairs (2) < MIN_SPRT_PAIRS (3)
        hist.update_pair(1.0, 1.0);
        hist.update_pair(1.0, 1.0);

        sprt.evaluate(&mut hist);

        // Invariant: Status MUST remain Active until MIN_SPRT_PAIRS is met
        assert_eq!(hist.status, MatchStatus::Active);
    }

    #[test]
    fn test_numerical_boundary_robustness() {
        let mut fw = FloatWeights::from_heuristic(HeuristicWeights::default());

        // Test 0.0 (Log safety)
        fw.weight_outer = 0.0;
        let mut fw2 = fw;
        fw2.weight_outer = 1.0;
        let d = fw.distance(&fw2);
        assert!(d.is_finite() && d > 0.0);

        // Test Large Scale (Sensitivity limit)
        fw.score_three = 1_000_000.0;
        fw2.score_three = 1_000_100.0;
        let d_large = fw.distance(&fw2);
        assert!(d_large.is_finite() && d_large > 0.0);

        // Test Conversion Clamping at extremes
        fw.score_three = -100.0;
        let mut c = Candidate::new_random(0);
        c.weights = fw;
        c.clamp();
        assert!(c.weights.score_three >= 0.0); // Minimum clamp
    }

    #[test]
    fn test_parallel_orchestration_layer() {
        // Mini-tuner setup
        let tuner = Tuner::new();
        let mut population: Vec<Candidate> = (0..4).map(Candidate::new_random).collect();
        let depth = 1; // Fast search
        let match_limit = 2; // Fast match

        let sprt_params = SprtParams {
            alpha: 0.1,
            beta: 0.1,
            elo0: 0.0,
            elo1: 100.0,
        };
        let gatekeeper = HeuristicWeights::default();
        let opponents = vec![FloatWeights::from_heuristic(gatekeeper)];
        let challenges = tuner.generate_challenges(1, &opponents, 3);

        // This verifies that the orchestration works
        tuner.evaluate_population(
            &mut population,
            depth,
            match_limit,
            sprt_params,
            &challenges,
            &opponents,
        );

        for c in population {
            assert!(c.match_history.game_pairs > 0);
        }
    }

    #[test]
    fn test_de_current_to_best_logic() {
        let mut target = FloatWeights::from_heuristic(HeuristicWeights::default());
        target.score_three = 100.0;

        let mut best = target;
        best.score_three = 200.0;

        let mut r1 = target;
        r1.score_three = 300.0;

        let mut r2 = target;
        r2.score_three = 300.0; // r1 - r2 = 0

        // Case 1: Only exploitation (Current -> Best)
        // Formula: V = Target + F*(Best - Target) + F*(R1 - R2)
        // V = 100 + 0.5 * (200 - 100) + 0 = 150
        let result1 = mutate_de(target, best, r1, r2, 0.5, 1.0);
        assert!(
            (result1.score_three - 150.0).abs() < f32::EPSILON,
            "Failed exploitation logic. Expected 150.0, got {}",
            result1.score_three
        );

        // Case 2: Full formula with exploration
        // r1 = 300, r2 = 200 -> diff = 100
        // V = 100 + 0.5*(100) + 0.5*(100) = 200
        let mut r2_diff = r2;
        r2_diff.score_three = 200.0;
        let result2 = mutate_de(target, best, r1, r2_diff, 0.5, 1.0);
        assert!(
            (result2.score_three - 200.0).abs() < f32::EPSILON,
            "Failed full logic. Expected 200.0, got {}",
            result2.score_three
        );
    }
}

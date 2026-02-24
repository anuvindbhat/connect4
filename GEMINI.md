# Project Overview: Connect 4 (Library/TUI)

A high-performance implementation of Connect 4, featuring a sophisticated bitboard engine, an advanced negamax AI with stochastic move selection, and LAN-based remote multiplayer capabilities.

The project is structured as a dual-purpose codebase:

1. **Core Engine (Library)**: A headless, high-performance library usable for simulations and AI research.
2. **TUI Application**: A polished Terminal User Interface built on top of the library.

## Core Technologies

- **Language**: Rust (2024 Edition)
- **Concurrency & Async**: `tokio` for networking and background tasks.
- **TUI Framework**: `ratatui` with `crossterm` backend (feature-gated).
- **Serialization**: `postcard` (binary) and `serde` for network protocols.
- **Networking**: UDP Discovery (local broadcast) via `network-interface` and TCP for game synchronization.
- **Diagnostics & Logging**: `tracing` with `tracing-appender` for structured file logging.
- **Error Handling**: `thiserror` (lib) and `color-eyre` (app).
- **Argument Parsing**: `clap` for CLI configuration.

## Project Structure

### Library Crate (`src/lib.rs`)

- `src/types.rs`: Consolidated domain models (Player, Difficulty, GameMode, etc.).
- `src/error.rs`: Central library error handling via `Connect4Error`.
- `src/game.rs`: Pure bitboard engine. Manages state, validation, and win detection. Supports u64 and u128 bitboards.
- `src/engine.rs`: High-level AI search engine abstraction. Manages bitboard variants and Transposition Table (TT) lifecycle (Owned/Pooled).
- `src/ai.rs`: Negamax search with Alpha-Beta pruning, Principal Variation Search (PVS), Boltzmann selection, and Heuristic Evaluation.
- `src/tt.rs`: Two-Tier Transposition Table with depth-preferred and always-replace slots.
- `src/zobrist.rs`: Deterministic Zobrist hashing for O(1) board state updates and indexing.
- `src/network.rs`: LAN multiplayer implementation (Discovery & Synchronization).
- `src/game_session.rs`: Logical session manager (Logic, History, Analytics).
- `src/config.rs`: Engine configuration (AI depths, heuristic weights, TT sizes).

### Binary Application (`src/main.rs`)

- `src/main.rs`: TUI entry point. Orchestrates the `App` state machine.
- `src/ui.rs`: `ratatui` rendering logic. Uses `RenderContext` and `UiBoardGeometry` for layout and cell mapping.
- `src/input.rs`: TUI event handling and semantic action mapping.
- `src/ui_session.rs`: TUI View-Model (Cosmetic animations, cursor state).
- `src/ui_config.rs`: TUI configuration (Layout, colors, labels, cosmetic physics).

### Tools (`src/bin/`)

- `src/bin/tune_ai.rs`: Evolutionary Strategy (jDE) tuner for heuristic weights using SPRT.
- `src/bin/sim_ai.rs`: AI vs AI tournament simulator and Elo rating calculator.
- `src/bin/bench_ai.rs`: Performance benchmarking and tactical suite runner.
- `src/bin/profile_ai.rs`: Simple tool for profiling AI search performance.

## Building and Running

This project uses `mise` for toolchain management.

- **Run TUI App**: `cargo run`
- **Run Headless Lib Check**: `cargo check --no-default-features --all-targets`
- **Run a Tool (Tuner)**: `cargo run --features tools --bin tune_ai`
- **Run All Tests**: `cargo test --all-features --all-targets`
- **Lint All Code**: `cargo clippy --all-targets --all-features`
- **Run Formatter**: `cargo fmt`

---

## Technical Architecture & Best Practices

### 1. The Bitboard Engine (`game.rs`)

- **State Representation**: Uses two `Bitboard` bitsets (u64 for small boards, u128 for large).
- **Flexible Geometry**: Supports Standard (7x6), Large (8x7), and Giant (9x7) via `BoardGeometry`.
- **Sentinel Bit Strategy**: Each column includes a sentinel bit at the top (row N+1, always 0). This prevents horizontal and diagonal wins from wrapping across column boundaries during bit-shifts.
- **Sliding Intersection**: Win detection uses optimized $O(1)$ shifts: `m = bits & (bits >> d); if m & (m >> 2d) != 0 { win }`.
- **Carry Chain Trick**: Finding the first empty row in a column is done via `((occupied & col_mask) + bottom_bit) & col_mask`, avoiding iterative row checks and branching.
- **Move Ordering**: `get_move_order` calculates a high-quality move-order extremely quickly using tactical tiers (Win, Block, Setup) and positional masks.

### 2. Search Engine (`engine.rs`, `ai.rs`, `tt.rs`, `zobrist.rs`)

- **Engine Abstraction**: `Engine<T>` encapsulates the search logic, heuristic weights, and Transposition Table. `DynamicEngine` provides a runtime choice between u64 and u128 variants.
- **Search Algorithm**: Negamax with Alpha-Beta pruning and Principal Variation Search (PVS) for enhanced pruning.
- **Iterative Deepening**: Provides move ordering for the root search, prioritizing tactical threats and TT hints.
- **Transposition Tables (TT)**:
  - **Strategy**: Two-Tier replacement (Depth-Preferred and Always-Replace). `MIN_TT_DEPTH = 2`.
  - **Collision Protection**: Stores the full `BoardState` in each entry to guarantee 100% collision integrity.
  - **Pooling**: Uses thread-local RAII guards (`TTGuard`) for zero-allocation reuse of large tables across searches via `TTHolder`.
- **Zobrist Hashing**: Provides O(1) incremental state updates and fast TT indexing using 64-bit deterministic keys seeded from "Connect4".
- **Stochastic Selection**: Uses Boltzmann (Softmax) selection ($P(i) \propto e^{score_i / \tau}$) at the root. Temperature $\tau$ (base 15.0) decays quadratically with board occupancy.
- **Heuristic Hierarchy**: Detailed heuristic evaluation with tapering (positional, structural, tactical, mobility, fork).
  - **Terminal**: `SCORE_WIN` (100,000,000). Shorter wins are prioritized via a per-ply "tax".
  - **Tactical**: Immediate and future threats. These do not decay.
  - **Structural/Connections**: Connection counts (2-in-a-window, 3-in-a-window). These scores decay linearly (`rem / max`) as the board fills.
  - **Mobility**: Open window counts. These decay linearly as the board fills.
  - **Positional**: Heatmap-based cell control. These scores decay quadratically (`rem^2 / max^2`) as the board fills.

### 3. Remote Multiplayer (`network.rs`)

- **Discovery**: Uses UDP broadcasts on port 4445. Utilizes `network-interface` to identify all local network interfaces and send directed-broadcasts to each subnet.
- **Synchronization**: Uses TCP for reliable transmission. Framed with length-delimited encoding.
- **Protocol**: Compact binary serialization via `postcard`. `PROTOCOL_VERSION = 1`.
- **Desync Detection**: Move messages include a `state_hash` (u128). If hashes don't match, the game enters a `Desync` terminal state.
- **Handshake**: Symmetric protocol ensures both host and client agree on board geometry and colors before starting. `DiscoveryInfo` includes `instance_id` and `local_name`.
- **Heartbeat**: Periodic `Ping`/`Pong` messages monitor connection health and calculate RTT (latency) displayed in the UI.

### 4. Modular TUI Architecture

- **State Machine**: `AppState` manages transitions (Menu, Lobby, Playing, Game Over, Error).
- **View-Model Separation**: `UiGameSession` (in `src/ui_session.rs`) wraps the library's `GameSession` to separate cosmetic state from game logic.
- **Rendering**: `src/ui.rs` uses a `RenderContext` to manage drawing operations and `UiBoardGeometry` to map board coordinates to terminal cells.
- **Asynchronous Execution**: Both AI move selection and tactical analysis run in background tasks via `tokio::spawn`, ensuring the TUI remains responsive.
- **3-Phase Animation System**:
  1. **Logic State**: Board state is updated immediately upon move entry (`make_move`) for stable search and sync.
  2. **Global Hide**: The target cell in the board grid is hidden from rendering if it exists in the `falling_pieces` queue to prevent ghosting.
  3. **Cosmetic Overlay**: A non-blocking `falling_pieces` queue interpolates piece position for smooth visual descent using physics-based gravity (`GRAVITY = 230.0`).
- **Professional Keybindings**:
  - `Alt+A`: Toggle Analysis Pane (Tactical Breakdown & Evaluation Bar).
  - `Alt+H`: Toggle Move History Pane.
  - `Alt+C`: Toggle Chat Visibility.
  - `/`: Open Chat Input (Remote mode only).
  - `1-9`: Direct column selection.
  - `S`: Swap sides and restart (Local/AI mode only, from Game Over screen).
  - `Esc`: Close Chat Input or return to Menu.

### 5. Diagnostics & Tuning (`src/bin/`)

- **Tuner (jDE)**: Self-Adaptive Differential Evolution with SPRT statistical validation. Tunes heuristic weights against a SPECTRUM of opening positions.
- **Simulator**: Calculates Elo ratings and win rates for different AI versions/depths.
- **Bench**: Evaluates search performance (NPS, Nodes, EBF) and tactical accuracy against a suite of puzzles.
- **Profile**: Minimal tool for measuring search throughput and TT efficiency.

### 6. Engineering Workflow

- **Atomic Commits**: Small, single-purpose commits.
- **Commit Auditing**: Before committing, always perform a line-by-line analysis of `git diff` to ensure the changes are correct.
- **Logging First**: Use `tracing` for debugging networking and AI. Standard log level is `debug`.
- **Mandatory Lints**: Never commit without passing `cargo clippy --all-targets --all-features` and `cargo fmt`. Note that `clippy --pedantic` and `--all` are enforced at the project level via `Cargo.toml`.
- **Headless Integrity**: Always verify library compilation with `cargo check --no-default-features --all-targets`.
- **Verification**: Always run `cargo test --all-features --all-targets` before submitting changes.
- **UI Constraints**: The TUI requires a minimum size of `105x38` characters.

### 7. Code Guidelines

- **Idiomatic Rust**: Ensure code is idiomatic, modular, and follows best practices.
- **Assert Invariants**: Add assertions for invariants (with helpful messages) especially if they're cheap to compute.
- **Clippy**: `#[allow(clippy::...)]` annotations must only be used as a last resort.
- **Panics**: All panics (even assertions in tests) must have a helpful and informative message.
- **Unwrapping/Error Handling**: It is okay (even preferable) to use `.expect` when an error isn't recoverable. Prefer this to using `.unwrap_or` to fall-back to an inaccurate default value.

# Connect 4 (Library/TUI)

A high-performance implementation of Connect 4, featuring a sophisticated bitboard engine, an advanced Negamax AI with stochastic selection, and LAN-based remote multiplayer capabilities.

This project is structured as a dual-purpose codebase:

1. **Core Engine (Library)**: A headless, high-performance library suitable for simulations, AI research, and integration into other Rust projects.
2. **TUI Application**: A polished Terminal User Interface built on top of the library using `ratatui`.

---

## Key Features

### Gameplay and User Interface

- **Three Game Modes**: Single Player (vs AI), Local Two-Player, and Remote Multiplayer (LAN).
- **Flexible Board Sizes**: Supports Standard (7x6), Large (8x7), and Giant (9x7) geometries with dynamic bitboard selection.
- **Physics-Based Animations**: Smooth piece dropping with realistic gravity, terminal velocity, and bounce effects.
- **Advanced Analysis**: Real-time tactical breakdown, win-probability evaluation bars, and move history.
- **In-Game Chat**: Integrated chat system for remote multiplayer sessions.

### AI and Search Engine

- **High-Performance Bitboard**: $O(1)$ win detection and piece dropping using specialized bitwise operations.
- **Negamax Search**: Enhanced with Alpha-Beta pruning, Principal Variation Search (PVS), and Iterative Deepening.
- **Transposition Tables**: Two-tier replacement strategy with Zobrist hashing for efficient state memoization.
- **Stochastic Selection**: Uses Boltzmann (Softmax) selection with quadratic temperature decay for a sophisticated challenge.
- **Tapered Heuristics**: Evaluation function that dynamically shifts focus from positional control to tactical urgency as the board fills.

### Networking

- **Zero-Config Discovery**: Automatic peer discovery on local networks via UDP broadcasts.
- **Reliable Synchronization**: Custom TCP protocol with framed length-delimited encoding and desync detection.
- **Heartbeat and Latency**: Real-time RTT (ping) monitoring displayed in the UI.

---

## Getting Started

### Prerequisites

- **Rust**: 2024 Edition (Stable).
- **mise**: (Recommended) Used for toolchain and environment management.
- **Terminal Size**: Minimum **105x39** characters required for the TUI.

### Installation and Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/connect4.git
   cd connect4
   ```
2. **Run the TUI Application**:
   ```bash
   cargo run --release
   ```
3. **Run All Tests**:
   ```bash
   cargo test --all-features --all-targets
   ```

---

## TUI Guide

### Controls

| Key | Action |
| :--- | :--- |
| `1`-`9` | Direct column selection |
| `Left` / `Right` | Navigate columns |
| `Enter` | Drop piece |
| `Alt+A` | Toggle Analysis Pane (Tactical Breakdown and Evaluation Bar) |
| `Alt+H` | Toggle Move History Pane |
| `Alt+C` | Toggle Chat Visibility (Remote only) |
| `/` | Open Chat Input (Remote only) |
| `S` | Swap sides and restart (Game Over screen, Local/AI only) |
| `R` | Restart game (Game Over screen, Local/AI only) |
| `Esc` / `Q` | Return to Menu / Exit |

### UI Layout

- **Main Board**: Centrally located with animated piece drops.
- **Status Header**: Displays current turn, RTT (if remote), and game status.
- **Intelligence Pane (`Alt+A`)**: Shows AI's evaluation of the current state and a tactical breakdown (threats, setups, mobility).
- **History Pane (`Alt+H`)**: A chronological log of all moves made in the current session.
- **Chat Pane (`Alt+C`)**: Displays recent messages from your opponent in LAN games.

---

## Developer Documentation

### Project Structure

- `src/lib.rs`: Library entry point.
- `src/game.rs`: Core bitboard engine (win detection, piece dropping).
- `src/ai.rs`: Negamax search and heuristic evaluation.
- `src/engine.rs`: High-level AI abstraction and search orchestration.
- `src/tt.rs`: Transposition table implementation.
- `src/network.rs`: LAN discovery and TCP synchronization protocol.
- `src/main.rs`: TUI application entry point.
- `src/ui.rs`: Rendering logic using `ratatui`.

**API Reference**: [anuvind.com/hosted/connect4/doc/connect4/](https://anuvind.com/hosted/connect4/doc/connect4/)

### Specialized Tools

The project includes several binaries for development and tuning, located in `src/bin/`:

- **AI Tuner (`tune-ai`)**: Optimizes heuristic weights using Differential Evolution (jDE) and SPRT.
  ```bash
  cargo run --features tools --bin tune-ai
  ```
- **Tournament Simulator (`sim-ai`)**: Calculates Elo ratings by running AI vs AI tournaments.
  ```bash
  cargo run --features tools --bin sim-ai
  ```
- **Performance Bench (`bench-ai`)**: Measures search throughput (NPS) and tactical accuracy.
  ```bash
  cargo run --bin bench-ai
  ```
- **Profiler (`profile-ai`)**: A minimal tool for profiling search performance and TT efficiency.
  ```bash
  cargo run --bin profile-ai
  ```

### Technical Architecture Highlights

- **Sentinel Bit Strategy**: Columns include an extra bit at the top to prevent bit-shifts from wrapping across boundaries.
- **Carry Chain Trick**: Finding the first empty row is done in constant time via `((occupied & col_mask) + bottom_bit) & col_mask`.
- **Sliding Intersection**: Win detection uses $O(1)$ shifts: `m = bits & (bits >> d); if m & (m >> 2d) != 0 { win }`.
- **Dynamic Bitboards**: The engine automatically chooses between `u64` (Standard/Large) and `u128` (Giant) based on the board geometry.

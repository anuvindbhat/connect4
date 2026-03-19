//! # Connect 4 TUI
//!
//! This is the entry point for the Connect 4 terminal application. It orchestrates
//! the game's state machine, manages networking for remote play, and coordinates
//! between the user interface and the underlying game logic.
//!
//! The application uses a central `App` struct to maintain state and drive the
//! execution of the TUI using the `ratatui` library.

#![cfg(feature = "tui")]

use crate::ui_config::{
    BOARD_SIZES, CHAT_HISTORY_LIMIT, COMPACT_NAME_LIMIT, INITIAL_VELOCITY, TICK_RATE_MS,
};
use crate::ui_session::{FallingPiece, UiGameSession};
use clap::Parser;
use color_eyre::Result;
use connect4::config::ANALYTICS_DEPTH;
use connect4::network::{
    DiscoveryInfo, NetworkCommand, NetworkEvent, NetworkManager, RemoteMessage,
};
use connect4::types::{Difficulty, GameMode, Player};
use crossterm::{
    event::{self, Event},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::collections::VecDeque;
use std::io;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod input;
mod ui;
mod ui_config;
mod ui_session;

// ========================================================================================
// CLI ARGUMENTS
// ========================================================================================

/// Command-line arguments for the TUI application.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Optional path to a log file. If provided, logging will be initialized to this file.
    #[arg(short = 'f', long, value_name = "FILE")]
    log_file: Option<std::path::PathBuf>,
    /// The logging level (e.g., "info", "debug", "error"). Defaults to "info".
    #[arg(short, long, default_value = "info")]
    level: String,
}

/// Initializes logging based on the provided CLI arguments.
///
/// If a log file is specified in `Cli`, it sets up a rolling file appender and
/// returns a `WorkerGuard` that must be held for the duration of the application
/// to ensure all logs are flushed.
fn init_logging(cli: &Cli) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    if let Some(ref log_path) = cli.log_file {
        let file_name = log_path.file_name()?.to_str()?;
        let directory = log_path.parent()?;
        let file_appender = tracing_appender::rolling::never(directory, file_name);
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.level));
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
            .init();
        info!("Logging initialized to {:?}", log_path);
        Some(guard)
    } else {
        None
    }
}

// ========================================================================================
// DATA STRUCTURES
// ========================================================================================

/// Manages the networking state and coordination for remote game sessions.
pub struct NetworkContext {
    /// The underlying network manager that handles low-level communication.
    pub manager: NetworkManager,
    /// A list of hosts discovered on the local network, along with the time they were last seen.
    pub discovered_hosts: Vec<(DiscoveryInfo, std::time::Instant)>,
    /// Whether the local application is currently acting as a host for a remote game.
    pub is_hosting: bool,
    /// The name of the local player.
    pub local_name: String,
    /// The name of the remote player.
    pub peer_name: String,
    /// The current network latency (RTT) in milliseconds, if available.
    pub latency: Option<u64>,
    /// A history of chat messages received during a remote session.
    pub chat_history: VecDeque<String>,
    /// The current content of the chat input field.
    pub chat_input: String,
    /// Whether to display the chat overlay.
    pub show_chat: bool,
}

impl NetworkContext {
    /// Creates a new `NetworkContext` with default settings and starts the network manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            manager: NetworkManager::start(),
            discovered_hosts: Vec::new(),
            is_hosting: false,
            local_name: String::new(),
            peer_name: "Opponent".to_string(),
            latency: None,
            chat_history: VecDeque::with_capacity(CHAT_HISTORY_LIMIT),
            chat_input: String::new(),
            show_chat: false,
        }
    }

    /// Updates internal maintenance tasks like host pruning.
    /// Returns (`selected_idx`, `selected_id`) for lobby UI synchronization.
    pub fn update(&mut self, state: &AppState) -> (usize, Option<u64>) {
        // 1. Prune stale hosts
        self.discovered_hosts
            .retain(|(_, last_seen)| last_seen.elapsed().as_secs() < 5);

        // 2. Sync selection if in Lobby
        if let AppState::LocalLobby {
            selected_idx,
            selected_id,
        } = state
        {
            let count = self.discovered_hosts.len();
            if count == 0 {
                (0, None)
            } else {
                // Identity-aware selection synchronization
                if let Some(id) = *selected_id {
                    if let Some(new_pos) = self
                        .discovered_hosts
                        .iter()
                        .position(|(h, _)| h.instance_id == id)
                    {
                        (new_pos, Some(id))
                    } else {
                        let new_idx = (*selected_idx).min(count - 1);
                        let new_id = self
                            .discovered_hosts
                            .get(new_idx)
                            .map(|(h, _)| h.instance_id);
                        (new_idx, new_id)
                    }
                } else {
                    (0, self.discovered_hosts.first().map(|(h, _)| h.instance_id))
                }
            }
        } else {
            (0, None)
        }
    }

    /// Resets transient metadata for a new game session.
    pub fn reset_session_state(&mut self) {
        self.peer_name = "Opponent".to_string();
        self.latency = None;
        self.chat_history.clear();
        self.chat_input.clear();
        self.show_chat = false;
    }
}

impl Default for NetworkContext {
    /// Creates a default `NetworkContext` using `NetworkContext::new()`.
    fn default() -> Self {
        Self::new()
    }
}

/// User interface configuration settings.
pub struct UiSettings {
    /// Whether to display AI analysis (move evaluations) in the UI.
    pub show_analysis: bool,
    /// Whether to display the game move history in the UI.
    pub show_history: bool,
}

/// Reasons why a game session might end.
#[derive(Debug, PartialEq, Clone)]
pub enum GameOverReason {
    /// A player has won the game.
    Win(Player),
    /// The game ended in a draw with no remaining valid moves.
    Stalemate,
    /// The game ended due to a network communication error.
    NetworkError(String),
    /// The game ended because a state desynchronization was detected between players.
    Desync,
}

/// Represents the high-level state of the application state machine.
#[derive(Debug, PartialEq, Clone)]
pub enum AppState {
    /// The main menu screen.
    Menu {
        /// The currently selected menu option index.
        selected_idx: usize,
    },
    /// The difficulty selection screen for single-player games.
    DifficultySelection {
        /// The currently selected difficulty level index.
        selected_idx: usize,
    },
    /// The board size selection screen.
    SizeSelection {
        /// The currently selected board size index.
        selected_idx: usize,
    },
    /// The player color/turn selection screen.
    PlayerSelection {
        /// The currently selected player option index.
        selected_idx: usize,
    },
    /// The local LAN lobby screen for remote play.
    LocalLobby {
        /// The currently selected host index in the list.
        selected_idx: usize,
        /// The ID of the currently selected host, used for identity-aware tracking.
        selected_id: Option<u64>,
    },
    /// Screen for entering the host's name when starting a remote game.
    RemoteNameInput,
    /// Screen for entering the guest's name when joining a remote game.
    GuestNameInput(std::net::SocketAddr),
    /// The active gameplay state.
    Playing,
    /// State when the user is typing a message into the chat.
    ChatInput,
    /// The game over screen showing the final result.
    GameOver(GameOverReason),
    /// An error screen displaying a critical application error.
    Error(String),
}

/// Result of a background AI analysis search.
struct AnalyticsResult {
    /// Column scores evaluated by the engine.
    scores: Vec<Option<i32>>,
    /// The hash of the board state for which these results were calculated.
    state_hash: u128,
}

/// Result of an asynchronous AI move search.
struct AiMoveResult {
    /// The column chosen by the AI.
    col: u32,
    /// The hash of the board state for which this move was decided.
    state_hash: u128,
}

/// Central application state and controller.
///
/// This struct manages the high-level state machine (`AppState`),
/// holds the current game session (if any), and coordinates the network layer.
/// It acts as the "Controller" in the MVC pattern, receiving input from `input.rs`
/// and directing rendering via `ui.rs`.
pub struct App {
    /// The current high-level state of the application.
    pub state: AppState,
    /// The active game session, if any.
    pub session: Option<UiGameSession>,
    /// Networking context for remote play, initialized as needed.
    pub network: Option<NetworkContext>,
    /// Global application UI settings.
    pub settings: UiSettings,
    /// Unique ID used to identify this client in LAN discovery to prevent self-discovery.
    pub instance_id: u64,
    /// The game mode to be used for the next session.
    pub pending_mode: GameMode,
    /// The difficulty level to be used for the next session.
    pub pending_diff: Difficulty,
    /// The index of the board size to be used for the next session.
    pub pending_size_idx: usize,
    /// The player color to be assigned to the local user for the next session.
    pub pending_player: Player,

    /// Channel for sending requests to the background analytics task.
    analytics_tx: tokio::sync::mpsc::Sender<AnalyticsResult>,
    /// Channel for receiving results from the background analytics task.
    analytics_rx: tokio::sync::mpsc::Receiver<AnalyticsResult>,
    /// Channel for sending requests to the background AI search task.
    ai_move_tx: tokio::sync::mpsc::Sender<AiMoveResult>,
    /// Channel for receiving results from the background AI search task.
    ai_move_rx: tokio::sync::mpsc::Receiver<AiMoveResult>,
}

impl App {
    /// Creates a new `App` instance with default state and initialized communication channels.
    fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let (ai_tx, ai_rx) = tokio::sync::mpsc::channel(8);
        Self {
            state: AppState::Menu { selected_idx: 0 },
            session: None,
            network: None,
            settings: UiSettings {
                show_analysis: true,
                show_history: true,
            },
            instance_id: rand::random(),
            pending_mode: GameMode::Single,
            pending_diff: Difficulty::Medium,
            pending_size_idx: 0,
            pending_player: Player::Red,
            analytics_tx: tx,
            analytics_rx: rx,
            ai_move_tx: ai_tx,
            ai_move_rx: ai_rx,
        }
    }

    /// Triggers background search for move evaluations.
    ///
    /// # Panics
    ///
    /// Panics if the analytics transposition table mutex is poisoned.
    pub fn trigger_analytics(&mut self) {
        let Some(session) = &mut self.session else {
            return;
        };

        let board = session.core.board.clone();
        let player = session.core.current_player;
        let state_hash = board.state_hash();
        let engine = session.core.analytics_engine.clone();
        let tx = self.analytics_tx.clone();

        tracing::debug!(
            "Triggering background analytics search at state hash: {:016X}",
            state_hash
        );

        tokio::spawn(async move {
            let scores = {
                let mut engine_lock = engine.lock().expect("Analytics Engine Mutex poisoned");
                engine_lock.get_column_scores(&board.state, player, ANALYTICS_DEPTH)
            };
            let _ = tx.send(AnalyticsResult { scores, state_hash }).await;
        });
    }

    /// Triggers an asynchronous AI search for the next move.
    ///
    /// # Panics
    ///
    /// Panics if the AI search fails to return a valid move choice or if the AI
    /// transposition table mutex is poisoned.
    pub fn trigger_ai_move(&mut self) {
        let Some(session) = &mut self.session else {
            return;
        };

        if session.core.is_ai_thinking {
            return;
        }

        let board = session.core.board.clone();
        let ai_player = session.core.local_player.other();
        let depth = session.core.difficulty.depth();
        let state_hash = board.state_hash();
        let engine = session.core.ai_engine.clone();
        let tx = self.ai_move_tx.clone();

        session.core.is_ai_thinking = true;
        tracing::debug!(
            "Triggering AI move search at depth {} and state hash: {:016X}",
            depth,
            state_hash
        );

        tokio::spawn(async move {
            let col = {
                let mut engine_lock = engine.lock().expect("AI Engine Mutex poisoned");
                engine_lock.find_best_move(&board.state, ai_player, depth, true)
            };
            // We expect the AI to find a move because find_best_move is only triggered on valid, non-terminal turns.
            let _ = tx
                .send(AiMoveResult {
                    col: col
                        .expect("AI search failed to find a valid move on a non-terminal state"),
                    state_hash,
                })
                .await;
        });
    }

    /// Initializes a new game session using the current `pending` configuration.
    /// Resets analytics and AI search channels to ensure no stale results carry over.
    pub fn start_session(&mut self) {
        // Hard reset channels to discard any pending background tasks from previous sessions
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        self.analytics_tx = tx;
        self.analytics_rx = rx;

        let (ai_tx, ai_rx) = tokio::sync::mpsc::channel(8);
        self.ai_move_tx = ai_tx;
        self.ai_move_rx = ai_rx;

        let config = &BOARD_SIZES[self.pending_size_idx];
        tracing::info!(
            "Starting new session: mode={:?}, diff={:?}, player={:?}, size={}x{}",
            self.pending_mode,
            self.pending_diff,
            self.pending_player,
            config.size.cols,
            config.size.rows
        );

        let session = UiGameSession::new(
            config.size.cols,
            config.size.rows,
            self.pending_mode,
            self.pending_diff,
            self.pending_player,
        );
        self.session = Some(session);
        self.trigger_analytics();
        self.state = AppState::Playing;
    }

    /// Unified cleanup and state reset to return the application to the main menu.
    /// Handles notifying peers, stopping networking, and resetting session metadata.
    pub fn return_to_menu(&mut self) {
        tracing::info!("Returning to menu");
        if let Some(session) = &self.session
            && session.core.game_mode == GameMode::Remote
            && let Some(net) = &self.network
        {
            net.manager
                .try_send(NetworkCommand::Send(RemoteMessage::Quit));
        }

        if let Some(net) = &mut self.network {
            net.manager.try_send(NetworkCommand::StopDiscovery);
            net.is_hosting = false;
            net.reset_session_state();
        }

        self.session = None;
        self.state = AppState::Menu { selected_idx: 0 };
    }

    /// Main update loop called once per tick.
    ///
    /// Responsibilities:
    /// 1. Process pending network events (moves, chat, discovery).
    /// 2. Prune stale network hosts and sync lobby selection.
    /// 3. Process background task results (analytics and AI moves).
    /// 4. Update cosmetic animations (falling pieces).
    /// 5. Execute AI moves if applicable.
    fn update(&mut self) {
        self.handle_network_events();

        // 2. Network Maintenance
        if let Some(net) = &mut self.network {
            let (new_idx, new_id) = net.update(&self.state);
            if let AppState::LocalLobby {
                ref mut selected_idx,
                ref mut selected_id,
            } = self.state
            {
                *selected_idx = new_idx;
                *selected_id = new_id;
            }
        }

        self.process_analytics_results();
        self.process_ai_moves();
        self.update_animations();
        self.check_ai_turn();
    }

    /// Pulls all available events from the network manager and processes them sequentially.
    fn handle_network_events(&mut self) {
        let mut events = Vec::new();
        if let Some(net) = &mut self.network {
            while let Some(event) = net.manager.try_recv() {
                events.push(event);
            }
        }

        for event in events {
            self.process_network_event(event);
        }
    }

    /// Dispatches a single `NetworkEvent` to its corresponding handler method.
    fn process_network_event(&mut self, event: NetworkEvent) {
        tracing::debug!("Processing network event: {:?}", event);
        match event {
            NetworkEvent::HostDiscovered(info) => self.handle_host_discovered(info),
            NetworkEvent::PeerConnected(peer_name) => self.handle_peer_connected(&peer_name),
            NetworkEvent::Ready {
                peer_name,
                board_size,
                peer_player,
            } => self.handle_ready(peer_player, board_size, &peer_name),
            NetworkEvent::MoveReceived { col, state_hash } => {
                self.make_move(col, Some(state_hash));
            }
            NetworkEvent::ChatReceived(msg) => self.handle_chat_received(&msg),
            NetworkEvent::Latency(rtt) => {
                if let Some(net) = &mut self.network {
                    net.latency = Some(rtt);
                }
            }
            NetworkEvent::Disconnected(reason) => self.handle_disconnect(reason),
            NetworkEvent::Error(err) => self.handle_network_error(err),
        }
    }

    /// Updates or adds a host to the list of discovered LAN hosts.
    /// Prevents self-discovery by checking the `instance_id`.
    fn handle_host_discovered(&mut self, mut info: DiscoveryInfo) {
        if info.instance_id == self.instance_id {
            return;
        }
        info.local_name = sanitize_string(&info.local_name);
        if let Some(net) = &mut self.network {
            if let Some(pos) = net
                .discovered_hosts
                .iter()
                .position(|(h, _)| h.instance_id == info.instance_id)
            {
                // Identity-based update: Refresh entire info (latest IP wins) and timestamp
                net.discovered_hosts[pos].0 = info;
                net.discovered_hosts[pos].1 = std::time::Instant::now();
            } else {
                net.discovered_hosts.push((info, std::time::Instant::now()));
                // If we're in the lobby and have no selection, select this new host
                if let AppState::LocalLobby {
                    ref mut selected_id,
                    ..
                } = self.state
                    && selected_id.is_none()
                {
                    *selected_id = net.discovered_hosts.last().map(|(h, _)| h.instance_id);
                }
            }
        }
    }

    /// Handles a peer successfully connecting to our host.
    /// Transitions the application to the game session once a peer joins.
    fn handle_peer_connected(&mut self, peer_name: &str) {
        if matches!(self.state, AppState::LocalLobby { .. }) {
            if let Some(net) = &mut self.network {
                net.is_hosting = false;
                net.peer_name = sanitize_string(peer_name);
            }
            self.pending_mode = GameMode::Remote;
            self.start_session();
        }
    }

    /// Handles the "ready" signal from a peer, which includes their preferred board size and player color.
    /// Resets the local session to match the peer's configuration and starts the game.
    fn handle_ready(&mut self, peer_player: Player, board_size: (u32, u32), peer_name: &str) {
        if matches!(self.state, AppState::LocalLobby { .. }) {
            if let Some(net) = &mut self.network {
                net.peer_name = sanitize_string(peer_name);
                net.is_hosting = false;
            }
            self.pending_mode = GameMode::Remote;
            self.pending_player = peer_player.other();
            self.pending_size_idx = BOARD_SIZES
                .iter()
                .position(|s| s.size.cols == board_size.0 && s.size.rows == board_size.1)
                .unwrap_or(0);
            self.start_session();
        }
    }

    /// Handles an incoming chat message.
    /// Sanitizes the message and peer name, updates chat history, and ensures the chat is visible.
    fn handle_chat_received(&mut self, msg: &str) {
        if let Some(net) = &mut self.network {
            let sanitized_msg = sanitize_string(msg);
            let display_name = if net.peer_name.chars().count() > COMPACT_NAME_LIMIT {
                let mut s: String = net
                    .peer_name
                    .chars()
                    .take(COMPACT_NAME_LIMIT.saturating_sub(1))
                    .collect();
                s.push('…');
                s
            } else {
                net.peer_name.clone()
            };
            net.chat_history
                .push_back(format!("[{display_name}] {sanitized_msg}"));
            if net.chat_history.len() > CHAT_HISTORY_LIMIT {
                net.chat_history.pop_front();
            }
            net.show_chat = true;
        }
    }

    /// Handles a network disconnection event.
    /// Transitions to an error state or game over state depending on the current application state.
    fn handle_disconnect(&mut self, reason: String) {
        if matches!(self.state, AppState::Playing | AppState::ChatInput) {
            self.state = AppState::GameOver(GameOverReason::NetworkError(reason));
        } else if matches!(
            self.state,
            AppState::LocalLobby { .. } | AppState::GuestNameInput(_) | AppState::RemoteNameInput
        ) {
            self.state = AppState::Error(reason);
        }
    }

    /// Handles critical network errors.
    /// Logs the error, updates application state, and stops any active hosting.
    fn handle_network_error(&mut self, err: String) {
        error!("Network error: {}", err);
        self.state = AppState::Error(err);
        if let Some(net) = &mut self.network {
            net.is_hosting = false;
        }
    }

    /// Pulls and applies background analytics results from the receiver channel.
    /// Only updates the session if the result corresponds to the current board state hash.
    fn process_analytics_results(&mut self) {
        while let Ok(result) = self.analytics_rx.try_recv() {
            tracing::debug!(
                "Received analytics results for state hash: {:016X}",
                result.state_hash
            );
            if let Some(session) = &mut self.session {
                let current_hash = session.core.board.state_hash();
                if result.state_hash == current_hash {
                    session.core.cached_scores = result.scores;
                    session.core.is_stale_analytics = false;
                }
            }
        }
    }

    /// Pulls and applies background AI move results from the receiver channel.
    /// Validates the state hash before applying the move to prevent stale moves from being made.
    fn process_ai_moves(&mut self) {
        let mut ai_move_to_apply = None;
        while let Ok(result) = self.ai_move_rx.try_recv() {
            tracing::debug!(
                "Received AI move result: col {} for state hash: {:016X}",
                result.col,
                result.state_hash
            );
            if let Some(session) = &mut self.session {
                let current_hash = session.core.board.state_hash();
                if result.state_hash == current_hash {
                    ai_move_to_apply = Some(result.col);
                }
                session.core.is_ai_thinking = false;
            }
        }

        if let Some(col) = ai_move_to_apply {
            self.make_move(col, None);
        }
    }

    /// Updates the state of cosmetic animations (e.g., falling pieces).
    /// Advances positions and handles floor collisions for all active `FallingPiece` objects.
    fn update_animations(&mut self) {
        use crate::ui_config::{
            BOUNCE_COEFFICIENT, GRAVITY, MIN_BOUNCE_VELOCITY, TERMINAL_VELOCITY,
        };
        if let Some(session) = &mut self.session {
            // Advance cosmetic animations
            if let Some(front) = session.falling_pieces.front_mut() {
                let now = std::time::Instant::now();
                let dt = now.duration_since(front.last_update).as_secs_f64();
                front.last_update = now;

                // Physics: v = v + g*dt
                front.velocity = (front.velocity + GRAVITY * dt).min(TERMINAL_VELOCITY);
                // Physics: y = y - v*dt
                front.current_y -= front.velocity * dt;

                let floor = f64::from(front.target_row);
                if front.current_y <= floor && front.velocity > 0.0 {
                    // Collision with floor: reflect velocity
                    let reflected_v = -front.velocity * BOUNCE_COEFFICIENT;

                    if reflected_v.abs() < MIN_BOUNCE_VELOCITY {
                        // Velocity too low, piece settles
                        tracing::debug!(
                            "Animation settled for player {:?} in column {}, target row {}",
                            front.player,
                            front.col,
                            front.target_row
                        );
                        session.falling_pieces.pop_front();
                    } else {
                        // Bounce!
                        front.velocity = reflected_v;
                        front.current_y = floor;
                    }
                }
            }
        }
    }

    /// Checks if it is the AI's turn and triggers a search for the next move if necessary.
    /// This only happens in single-player mode when no animations are currently running.
    fn check_ai_turn(&mut self) {
        if let Some(session) = &mut self.session {
            // Trigger AI move if it's the AI's turn and game is active
            if session.core.game_mode == GameMode::Single
                && self.state == AppState::Playing
                && session.core.current_player != session.core.local_player
                && session.falling_pieces.is_empty()
                && !session.core.is_ai_thinking
            {
                self.trigger_ai_move();
            }
        }
    }

    /// High-level entry point for move requests from UI or Network.
    ///
    /// Validates if a move is currently permitted based on application state
    /// and turn rules, then delegates to `execute_move` for state mutation.
    pub fn make_move(&mut self, col: u32, expected_hash: Option<u128>) {
        if !matches!(self.state, AppState::Playing | AppState::ChatInput) {
            return;
        }

        let Some(session) = &self.session else {
            return;
        };

        // Remote Turn Guard: Local moves (hash=None) are only allowed on the local player's turn.
        if session.core.game_mode == GameMode::Remote
            && expected_hash.is_none()
            && session.core.current_player != session.core.local_player
        {
            return;
        }

        if let Err(e) = self.execute_move(col, expected_hash) {
            error!("Move execution failed: {e}");
        } else if let AppState::Playing | AppState::ChatInput = self.state {
            // If the move succeeded and didn't end the game, refresh analytics.
            self.trigger_analytics();
        }
    }

    /// Performs an atomic move transaction, converging all system components.
    ///
    /// This method:
    /// 1. Mutates logical board state.
    /// 2. Handles network synchronization for remote moves.
    /// 3. Resolves terminal game states (Win/Stalemate/Desync).
    /// 4. Triggers cosmetic animations for the dropped piece.
    ///
    /// # Errors
    /// Returns `Connect4Error` if the board mutation itself fails (e.g. column full).
    fn execute_move(&mut self, col: u32, expected_hash: Option<u128>) -> Result<u32> {
        use connect4::game_session::MoveResult;

        let session = self
            .session
            .as_mut()
            .expect("execute_move called without active session");

        let player = session.core.current_player;

        // 1. Perform Logical Mutation
        let move_res = session.core.execute_move(col)?;

        // 2. Inbound Desync Validation (Application Layer)
        // We validate AFTER applying the move locally to ensure we compare symmetric state hashes.
        if let Some(expected) = expected_hash {
            let current_hash = session.core.board.state_hash();
            if expected != current_hash {
                tracing::error!(
                    "Desync detected! Expected {:X}, got {:X}",
                    expected,
                    current_hash
                );
                self.state = AppState::GameOver(GameOverReason::Desync);
                return Err(color_eyre::eyre::eyre!("Desync detected"));
            }
        }

        // 3. Outbound Network Sync
        if session.core.game_mode == GameMode::Remote
            && player == session.core.local_player
            && let Some(net) = &self.network
        {
            net.manager
                .try_send(NetworkCommand::Send(RemoteMessage::Move {
                    col,
                    state_hash: session.core.board.state_hash(),
                }));
        }

        // 4. Process Result & Terminal State Resolution
        let row = match move_res {
            MoveResult::Win { row, winner } => {
                tracing::info!("Game over: Winner is {:?}", winner);
                self.state = AppState::GameOver(GameOverReason::Win(winner));
                row
            }
            MoveResult::Draw { row } => {
                tracing::info!("Game over: Stalemate");
                self.state = AppState::GameOver(GameOverReason::Stalemate);
                row
            }
            MoveResult::Success { row, .. } => row,
        };

        // 5. Cosmetic Animation Trigger
        tracing::debug!(
            "Triggering animation for player {:?} in column {}, target row {}",
            player,
            col,
            row
        );
        session.falling_pieces.push_back(FallingPiece {
            col,
            target_row: row,
            current_y: f64::from(session.core.board.rows()),
            velocity: INITIAL_VELOCITY,
            last_update: std::time::Instant::now(),
            player,
        });

        Ok(row)
    }
}

/// The main entry point for the Connect 4 TUI application.
/// Initializes logging, sets up the Tokio runtime, and drives the application state machine.
fn main() -> Result<()> {
    let cli = Cli::parse();
    let _log_guard = init_logging(&cli);
    tracing::info!("Starting TUI application");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _guard = rt.enter();
    color_eyre::install()?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    let mut app = App::new();
    let _res = run_app(&mut terminal, &mut app);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    drop(rt);
    tracing::info!("TUI application shut down gracefully");
    Ok(())
}

/// The core application loop that handles rendering and event polling.
/// Returns once the application is requested to exit.
fn run_app<B: io::Write + Send + Sync + 'static>(
    terminal: &mut Terminal<CrosstermBackend<B>>,
    app: &mut App,
) -> Result<()> {
    let tick_rate = std::time::Duration::from_millis(TICK_RATE_MS);
    loop {
        terminal.draw(|f| ui::render(f, app))?;
        if event::poll(tick_rate)?
            && let Event::Key(key) = event::read()?
        {
            // Filter out 'Release' events to prevent double-input on Windows.
            // We process 'Press' and 'Repeat' to ensure held keys (like arrows or backspace)
            // behave naturally across all platforms.
            if key.kind != event::KeyEventKind::Release && input::handle_key_event(app, key) {
                return Ok(());
            }
        }
        app.update();
    }
}

/// Sanitizes a string for use in the TUI.
///
/// Filters out control characters and trims leading/trailing whitespace.
/// This prevents layout-breaking characters (like newlines or tabs)
/// from entering the application state from either user input or network data.
#[must_use]
pub fn sanitize_string(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui_config::MAX_NAME_LEN;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn test_sanitize_string_removes_controls() {
        let input = "Hello\nWorld\t!\r";
        assert_eq!(sanitize_string(input), "HelloWorld!");
    }

    #[test]
    fn test_sanitize_string_trims() {
        let input = "  Clean Name   ";
        assert_eq!(sanitize_string(input), "Clean Name");
    }

    #[test]
    fn test_sanitize_string_handles_empty() {
        let input = "   \n\r   ";
        assert_eq!(sanitize_string(input), "");
    }

    #[tokio::test]
    async fn test_app_initialization() {
        let app = App::new();
        assert!(
            matches!(app.state, AppState::Menu { selected_idx: 0 }),
            "App must start in Menu state with first option selected"
        );
    }

    #[tokio::test]
    async fn test_network_context_reset_bug() {
        let mut net = NetworkContext::new();
        net.peer_name = "Alice".to_string();
        net.latency = Some(42);
        net.chat_history.push_back("Hello".to_string());
        net.show_chat = true;
        net.chat_input = "Hi".to_string();

        net.reset_session_state();
        assert_eq!(net.peer_name, "Opponent");
        assert!(net.latency.is_none());
        assert!(net.chat_history.is_empty());
        assert!(!net.show_chat);
        assert!(net.chat_input.is_empty());
    }

    #[tokio::test]
    async fn test_peer_name_reset_bug() {
        let mut app = App::new();
        app.network = Some(NetworkContext::new());
        app.pending_mode = GameMode::Remote;

        // Simulate handshake having set the name
        app.network.as_mut().unwrap().peer_name = "Alice".to_string();

        // Start session
        app.start_session();

        // Should keep Alice
        assert_eq!(app.network.as_ref().unwrap().peer_name, "Alice");
    }

    #[tokio::test]
    async fn test_name_input_limit_bug() {
        let mut app = App::new();
        app.network = Some(NetworkContext::new());
        app.network.as_mut().unwrap().local_name.clear();
        app.state = AppState::RemoteNameInput;

        let long_name = "This name is definitely way too long for our game";
        for c in long_name.chars() {
            input::handle_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char(c), KeyModifiers::empty()),
            );
        }

        let name = &app.network.as_ref().unwrap().local_name;
        assert!(name.len() <= MAX_NAME_LEN);
        assert_eq!(name, &long_name[..MAX_NAME_LEN]);
    }

    #[tokio::test]
    async fn test_chat_toggle_bug() {
        let mut app = App::new();
        app.network = Some(NetworkContext::new());
        app.session = Some(UiGameSession::new(
            7,
            6,
            GameMode::Remote,
            Difficulty::Medium,
            Player::Red,
        ));
        app.state = AppState::Playing;
        app.network.as_mut().unwrap().show_chat = false;

        // '/' should open chat input
        input::handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('/'), KeyModifiers::empty()),
        );
        assert_eq!(app.state, AppState::ChatInput);
        assert!(app.network.as_ref().unwrap().show_chat);

        // Escape back to playing
        input::handle_key_event(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()));
        assert_eq!(app.state, AppState::Playing);

        // Alt+C should toggle visibility OFF
        app.network.as_mut().unwrap().show_chat = true;
        input::handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::ALT),
        );
        assert_eq!(app.state, AppState::Playing);
        assert!(!app.network.as_ref().unwrap().show_chat);
    }

    #[tokio::test]
    async fn test_restart_behavior_bug() {
        let mut app = App::new();
        app.pending_mode = GameMode::Single;
        app.pending_player = Player::Red;
        app.start_session();

        // Initially Red
        assert_eq!(app.session.as_ref().unwrap().core.local_player, Player::Red);

        // Simulate Game Over
        app.state = AppState::GameOver(GameOverReason::Stalemate);

        // Press 'S' to swap
        input::handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::empty()),
        );

        // Should now be Yellow
        assert_eq!(app.pending_player, Player::Yellow);
        assert_eq!(
            app.session.as_ref().unwrap().core.local_player,
            Player::Yellow
        );
        assert!(matches!(app.state, AppState::Playing));

        // Test Remote mode protection
        app.pending_mode = GameMode::Remote;
        app.start_session();
        app.state = AppState::GameOver(GameOverReason::Stalemate);

        // Press 'S'
        input::handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::empty()),
        );
        assert!(matches!(app.state, AppState::GameOver(_)));

        // Press 'r'
        input::handle_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('r'), KeyModifiers::empty()),
        );
        assert!(matches!(app.state, AppState::GameOver(_)));
    }

    #[tokio::test]
    async fn test_remote_move_sync_success() {
        let mut app = App::new();
        app.pending_size_idx = 0; // 7x6
        app.start_session();
        {
            let session = app.session.as_mut().unwrap();
            session.core.game_mode = GameMode::Remote;
            session.core.local_player = Player::Red;
        }

        // 1. Calculate what the hash SHOULD be after Yellow moves in column 0.
        // We do this by manually applying the move to a temporary board.
        let mut board = app.session.as_ref().unwrap().core.board.clone();
        board.drop_piece(0, Player::Red).unwrap(); // First Red move (local)
        app.execute_move(0, None).unwrap(); // Apply it to app

        let mut expected_board = board.clone();
        expected_board.drop_piece(1, Player::Yellow).unwrap(); // Then Yellow move (remote)
        let expected_hash = expected_board.state_hash();

        // 2. Execute the remote move (Yellow) with the CORRECT post-move hash.
        // Before the fix, this would have triggered a Desync because it compared
        // expected_hash (post-move) with board.state_hash() (pre-move).
        app.execute_move(1, Some(expected_hash))
            .expect("Move should succeed");

        // 3. Verify we are still playing and not desynced.
        assert_eq!(app.state, AppState::Playing);
        assert_eq!(
            app.session.as_ref().unwrap().core.board.state_hash(),
            expected_hash
        );
    }

    #[tokio::test]
    async fn test_desync_handling_bug() {
        let mut app = App::new();
        app.pending_size_idx = 0; // 7x6
        app.start_session();
        {
            let session = app.session.as_mut().unwrap();
            session.core.game_mode = GameMode::Remote;
            session.core.local_player = Player::Red;
        }

        // Human (Red) moves first
        app.execute_move(3, None).unwrap();

        // Execute the remote move (Yellow) with wrong hash
        let _ = app.execute_move(0, Some(99999));

        assert!(matches!(
            app.state,
            AppState::GameOver(GameOverReason::Desync)
        ));
    }

    #[tokio::test]
    async fn test_async_analytics_update() {
        let mut app = App::new();
        app.state = AppState::Playing;
        app.pending_size_idx = 0; // 7x6
        app.start_session();

        // Wait for initial analytics
        for _ in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            app.update();
            if !app.session.as_ref().unwrap().core.is_stale_analytics {
                break;
            }
        }

        // Force a move
        app.make_move(3, None);

        let session = app.session.as_ref().unwrap();
        // Immediately after move, flag MUST be true, but scores are NOT cleared anymore
        assert!(session.core.is_stale_analytics);

        // Poll update a few times with small sleeps to allow background task to finish
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            app.update();
            let session = app.session.as_ref().unwrap();
            if !session.core.is_stale_analytics {
                break;
            }
        }

        let session = app.session.as_ref().unwrap();
        assert!(
            !session.core.is_stale_analytics,
            "Analytics flag was never cleared"
        );
        assert!(session.core.cached_scores.iter().any(Option::is_some));
    }
}

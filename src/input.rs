#![cfg(feature = "tui")]

//! # Input Handling Module
//!
//! This module provides the logic for processing keyboard events within the Terminal User Interface (TUI).
//! It translates raw `crossterm` key events into semantic actions that update the application state
//! or trigger game-related logic.
//!
//! ## Architecture
//!
//! The module follows a state-based delegation pattern:
//! - [`handle_key_event`] serves as the central dispatcher.
//! - State-specific handlers (e.g., [`handle_menu_key`], [`handle_playing_key`]) manage input based on the current [`AppState`].
//! - Generic helpers (e.g., [`handle_vertical_navigation`], [`handle_generic_text_input`]) provide reusable logic for common UI patterns like list navigation and text entry.
//!
//! This separation ensures that input logic is decoupled from rendering and core game engine mechanics.

use crate::App;
use crate::ui_config::{BOARD_SIZES, CHAT_HISTORY_LIMIT, MAX_CHAT_LEN, MAX_NAME_LEN};
use crate::{AppState, NetworkContext};
use connect4::network::{NetworkCommand, RemoteMessage};
use connect4::types::{Difficulty, GameMode, Player};
use crossterm::event::{self, KeyCode};
use tracing::info;

/// The main entry point for processing keyboard input.
///
/// This function captures the current [`AppState`], dispatches the key event to the
/// appropriate sub-handler, and performs post-processing tasks like logging state transitions.
///
/// ### Arguments
/// * `app` - A mutable reference to the main application state.
/// * `key` - The raw [`event::KeyEvent`] received from the terminal.
///
/// ### Returns
/// * `bool` - Returns `true` if the input indicates that the application should exit; otherwise, `false`.
pub fn handle_key_event(app: &mut App, key: event::KeyEvent) -> bool {
    let old_state = app.state.clone();
    let res = match app.state.clone() {
        AppState::Menu { selected_idx } => handle_menu_key(app, key, selected_idx),
        AppState::DifficultySelection { selected_idx } => {
            handle_difficulty_key(app, key, selected_idx)
        }
        AppState::PlayerSelection { selected_idx } => {
            handle_player_selection_key(app, key, selected_idx)
        }
        AppState::SizeSelection { selected_idx } => {
            handle_size_selection_key(app, key, selected_idx)
        }
        AppState::LocalLobby {
            selected_idx,
            selected_id,
        } => handle_local_lobby_key(app, key, selected_idx, selected_id),
        AppState::RemoteNameInput | AppState::GuestNameInput(_) => handle_name_input_key(app, key),
        AppState::Playing => handle_playing_key(app, key),
        AppState::ChatInput => handle_chat_input_key(app, key),
        AppState::GameOver(_) => handle_game_over_key(app, key),
        AppState::Error(_) => handle_error_key(app, key),
    };

    // Log state transitions for debugging and observability
    if app.state != old_state {
        info!("State transition: {:?} -> {:?}", old_state, app.state);
    }
    res
}

// ========================================================================================
// GENERIC INTERACTION HELPERS
// ========================================================================================

/// Calculates a new selection index based on vertical navigation keys.
///
/// This helper abstracts the common logic of incrementing or decrementing a selection
/// index while staying within the bounds of a list.
///
/// ### Arguments
/// * `key` - The key event to process. Responds to [`KeyCode::Up`] and [`KeyCode::Down`].
/// * `current` - The currently selected index.
/// * `count` - The total number of items in the list.
///
/// ### Returns
/// * `usize` - The new index after navigation.
fn handle_vertical_navigation(key: event::KeyEvent, current: usize, count: usize) -> usize {
    match key.code {
        KeyCode::Up => current.saturating_sub(1),
        KeyCode::Down => {
            if current < count.saturating_sub(1) {
                current + 1
            } else {
                current
            }
        }
        _ => current,
    }
}

/// Updates a string buffer by processing alphanumeric and backspace input.
///
/// This helper is used for capturing text input (e.g., player names or chat messages)
/// while enforcing a maximum character limit.
///
/// ### Arguments
/// * `key` - The key event to process.
/// * `buffer` - The mutable string buffer to update.
/// * `limit` - The maximum number of characters allowed in the buffer.
fn handle_generic_text_input(key: event::KeyEvent, buffer: &mut String, limit: usize) {
    match key.code {
        KeyCode::Char(c) if !c.is_control() && buffer.chars().count() < limit => buffer.push(c),
        KeyCode::Backspace => {
            buffer.pop();
        }
        _ => {}
    }
}

// ========================================================================================
// STATE-SPECIFIC HANDLERS
// ========================================================================================

/// Processes input for the main menu screen.
///
/// In this state, the user navigates between game modes (Single Player, Local Multiplayer, Remote).
/// Pressing `Enter` selects the mode and transitions to the next configuration step.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
/// * `selected_idx` - The currently highlighted menu item.
///
/// ### Returns
/// * `bool` - `true` if the app should exit.
fn handle_menu_key(app: &mut App, key: event::KeyEvent, selected_idx: usize) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Down => {
            app.state = AppState::Menu {
                selected_idx: handle_vertical_navigation(key, selected_idx, GameMode::ALL.len()),
            };
        }
        KeyCode::Enter => {
            // Selecting a mode starts the multi-step configuration wizard
            app.pending_mode = GameMode::ALL[selected_idx];
            tracing::info!("Menu selection: {:?}", app.pending_mode);
            match app.pending_mode {
                GameMode::Single => {
                    app.state = AppState::DifficultySelection {
                        selected_idx: Difficulty::Medium.index(),
                    }
                }
                GameMode::Remote => {
                    // Ensure networking is initialized when entering the remote flow
                    if app.network.is_none() {
                        app.network = Some(NetworkContext::new());
                    }
                    app.state = AppState::LocalLobby {
                        selected_idx: 0,
                        selected_id: None,
                    };
                }
                GameMode::LocalTwo => app.state = AppState::SizeSelection { selected_idx: 0 },
            }
        }
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => return true, // Signal application exit
        _ => {}
    }
    false
}

/// Processes input for the LAN multiplayer lobby.
///
/// This state manages discovery of local hosts and allows the user to:
/// - Navigate and join a discovered host.
/// - Manually refresh the host list (`R`).
/// - Initiate a new hosting flow (`H`).
/// - Stop hosting (`S`).
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
/// * `selected_idx` - The currently highlighted host in the lobby list.
/// * `_selected_id` - The unique identifier of the selected host (used for sticky selection).
fn handle_local_lobby_key(
    app: &mut App,
    key: event::KeyEvent,
    selected_idx: usize,
    _selected_id: Option<u64>,
) -> bool {
    let Some(net) = &mut app.network else {
        app.state = AppState::Menu {
            selected_idx: GameMode::Single.index(),
        };
        return false;
    };

    match key.code {
        KeyCode::Up | KeyCode::Down => {
            let count = net.discovered_hosts.len();
            if count > 0 {
                let new_idx = handle_vertical_navigation(key, selected_idx, count);
                // Sticky selection: track by ID rather than just index to handle list changes
                let new_id = net
                    .discovered_hosts
                    .get(new_idx)
                    .map(|(h, _)| h.instance_id);
                app.state = AppState::LocalLobby {
                    selected_idx: new_idx,
                    selected_id: new_id,
                };
            }
        }
        KeyCode::Enter => {
            // Attempt to join the selected host
            if let Some((host, _)) = net.discovered_hosts.get(selected_idx) {
                tracing::info!("Joining host: {} ({})", host.local_name, host.addr);
                app.state = AppState::GuestNameInput(host.addr);
            }
        }
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => {
            // Clean up hosting/discovery state when leaving the lobby
            if net.is_hosting {
                tracing::info!("Stopping hosting and leaving lobby");
                net.manager.try_send(NetworkCommand::StopDiscovery);
                net.is_hosting = false;
            }
            app.state = AppState::Menu {
                selected_idx: GameMode::Remote.index(),
            };
        }
        KeyCode::Char('r' | 'R') => {
            // Manual refresh of the discovery list
            tracing::info!("Refreshing lobby host list");
            net.discovered_hosts.clear();
            app.state = AppState::LocalLobby {
                selected_idx: 0,
                selected_id: None,
            };
        }
        KeyCode::Char('h' | 'H') => {
            // Start the hosting flow (select color -> select size -> enter name)
            if !net.is_hosting {
                tracing::info!("Starting host configuration flow");
                app.pending_mode = GameMode::Remote;
                app.state = AppState::PlayerSelection { selected_idx: 0 };
            }
        }
        KeyCode::Char('s' | 'S') => {
            // Stop hosting without leaving the lobby
            if net.is_hosting {
                tracing::info!("Stopping hosting");
                net.manager.try_send(NetworkCommand::StopDiscovery);
                net.is_hosting = false;
            }
        }
        _ => {}
    }
    false
}

/// Processes input for the AI difficulty selection screen.
///
/// Allows the user to select the skill level of the computer opponent.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
/// * `selected_idx` - The currently highlighted difficulty level.
fn handle_difficulty_key(app: &mut App, key: event::KeyEvent, selected_idx: usize) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Down => {
            app.state = AppState::DifficultySelection {
                selected_idx: handle_vertical_navigation(key, selected_idx, Difficulty::ALL.len()),
            };
        }
        KeyCode::Enter => {
            app.pending_diff = Difficulty::ALL[selected_idx];
            tracing::info!("Difficulty selected: {:?}", app.pending_diff);
            app.state = AppState::PlayerSelection { selected_idx: 0 };
        }
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => app.state = AppState::Menu { selected_idx: 0 },
        _ => {}
    }
    false
}

/// Processes input for choosing a player color (Red or Yellow).
///
/// In local games, this selects the starting player. In remote games, this selects the host's color.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
/// * `selected_idx` - The currently highlighted player option.
fn handle_player_selection_key(app: &mut App, key: event::KeyEvent, selected_idx: usize) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Down => {
            app.state = AppState::PlayerSelection {
                selected_idx: handle_vertical_navigation(key, selected_idx, Player::ALL.len()),
            };
        }
        KeyCode::Enter => {
            app.pending_player = Player::ALL[selected_idx];
            tracing::info!("Player color selected: {:?}", app.pending_player);
            app.state = AppState::SizeSelection { selected_idx: 0 };
        }
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => {
            if app.pending_mode == GameMode::Remote {
                app.state = AppState::LocalLobby {
                    selected_idx: 0,
                    selected_id: None,
                };
            } else {
                app.state = AppState::DifficultySelection {
                    selected_idx: Difficulty::Medium.index(),
                };
            }
        }
        _ => {}
    }
    false
}

/// Processes input for board dimension selection.
///
/// Users can choose from a set of predefined board sizes (e.g., standard 7x6).
/// After selection, it either starts the session (for local modes) or moves to name entry (for remote).
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
/// * `selected_idx` - The currently highlighted board size option.
fn handle_size_selection_key(app: &mut App, key: event::KeyEvent, selected_idx: usize) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Down => {
            app.state = AppState::SizeSelection {
                selected_idx: handle_vertical_navigation(key, selected_idx, BOARD_SIZES.len()),
            };
        }
        KeyCode::Enter => {
            app.pending_size_idx = selected_idx;
            let config = &BOARD_SIZES[selected_idx];
            tracing::info!(
                "Board size selected: {}x{}",
                config.size.cols,
                config.size.rows
            );
            if app.pending_mode == GameMode::Remote {
                app.state = AppState::RemoteNameInput;
            } else {
                // For local games, we have all settings and can start immediately
                app.start_session();
            }
        }
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => match app.pending_mode {
            GameMode::Single => app.state = AppState::PlayerSelection { selected_idx: 0 },
            GameMode::Remote => {
                app.state = AppState::LocalLobby {
                    selected_idx: 0,
                    selected_id: None,
                }
            }
            GameMode::LocalTwo => {
                app.state = AppState::Menu {
                    selected_idx: GameMode::LocalTwo.index(),
                }
            }
        },
        _ => {}
    }
    false
}

/// Processes alphanumeric input for player names during remote game setup.
///
/// This handler is used both when hosting (entering your own name) and when joining
/// (entering your name before connecting to a host).
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
///
/// ### Returns
/// * `bool` - Always `false` as this state does not trigger app exit.
fn handle_name_input_key(app: &mut App, key: event::KeyEvent) -> bool {
    let current_state = app.state.clone();
    let Some(net) = &mut app.network else {
        app.state = AppState::Menu { selected_idx: 0 };
        return false;
    };

    match key.code {
        KeyCode::Enter => {
            let sanitized = crate::sanitize_string(&net.local_name);
            if !sanitized.is_empty() {
                net.local_name = sanitized;
                net.reset_session_state();
                match current_state {
                    AppState::RemoteNameInput => {
                        let config = &BOARD_SIZES[app.pending_size_idx];
                        net.manager.try_send(NetworkCommand::StartHosting {
                            instance_id: app.instance_id,
                            player_name: net.local_name.clone(),
                            board_size: (config.size.cols, config.size.rows),
                            host_player: app.pending_player,
                        });
                        net.is_hosting = true;
                    }
                    AppState::GuestNameInput(addr) => {
                        net.manager
                            .try_send(NetworkCommand::Connect(addr, net.local_name.clone()));
                    }
                    _ => unreachable!(
                        "Name input only occurs in RemoteNameInput or GuestNameInput states"
                    ),
                }
                app.state = AppState::LocalLobby {
                    selected_idx: 0,
                    selected_id: None,
                };
            }
        }
        KeyCode::Esc => {
            app.state = AppState::LocalLobby {
                selected_idx: 0,
                selected_id: None,
            };
        }
        _ => handle_generic_text_input(key, &mut net.local_name, MAX_NAME_LEN),
    }
    false
}

/// Processes input during active gameplay.
///
/// This is the most complex handler, managing:
/// - Column navigation (Left/Right arrows or Number keys).
/// - Dropping a piece (`Enter`).
/// - UI toggles (Alt+A for analysis, Alt+H for history, Alt+C for chat).
/// - Entering chat mode (`/`).
/// - Returning to the menu (`Esc` or `Q`).
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
///
/// ### Returns
/// * `bool` - Always `false`.
fn handle_playing_key(app: &mut App, key: event::KeyEvent) -> bool {
    let mut col_to_make = None;
    if let Some(session) = &mut app.session {
        match key.code {
            KeyCode::Left => session.selected_column = session.selected_column.saturating_sub(1),
            KeyCode::Right => {
                if session.selected_column < session.core.board.columns().saturating_sub(1) {
                    session.selected_column += 1;
                }
            }
            // Direct column selection via number keys (1-9)
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let col = u32::from(c as u8 - b'1');
                if col < session.core.board.columns() {
                    session.selected_column = col;
                }
            }
            KeyCode::Enter => {
                tracing::info!("Column selected via Enter: {}", session.selected_column);
                col_to_make = Some(session.selected_column);
            }
            // UI Intel Toggles
            KeyCode::Char('a' | 'A') if key.modifiers.contains(event::KeyModifiers::ALT) => {
                app.settings.show_analysis = !app.settings.show_analysis;
            }
            KeyCode::Char('h' | 'H') if key.modifiers.contains(event::KeyModifiers::ALT) => {
                app.settings.show_history = !app.settings.show_history;
            }
            KeyCode::Char('c' | 'C') if key.modifiers.contains(event::KeyModifiers::ALT) => {
                if let Some(net) = &mut app.network {
                    net.show_chat = !net.show_chat;
                }
            }
            KeyCode::Char('/') => {
                if session.core.game_mode == GameMode::Remote {
                    if let Some(net) = &mut app.network {
                        net.show_chat = true;
                    }
                    app.state = AppState::ChatInput;
                }
            }
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => {
                app.return_to_menu();
            }
            _ => {}
        }
    }

    if let Some(col) = col_to_make {
        app.make_move(col, None);
    }
    false
}

/// Processes alphanumeric input for the in-game chat system.
///
/// Pressing `Enter` sends the message to the peer in remote games.
/// Pressing `Esc` cancels the input and returns to regular gameplay.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
///
/// ### Returns
/// * `bool` - Always `false`.
fn handle_chat_input_key(app: &mut App, key: event::KeyEvent) -> bool {
    let Some(net) = &mut app.network else {
        app.state = AppState::Playing;
        return false;
    };
    match key.code {
        KeyCode::Enter => {
            let msg = crate::sanitize_string(&net.chat_input);
            if !msg.is_empty() {
                net.chat_history.push_back(format!("[You] {msg}"));
                if net.chat_history.len() > CHAT_HISTORY_LIMIT {
                    net.chat_history.pop_front();
                }
                net.manager
                    .try_send(NetworkCommand::Send(RemoteMessage::Chat(msg)));
                net.chat_input.clear();
            }
            app.state = AppState::Playing;
        }
        KeyCode::Esc => {
            net.chat_input.clear();
            app.state = AppState::Playing;
        }
        _ => handle_generic_text_input(key, &mut net.chat_input, MAX_CHAT_LEN),
    }
    false
}

/// Processes input when a game has concluded (Win, Loss, or Draw).
///
/// Allows the user to:
/// - Restart the game with same settings (`R`).
/// - Swap sides and restart (`S`).
/// - Return to the main menu (`Esc` or `Q`).
///
/// Note: Restarting is disabled for remote games to avoid synchronization issues.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
///
/// ### Returns
/// * `bool` - Always `false`.
fn handle_game_over_key(app: &mut App, key: event::KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('r' | 'R') => {
            // Restarting is only allowed in local/AI modes.
            // In remote mode, the session is terminated to prevent desync complexity.
            if let Some(session) = &app.session
                && session.core.game_mode != GameMode::Remote
            {
                app.start_session();
            }
        }
        KeyCode::Char('s' | 'S') => {
            // Swap sides and restart
            if let Some(session) = &app.session
                && session.core.game_mode != GameMode::Remote
            {
                app.pending_player = app.pending_player.other();
                app.start_session();
            }
        }
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => {
            app.return_to_menu();
        }
        _ => {}
    }
    false
}

/// Processes input for the error overlay.
///
/// Simply allows the user to dismiss the error and return to the main menu.
///
/// ### Arguments
/// * `app` - Main application instance.
/// * `key` - The key event.
///
/// ### Returns
/// * `bool` - Always `false`.
fn handle_error_key(app: &mut App, key: event::KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => {
            app.return_to_menu();
        }
        _ => {}
    }
    false
}

//! # UI Configuration
//!
//! This module defines the cosmetic and layout configuration for the TUI application.
//! It includes constants for colors, dimensions, animation physics, and helper traits for
//! displaying game entities.

#![cfg(feature = "tui")]

use connect4::types::{BoardSize, Difficulty, GameMode, Player};
use ratatui::style::Color;

// ========================================================================================
// UI LABELS & EXTENSIONS
// ========================================================================================

/// Extension trait to provide TUI-specific display labels for core enums.
///
/// This trait is implemented for `Player`, `Difficulty`, and `GameMode` to provide
/// user-friendly string representations for the UI.
pub trait ToLabel {
    /// Returns a static string slice representing the entity's label.
    fn label(&self) -> &'static str;
}

impl ToLabel for Player {
    fn label(&self) -> &'static str {
        match self {
            Self::Red => "RED (Moves First)",
            Self::Yellow => "YELLOW (Moves Second)",
        }
    }
}

impl ToLabel for Difficulty {
    fn label(&self) -> &'static str {
        match self {
            Self::Easy => "Easy",
            Self::Medium => "Medium",
            Self::Hard => "Hard",
            Self::Expert => "Expert",
            Self::Grandmaster => "Grandmaster",
        }
    }
}

impl ToLabel for GameMode {
    fn label(&self) -> &'static str {
        match self {
            Self::Single => "Single Player (vs CPU)",
            Self::LocalTwo => "Two Player (Local)",
            Self::Remote => "Remote Multiplayer (LAN)",
        }
    }
}

/// Configuration for a specific board size, including its dimensions and display label.
pub struct BoardSizeConfig {
    /// The dimensions of the board (columns and rows).
    pub size: BoardSize,
    /// The display label for this board size configuration (e.g., "Standard (7x6)").
    pub label: &'static str,
}

/// Available board size configurations for the game.
pub const BOARD_SIZES: [BoardSizeConfig; 3] = [
    BoardSizeConfig {
        size: BoardSize { cols: 7, rows: 6 },
        label: "Standard (7x6)",
    },
    BoardSizeConfig {
        size: BoardSize { cols: 8, rows: 7 },
        label: "Large (8x7)",
    },
    BoardSizeConfig {
        size: BoardSize { cols: 9, rows: 7 },
        label: "Giant (9x7)",
    },
];

// ========================================================================================
// UI LAYOUT CONSTANTS
// ========================================================================================

/// Fixed width of the right-hand sidebar.
pub const SIDEBAR_WIDTH: u16 = 32;

/// Maximum characters allowed for a player's name in the status header.
pub const HEADER_NAME_LIMIT: usize = 12;

/// Maximum characters allowed for a player's name in the stats rows.
pub const STATS_NAME_LIMIT: usize = 15;

/// Maximum characters allowed for a player's name in the predictions and history views.
pub const COMPACT_NAME_LIMIT: usize = 5;

/// Character width of a single board cell in the TUI.
pub const CELL_WIDTH: u16 = 8;

/// Character height of a single board cell in the TUI.
pub const CELL_HEIGHT: u16 = 4;

/// Length in characters of the intelligence pane's evaluation bar.
pub const EVAL_BAR_LEN: usize = 14;

/// Maximum character length of tactical evaluation bars in the intelligence pane.
pub const TACTICAL_BAR_MAX_LEN: usize = 14;

/// Refresh rate for the main application loop in milliseconds.
pub const TICK_RATE_MS: u64 = 16;

/// Maximum length of a player's name in the UI.
pub const MAX_NAME_LEN: usize = 10;

/// Maximum length of a chat message.
pub const MAX_CHAT_LEN: usize = 50;

/// Maximum number of chat messages to keep in history.
pub const CHAT_HISTORY_LIMIT: usize = 10;

/// Horizontal padding inside sidebar panels (both left and right).
pub const SIDEBAR_HPAD: usize = 1;

/// Minimum terminal width required to display the UI without overlap.
pub const MIN_UI_WIDTH: u16 = 105;

/// Minimum terminal height required to display the UI without overlap.
pub const MIN_UI_HEIGHT: u16 = 38;

/// Total range used for scaling the intelligence pane's evaluation gauge.
/// Set to 4000 so that a single threat (1000) shows as a 25% shift (75% total win prob),
/// providing an intuitive visual correlation between tactical score and evaluation.
pub const EVAL_SCALE: i32 = 4000;

// ========================================================================================
// ANIMATION CONFIGURATION
// ========================================================================================

/// Acceleration due to gravity in rows per second squared.
///
/// Calculated based on Earth gravity (9.8 m/s^2) and TUI scaling:
/// 1. A standard piece is 3.2 cm in diameter and occupies 3 lines in our TUI.
/// 2. Our cells are 4 lines high, meaning a physical cell height of 4.266 cm (0.04266 m).
/// 3. Gravity in rows/s^2 = 9.8 m/s^2 / 0.04266 m/row ≈ 229.7.
pub const GRAVITY: f64 = 230.0;

/// Initial downward velocity when a piece starts falling (rows per second).
/// Set to 0.0 to simulate a realistic drop from rest.
pub const INITIAL_VELOCITY: f64 = 0.0;

/// Maximum falling speed to prevent tunneling (rows per second).
///
/// Based on a terminal velocity of ~8.27 m/s for a plastic disc in air,
/// which corresponds to approximately 194 rows/s in our coordinate system.
pub const TERMINAL_VELOCITY: f64 = 194.0;

/// Percentage of velocity retained after a bounce (0.0 to 1.0).
///
/// A value of 0.55 represents a typical plastic-on-plastic collision (COR ≈ 0.4-0.6),
/// providing a realistic balance between elasticity and energy loss.
pub const BOUNCE_COEFFICIENT: f64 = 0.55;

/// Minimum upward velocity required to trigger another bounce.
/// If the reflected velocity is below this, the piece settles.
///
/// Note: With GRAVITY=230, a velocity of ~7.6 rows/s is required to produce a
/// bounce height of 0.5 characters (the threshold for TUI visibility).
/// A value of 8.0 settles the piece just above the theoretical visibility limit.
pub const MIN_BOUNCE_VELOCITY: f64 = 8.0;

// ========================================================================================
// UI COLORS
// ========================================================================================

/// Color of the main game cabinet frame.
pub const COLOR_CABINET: Color = Color::Blue;

/// Color of the decorative side wings and structural foundation.
pub const COLOR_WINGS: Color = Color::Rgb(70, 80, 100);

/// Color for Team Red (Player 1).
pub const COLOR_TEAM_RED: Color = Color::Red;

/// Color for Team Yellow (Player 2).
pub const COLOR_TEAM_YELLOW: Color = Color::Yellow;

/// Color used for successful logic outcomes or positive indicators (e.g., "Intelligence").
pub const COLOR_SUCCESS: Color = Color::Green;

/// Color used for warnings or neutral threats (e.g., "Medium Probability").
pub const COLOR_WARNING: Color = Color::Yellow;

/// Color used for danger or critical threats (e.g., "Loss Imminent").
pub const COLOR_DANGER: Color = Color::Red;

/// Color for active selections, cursors, and highlights.
/// Magenta is chosen as it never clashes with the Red/Yellow team colors.
pub const COLOR_SELECTION: Color = Color::Magenta;

/// Color for primary screen headers (e.g., "MULTIPLAYER LOBBY").
pub const COLOR_HEADER_PRIMARY: Color = Color::Cyan;

/// Color for secondary headers and sidebar category labels.
pub const COLOR_HEADER_SECONDARY: Color = Color::LightBlue;

/// Color for technical metadata, table sub-headers, and stale data.
pub const COLOR_DIM: Color = Color::DarkGray;

/// Color for primary informative text.
pub const COLOR_PRIMARY: Color = Color::White;

/// Color for secondary text or keyboard legends.
pub const COLOR_SECONDARY: Color = Color::Gray;

/// Background color for Red turn indicator.
pub const COLOR_TEAM_RED_BG: Color = Color::Rgb(60, 0, 0);

/// Background color for Yellow turn indicator.
pub const COLOR_TEAM_YELLOW_BG: Color = Color::Rgb(60, 50, 0);

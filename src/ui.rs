//! Terminal User Interface (TUI) rendering module for Connect Four.
//!
//! This module provides the core rendering logic for the application using `ratatui`.
//! It handles the layout, game board visualization, sidebar with analytics,
//! and various menu screens.

#![cfg(feature = "tui")]

use crate::App;
use crate::ui_config::{
    BOARD_SIZES, CELL_HEIGHT, CELL_WIDTH, COLOR_CABINET, COLOR_DANGER, COLOR_DIM,
    COLOR_HEADER_PRIMARY, COLOR_HEADER_SECONDARY, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SELECTION,
    COLOR_SUCCESS, COLOR_TEAM_RED, COLOR_TEAM_RED_BG, COLOR_TEAM_YELLOW, COLOR_TEAM_YELLOW_BG,
    COLOR_WARNING, COLOR_WINGS, COMPACT_NAME_LIMIT, EVAL_BAR_LEN, EVAL_SCALE, HEADER_NAME_LIMIT,
    MIN_UI_HEIGHT, MIN_UI_WIDTH, SIDEBAR_HPAD, SIDEBAR_WIDTH, STATS_NAME_LIMIT,
    TACTICAL_BAR_MAX_LEN, ToLabel,
};
use crate::ui_session::UiGameSession;
use crate::{AppState, GameOverReason, NetworkContext};
use connect4::config::{SCORE_WIN, WIN_THRESHOLD};
use connect4::types::{Cell, Difficulty, GameMode, Player};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Padding, Paragraph},
};

/// Converts a `usize` to `u16` safely for TUI coordinates.
///
/// If the value exceeds `u16::MAX`, it is clamped to `u16::MAX`.
fn to_u16(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// Converts a `u32` to `u16` safely for TUI coordinates.
///
/// If the value exceeds `u16::MAX`, it is clamped to `u16::MAX`.
fn u32_to_u16(n: u32) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// Converts a `u16` to `usize` safely.
fn to_usize(n: u16) -> usize {
    usize::from(n)
}

/// Specialized helper for board-to-terminal coordinate mapping.
///
/// This struct manages the conversion between board coordinates (columns and rows)
/// and terminal screen coordinates (x and y).
struct UiBoardGeometry {
    /// The bounding rectangle of the board area.
    pub area: Rect,
    /// The total number of rows in the board.
    pub rows: u32,
}

impl UiBoardGeometry {
    /// Creates a new `UiBoardGeometry` instance.
    ///
    /// # Arguments
    /// * `area` - The `Rect` representing the total board area.
    /// * `_cols` - Number of columns (currently unused for initialization but kept for symmetry).
    /// * `rows` - Number of rows in the board.
    fn new(area: Rect, _cols: u32, rows: u32) -> Self {
        Self { area, rows }
    }

    /// Calculates the `Rect` for a specific cell on the board.
    ///
    /// # Arguments
    /// * `col` - The 0-indexed column number.
    /// * `row` - The 0-indexed row number (0 is bottom).
    fn cell_rect(&self, col: u32, row: u32) -> Rect {
        let display_row = self.rows.saturating_sub(1).saturating_sub(row);
        Rect {
            x: self.area.x + 1 + u32_to_u16(col) * CELL_WIDTH,
            y: self.area.y + 1 + u32_to_u16(display_row) * CELL_HEIGHT,
            width: CELL_WIDTH,
            height: CELL_HEIGHT,
        }
    }

    /// Calculates the X-coordinate for the column selection indicator.
    fn col_indicator_x(&self, col: u32) -> u16 {
        self.area.x + 1 + u32_to_u16(col) * CELL_WIDTH + (CELL_WIDTH / 2) - 1
    }

    /// Calculates the X-coordinate for the column label.
    fn col_label_x(&self, col: u32) -> u16 {
        self.col_indicator_x(col)
    }
}

/// Transient context containing pre-calculated layout areas for the current frame.
///
/// This context is recreated on every frame to handle terminal resizing
/// and dynamic layout changes.
struct RenderContext {
    /// Geometry and coordinate mapping for the game board.
    pub board: UiBoardGeometry,
    /// Area allocated for the sidebar.
    pub sidebar: Rect,
    /// Area allocated for the header.
    pub header: Rect,
    /// Area allocated for the footer.
    pub footer: Rect,
    /// The main body area (excluding header and footer).
    pub body: Rect,
}

impl RenderContext {
    /// Creates a new `RenderContext` by calculating the layout for the given frame.
    ///
    /// Returns `None` if the terminal size is smaller than `MIN_UI_WIDTH` or `MIN_UI_HEIGHT`.
    fn new(f: &Frame, app: &App) -> Option<Self> {
        let size = f.area();
        if size.width < MIN_UI_WIDTH || size.height < MIN_UI_HEIGHT {
            return None;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(size);

        let header = chunks[0];
        let body = chunks[1];
        let footer = chunks[2];

        let outer_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(SIDEBAR_WIDTH)])
            .split(body);

        let main_area = outer_layout[0];
        let sidebar = outer_layout[1];

        // For states without a session, we use a default geometry for centering popup menus
        let (cols, rows) = if let Some(session) = &app.session {
            (session.core.board.columns(), session.core.board.rows())
        } else {
            (7, 6)
        };

        let bw = u32_to_u16(cols) * CELL_WIDTH + 1;
        let bh = u32_to_u16(rows) * CELL_HEIGHT + 1;

        let board_area = Rect {
            x: main_area.x + (main_area.width.saturating_sub(bw)) / 2,
            y: main_area.y + (main_area.height.saturating_sub(bh)) / 2,
            width: bw,
            height: bh,
        };

        Some(Self {
            board: UiBoardGeometry::new(board_area, cols, rows),
            sidebar,
            header,
            footer,
            body,
        })
    }
}

/// Root UI rendering function.
///
/// Dispatches rendering based on the current `AppState`.
/// If the terminal is too small, it renders a size error screen.
pub fn render(f: &mut Frame, app: &App) {
    let Some(ctx) = RenderContext::new(f, app) else {
        draw_size_error(f, f.area());
        return;
    };

    draw_header(f, ctx.header);
    draw_footer(f, ctx.footer, app);

    match app.state {
        AppState::Menu { selected_idx } => draw_selection_menu(
            f,
            ctx.body,
            "MAIN MENU",
            &GameMode::ALL.map(|m| ToLabel::label(&m)),
            selected_idx,
        ),
        AppState::LocalLobby {
            selected_idx,
            selected_id: _,
        } => draw_local_lobby(f, ctx.body, app, selected_idx),
        AppState::DifficultySelection { selected_idx } => draw_selection_menu(
            f,
            ctx.body,
            "SELECT DIFFICULTY",
            &Difficulty::ALL.map(|d| ToLabel::label(&d)),
            selected_idx,
        ),
        AppState::PlayerSelection { selected_idx } => draw_selection_menu(
            f,
            ctx.body,
            "PLAY AS",
            &Player::ALL.map(|p| ToLabel::label(&p)),
            selected_idx,
        ),
        AppState::SizeSelection { selected_idx } => draw_selection_menu(
            f,
            ctx.body,
            "SELECT BOARD SIZE",
            &BOARD_SIZES.iter().map(|s| s.label).collect::<Vec<_>>(),
            selected_idx,
        ),
        AppState::RemoteNameInput | AppState::GuestNameInput(_) => {
            let name = app.network.as_ref().map_or("", |n| &n.local_name);
            draw_text_input(f, ctx.body, "ENTER YOUR NAME", name);
        }
        AppState::Error(ref msg) => {
            draw_error_screen(f, ctx.body, msg);
        }
        AppState::Playing | AppState::ChatInput | AppState::GameOver(_) => {
            draw_active_session(f, &ctx, app);
        }
    }
}

/// Renders the active game session, including the board and sidebar.
///
/// This function is called when the game is in `Playing`, `ChatInput`, or `GameOver` states.
fn draw_active_session(f: &mut Frame, ctx: &RenderContext, app: &App) {
    if let Some(session) = &app.session {
        draw_game(f, ctx, app, session);

        // Apply horizontal padding to the sidebar area once
        let sidebar_area = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(ctx.sidebar)[1];

        if let AppState::GameOver(ref reason) = app.state {
            let side_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(10), Constraint::Min(0)])
                .split(sidebar_area);

            draw_game_over(f, side_chunks[0], app, reason);
            draw_enhanced_sidebar(f, side_chunks[1], app, session);
        } else {
            draw_enhanced_sidebar(f, sidebar_area, app, session);
        }

        if app.state == AppState::ChatInput
            && let Some(net) = &app.network
        {
            draw_text_input(f, ctx.body, "CHAT MESSAGE", &net.chat_input);
        }
    }
}

/// Creates a standard UI block with a title and rounded borders.
///
/// # Arguments
/// * `title` - The title to display on the top border.
/// * `color` - The color for the border and title.
fn create_panel_block(title: &str, color: Color) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .border_type(ratatui::widgets::BorderType::Rounded)
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
}

/// Draws the application header with the game title.
fn draw_header(f: &mut Frame, area: Rect) {
    let title = Paragraph::new(" 🔴 CONNECT FOUR 🟡 ")
        .style(
            Style::default()
                .fg(COLOR_HEADER_PRIMARY)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center)
        .block(create_panel_block("CONNECT FOUR", COLOR_HEADER_PRIMARY));
    f.render_widget(title, area);
}

/// Draws the application footer with context-sensitive help text.
fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let help_text = match app.state {
        AppState::Menu { .. }
        | AppState::DifficultySelection { .. }
        | AppState::PlayerSelection { .. }
        | AppState::SizeSelection { .. } => " [↑/↓] Navigate  [Enter] Select  [Esc/Q] Back/Exit ",
        AppState::LocalLobby { .. } => {
            " [↑/↓] Navigate  [Enter] Join  [R] Refresh  [Esc/Q] Back to Menu "
        }
        AppState::RemoteNameInput | AppState::GuestNameInput(_) => {
            " [Enter] Confirm Name  [Esc] Cancel "
        }
        AppState::ChatInput => " [Enter] Send Message  [Esc] Cancel/Close Chat ",
        AppState::Playing => {
            if let Some(session) = &app.session {
                if session.core.game_mode == GameMode::Remote {
                    " [←/→/1-9] Move  [Enter] Drop  [Alt+A/H/C] HUD  [/] Chat  [Esc/Q] Menu "
                } else {
                    " [←/→/1-9] Move  [Enter] Drop  [Alt+A/H] HUD  [R] Restart  [Esc/Q] Menu "
                }
            } else {
                ""
            }
        }
        AppState::GameOver(_) | AppState::Error(_) => " [Esc/Q] Return to Main Menu ",
    };

    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(COLOR_DIM))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(COLOR_DIM)),
        );
    f.render_widget(footer, area);
}

/// Draws a centered text input field with a blinking cursor.
///
/// # Arguments
/// * `title` - The title for the input field.
/// * `value` - The current text value in the input.
fn draw_text_input(f: &mut Frame, area: Rect, title: &str, value: &str) {
    let input_area = centered_rect(50, 7, area);
    f.render_widget(Clear, input_area);

    let max_display_len = to_usize(input_area.width.saturating_sub(4));
    let display_value = if value.len() > max_display_len {
        &value[value.len() - max_display_len..]
    } else {
        value
    };

    // Blinking cursor logic (500ms interval)
    let show_cursor = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        / 500)
        .is_multiple_of(2);

    let display_text = if show_cursor {
        format!("{display_value}_")
    } else {
        format!("{display_value} ")
    };

    let p = Paragraph::new(Span::styled(
        display_text,
        Style::default().fg(COLOR_PRIMARY),
    ))
    .block(create_panel_block(title, COLOR_HEADER_PRIMARY))
    .alignment(Alignment::Center);
    f.render_widget(p, input_area);
}

/// Draws the local multiplayer lobby screen.
fn draw_local_lobby(f: &mut Frame, area: Rect, app: &App, selected_idx: usize) {
    let lobby_area = centered_rect(80, 25, area);
    f.render_widget(Clear, lobby_area);

    let mut lines = render_lobby_status(app);
    lines.extend(render_discovered_hosts(app, selected_idx));
    lines.extend(render_lobby_footer());

    let lobby = Paragraph::new(lines)
        .block(create_panel_block(
            "LOCAL MULTIPLAYER LOBBY",
            COLOR_HEADER_PRIMARY,
        ))
        .alignment(Alignment::Left);
    f.render_widget(lobby, lobby_area);
}

// ========================================================================================
// LOBBY RENDERING HELPERS
// ========================================================================================

const LOBBY_NAME_WIDTH: usize = 15;
const LOBBY_SIZE_WIDTH: usize = 8;
const LOBBY_COLOR_WIDTH: usize = 12;

/// Renders the host's current status (hosting or idle) for the lobby.
fn render_lobby_status(app: &App) -> Vec<Line<'_>> {
    let (is_hosting, host_name) = app
        .network
        .as_ref()
        .map_or((false, "Host"), |n| (n.is_hosting, n.local_name.as_str()));

    let (status_header, status_data, action_line) = if is_hosting {
        let config = &BOARD_SIZES[app.pending_size_idx];
        let color_label = app.pending_player.to_string().to_uppercase();
        let color_style = match app.pending_player {
            Player::Red => Style::default().fg(COLOR_TEAM_RED),
            Player::Yellow => Style::default().fg(COLOR_TEAM_YELLOW),
        };

        (
            Line::from(vec![Span::styled(
                " Your Status: ",
                Style::default()
                    .fg(COLOR_HEADER_SECONDARY)
                    .add_modifier(Modifier::BOLD),
            )]),
            vec![
                Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("{:<LOBBY_NAME_WIDTH$} ", "Name"),
                        Style::default().fg(COLOR_DIM),
                    ),
                    Span::styled(
                        format!("{:<LOBBY_SIZE_WIDTH$} ", "Size"),
                        Style::default().fg(COLOR_DIM),
                    ),
                    Span::styled(
                        format!("{:<LOBBY_COLOR_WIDTH$} ", "Host Color"),
                        Style::default().fg(COLOR_DIM),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("{host_name:<LOBBY_NAME_WIDTH$} "),
                        Style::default().fg(COLOR_SUCCESS),
                    ),
                    Span::styled(
                        format!(
                            "{:<LOBBY_SIZE_WIDTH$} ",
                            format!("{}x{}", config.size.cols, config.size.rows)
                        ),
                        Style::default().fg(COLOR_HEADER_PRIMARY),
                    ),
                    Span::styled(format!("{color_label:<LOBBY_COLOR_WIDTH$} "), color_style),
                ]),
            ],
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[S] Stop Hosting", Style::default().fg(COLOR_SECONDARY)),
            ]),
        )
    } else {
        (
            Line::from(vec![Span::styled(
                " Your Status: ",
                Style::default()
                    .fg(COLOR_HEADER_SECONDARY)
                    .add_modifier(Modifier::BOLD),
            )]),
            vec![Line::from(vec![
                Span::raw("    "),
                Span::styled("Idle (Not Hosting) ", Style::default().fg(COLOR_SECONDARY)),
            ])],
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[H] Host a New Game", Style::default().fg(COLOR_SECONDARY)),
            ]),
        )
    };

    let mut lines = vec![status_header, Line::from("")];
    lines.extend(status_data);
    lines.extend(vec![
        Line::from(""),
        action_line,
        Line::from(""),
        Line::from(vec![Span::styled(
            " Discovered Local Games: ",
            Style::default()
                .fg(COLOR_HEADER_PRIMARY)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
    ]);
    lines
}

/// Renders the list of discovered LAN hosts.
fn render_discovered_hosts(app: &App, selected_idx: usize) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    let Some(net) = &app.network else {
        return lines;
    };

    if net.discovered_hosts.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "  No other local games found yet.",
            Style::default().fg(COLOR_DIM),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "  Waiting for peers to announce themselves...",
            Style::default().fg(COLOR_DIM),
        )]));
        return lines;
    }

    // Header Row
    lines.push(Line::from(vec![
        Span::raw("    "),
        Span::styled(
            format!("{:<LOBBY_NAME_WIDTH$} ", "Name"),
            Style::default().fg(COLOR_DIM),
        ),
        Span::styled(
            format!("{:<LOBBY_SIZE_WIDTH$} ", "Size"),
            Style::default().fg(COLOR_DIM),
        ),
        Span::styled(
            format!("{:<LOBBY_COLOR_WIDTH$} ", "Host Color"),
            Style::default().fg(COLOR_DIM),
        ),
        Span::styled("Address", Style::default().fg(COLOR_DIM)),
    ]));

    for (i, (info, _)) in net.discovered_hosts.iter().enumerate() {
        let is_selected = i == selected_idx;
        let prefix = if is_selected { "  > " } else { "    " };

        let style_base = if is_selected {
            Style::default()
                .fg(COLOR_SELECTION)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(COLOR_PRIMARY)
        };

        let dim_style = if is_selected {
            Style::default().fg(COLOR_DIM).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(COLOR_DIM)
        };

        let color_label = info.host_player.to_string().to_uppercase();
        let base_team_color = match info.host_player {
            Player::Red => COLOR_TEAM_RED,
            Player::Yellow => COLOR_TEAM_YELLOW,
        };
        let team_style = if is_selected {
            Style::default()
                .fg(base_team_color)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(base_team_color)
        };

        lines.push(Line::from(vec![
            Span::styled(
                prefix,
                if is_selected {
                    Style::default()
                        .fg(COLOR_SELECTION)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(COLOR_DIM)
                },
            ),
            Span::styled(
                format!("{:<LOBBY_NAME_WIDTH$} ", info.local_name),
                style_base,
            ),
            Span::styled(
                format!(
                    "{:<LOBBY_SIZE_WIDTH$} ",
                    format!("{}x{}", info.board_size.0, info.board_size.1)
                ),
                if is_selected {
                    Style::default()
                        .fg(COLOR_HEADER_PRIMARY)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(COLOR_HEADER_PRIMARY)
                },
            ),
            Span::styled(format!("{color_label:<LOBBY_COLOR_WIDTH$} "), team_style),
            Span::styled(format!("({})", info.addr), dim_style),
        ]));
    }
    lines
}

/// Renders the footer for the lobby screen.
fn render_lobby_footer() -> Vec<Line<'static>> {
    vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "  [↑/↓] Select Host  [Enter] Join  [R] Refresh List  [Esc/Q] Back to Menu",
            Style::default().fg(COLOR_SECONDARY),
        )]),
    ]
}

/// Draws a generic selection menu with a list of options.
///
/// # Arguments
/// * `title` - The title for the menu.
/// * `options` - A slice of string labels for the menu options.
/// * `selected_idx` - The index of the currently selected option.
fn draw_selection_menu(
    f: &mut Frame,
    area: Rect,
    title: &str,
    options: &[&str],
    selected_idx: usize,
) {
    let menu_area = centered_rect(40, 12, area);
    f.render_widget(Clear, menu_area);

    let menu_items: Vec<Line> = options
        .iter()
        .enumerate()
        .map(|(i, &opt)| {
            let style = if i == selected_idx {
                Style::default()
                    .fg(COLOR_SELECTION)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(COLOR_SECONDARY)
            };
            Line::from(vec![
                Span::styled(if i == selected_idx { "  > " } else { "    " }, style),
                Span::styled(opt, style),
            ])
        })
        .collect();

    let menu = Paragraph::new(menu_items)
        .block(create_panel_block(title, COLOR_HEADER_PRIMARY))
        .alignment(Alignment::Left);
    f.render_widget(menu, menu_area);
}

/// Draws the enhanced sidebar containing session info, history, analysis, and chat.
fn draw_enhanced_sidebar(f: &mut Frame, area: Rect, app: &App, session: &UiGameSession) {
    let side_chunks = split_sidebar_layout(area, app);

    let mut next_idx = 0;
    draw_status_panel(f, side_chunks[next_idx], app, session);
    next_idx += 1;

    if app.settings.show_history {
        draw_history_pane(f, side_chunks[next_idx], app, session);
        next_idx += 1;
    }
    if app.settings.show_analysis {
        draw_analysis_pane(f, side_chunks[next_idx], app, session);
        next_idx += 1;
    }
    if let Some(net) = &app.network
        && net.show_chat
    {
        draw_chat_pane(f, side_chunks[next_idx], net);
    }
}

/// Splits the sidebar area into multiple chunks based on enabled features.
fn split_sidebar_layout(area: Rect, app: &App) -> std::rc::Rc<[Rect]> {
    let mut constraints = vec![Constraint::Length(7)]; // SESSION
    if app.settings.show_history {
        constraints.push(Constraint::Length(7)); // HISTORY
    }
    if app.settings.show_analysis {
        constraints.push(Constraint::Length(15)); // ANALYSIS
    }
    if let Some(net) = &app.network
        && net.show_chat
    {
        constraints.push(Constraint::Min(0)); // CHAT
    }

    Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area)
}

/// Draws the analysis pane with the evaluation bar and tactical breakdown.
fn draw_analysis_pane(f: &mut Frame, area: Rect, app: &App, session: &UiGameSession) {
    let mut intel_lines = vec![];

    // 1. Evaluation Bar (Team Momentum)
    intel_lines.extend(render_momentum_bar(app, session));

    // Separator
    intel_lines.push(Line::from(vec![Span::styled(
        "─".repeat(to_usize(area.width).saturating_sub(SIDEBAR_HPAD * 2 + 2)),
        Style::default().fg(COLOR_DIM),
    )]));

    // 2. Tactical Breakdown
    intel_lines.extend(render_tactical_breakdown(session));

    f.render_widget(
        Paragraph::new(intel_lines).block(
            create_panel_block("ANALYSIS", COLOR_HEADER_SECONDARY)
                .border_style(Style::default().fg(COLOR_DIM))
                .padding(Padding::horizontal(to_u16(SIDEBAR_HPAD).max(1))),
        ),
        area,
    );
}

/// State for rendering the momentum/evaluation bar.
struct MomentumState {
    /// Number of segments for the Red team.
    pub red_bars: usize,
    /// Number of segments for the Yellow team.
    pub yellow_bars: usize,
    /// Threat status message.
    pub threat_msg: &'static str,
    /// Color for the threat message.
    pub threat_color: Color,
    /// Whether the analytics are currently stale (e.g., AI still thinking).
    pub is_stale: bool,
}

impl MomentumState {
    /// Calculates the momentum state for the current application and session.
    fn new(app: &App, session: &UiGameSession) -> Self {
        let is_stale = session.core.is_stale_analytics;
        let (red_bars, yellow_bars, threat_msg, threat_color) = match app.state {
            AppState::GameOver(GameOverReason::Win(Player::Red)) => {
                (EVAL_BAR_LEN, 0, "GAME OVER", COLOR_SECONDARY)
            }
            AppState::GameOver(GameOverReason::Win(Player::Yellow)) => {
                (0, EVAL_BAR_LEN, "GAME OVER", COLOR_SECONDARY)
            }
            AppState::GameOver(GameOverReason::Stalemate) => {
                (EVAL_BAR_LEN / 2, EVAL_BAR_LEN / 2, "DRAW", COLOR_SECONDARY)
            }
            AppState::GameOver(GameOverReason::NetworkError(_)) => (
                EVAL_BAR_LEN / 2,
                EVAL_BAR_LEN / 2,
                "DISCONNECTED",
                COLOR_SECONDARY,
            ),
            AppState::GameOver(GameOverReason::Desync) => {
                (EVAL_BAR_LEN / 2, EVAL_BAR_LEN / 2, "DESYNC", COLOR_DANGER)
            }
            _ => {
                let scores: Vec<i32> = session
                    .core
                    .cached_scores
                    .iter()
                    .filter_map(|&s| s)
                    .collect();

                if scores.is_empty() {
                    (
                        EVAL_BAR_LEN / 2,
                        EVAL_BAR_LEN / 2,
                        "ANALYZING...",
                        COLOR_SECONDARY,
                    )
                } else {
                    // Find the best score for the current player (the one
                    // whose move it is).
                    let best_score_abs = if session.core.current_player == Player::Red {
                        *scores.iter().max().unwrap_or(&0)
                    } else {
                        *scores.iter().min().unwrap_or(&0)
                    };

                    let clamped_score = best_score_abs.clamp(-EVAL_SCALE / 2, EVAL_SCALE / 2);
                    let red_prob_scaled = (EVAL_SCALE / 2) + clamped_score;

                    let rb_scaled = u32::try_from(red_prob_scaled).expect("Score fits in u32");
                    let eval_total = u32::try_from(EVAL_SCALE).expect("EVAL_SCALE fits in u32");
                    let rb = (usize::try_from(rb_scaled).expect("Score fits in usize")
                        * EVAL_BAR_LEN)
                        / usize::try_from(eval_total).expect("EVAL_SCALE fits in usize");

                    let (msg, color) = get_threat_status(session);
                    (rb, EVAL_BAR_LEN - rb, msg, color)
                }
            }
        };

        Self {
            red_bars,
            yellow_bars,
            threat_msg,
            threat_color,
            is_stale,
        }
    }
}

/// Renders the momentum bar showing which team is currently ahead.
fn render_momentum_bar<'a>(app: &App, session: &UiGameSession) -> Vec<Line<'a>> {
    let state = MomentumState::new(app, session);
    let mut lines = vec![];
    let red_short = get_display_name(app, session, Player::Red, COMPACT_NAME_LIMIT, false);
    let ylw_short = get_display_name(app, session, Player::Yellow, COMPACT_NAME_LIMIT, false);

    let (red_color, yellow_color) = if state.is_stale {
        (COLOR_DIM, COLOR_DIM)
    } else {
        (COLOR_TEAM_RED, COLOR_TEAM_YELLOW)
    };

    lines.push(Line::from(vec![
        Span::styled(
            format!("{red_short:>5} "),
            Style::default().fg(COLOR_TEAM_RED),
        ),
        Span::styled("█".repeat(state.red_bars), Style::default().fg(red_color)),
        Span::styled(
            "█".repeat(state.yellow_bars),
            Style::default().fg(yellow_color),
        ),
        Span::styled(
            format!(" {ylw_short:<5}"),
            Style::default().fg(COLOR_TEAM_YELLOW),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Threat: ", Style::default().fg(COLOR_SECONDARY)),
        Span::styled(
            state.threat_msg,
            Style::default()
                .fg(state.threat_color)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    lines
}

/// Formats a raw score for display.
///
/// Scores exceeding `WIN_THRESHOLD` are displayed as "W" or "L" followed
/// by the number of moves to win/loss.
fn format_score(score: i32) -> String {
    let abs_score = score.abs();
    if abs_score >= WIN_THRESHOLD {
        let ply = SCORE_WIN - abs_score;
        let moves = (ply + 1) / 2;
        let prefix = if score > 0 { "W" } else { "L" };
        format!("{prefix}{moves}")
    } else {
        format!("{score}")
    }
}

/// Renders a column-by-column tactical breakdown of the current board state.
fn render_tactical_breakdown<'a>(session: &UiGameSession) -> Vec<Line<'a>> {
    let mut lines = vec![];
    let scores: Vec<i32> = session
        .core
        .cached_scores
        .iter()
        .filter_map(|&s| s)
        .collect();

    if scores.is_empty() {
        return vec![Line::from(vec![Span::styled(
            "Analyzing...",
            Style::default().fg(COLOR_DIM),
        )])];
    }

    let min_score = *scores.iter().min().unwrap_or(&0);
    let max_score = *scores.iter().max().unwrap_or(&1);
    let range = (max_score - min_score).max(1);

    let bar_color = if session.core.is_stale_analytics {
        COLOR_DIM
    } else {
        COLOR_SELECTION
    };

    lines.push(Line::from(vec![
        Span::styled("C  ", Style::default().fg(COLOR_DIM)),
        Span::styled("Evaluation", Style::default().fg(COLOR_DIM)),
        Span::styled(" ".repeat(TACTICAL_BAR_MAX_LEN - 10 + 1), Style::default()),
        Span::styled("Score", Style::default().fg(COLOR_DIM)),
    ]));

    for c in 0..session.core.board.columns() {
        if let Some(s) = session.core.cached_scores[c as usize] {
            // Bars and displayed scores should be relative to the current player.
            // Positive = good for them, Negative = bad.
            let relative_score = if session.core.current_player == Player::Red {
                s
            } else {
                -s
            };

            let diff = if session.core.current_player == Player::Red {
                s - min_score
            } else {
                max_score - s
            };

            let bar_len = if diff > 0 {
                (usize::try_from(u32::try_from(diff).expect("Score fits in u32"))
                    .expect("Score fits in usize")
                    * TACTICAL_BAR_MAX_LEN)
                    / usize::try_from(u32::try_from(range).expect("Score range fits in u32"))
                        .expect("Score range fits in usize")
            } else {
                0
            };
            let bar = "█".repeat(bar_len);
            let padding = " ".repeat(TACTICAL_BAR_MAX_LEN - bar_len);
            let display_score = format_score(relative_score);
            lines.push(Line::from(vec![
                Span::styled(format!("{}: ", c + 1), Style::default().fg(COLOR_SECONDARY)),
                Span::styled(bar, Style::default().fg(bar_color)),
                Span::styled(padding, Style::default()),
                Span::raw(" "),
                Span::styled(
                    format!("{display_score:>8}"),
                    Style::default().fg(bar_color),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled(format!("{}: ", c + 1), Style::default().fg(COLOR_SECONDARY)),
                Span::styled("FULL", Style::default().fg(COLOR_DIM)),
            ]));
        }
    }
    lines
}

/// Draws the move history pane.
fn draw_history_pane(f: &mut Frame, area: Rect, app: &App, session: &UiGameSession) {
    let mut history_lines = Vec::new();
    for (i, col_player) in session.core.move_history.iter().rev().take(5).enumerate() {
        let (col, player) = *col_player;
        let p_color = match player {
            Player::Red => COLOR_TEAM_RED,
            Player::Yellow => COLOR_TEAM_YELLOW,
        };
        let p_name = get_display_name(app, session, player, COMPACT_NAME_LIMIT, false);
        history_lines.push(Line::from(vec![
            Span::styled(
                format!("# {:<2} ", session.core.move_history.len() - i),
                Style::default().fg(COLOR_DIM),
            ),
            Span::styled(format!("{p_name:<12}"), Style::default().fg(p_color)),
            Span::styled(
                format!(" → Col {}", col + 1),
                Style::default().fg(COLOR_SECONDARY),
            ),
        ]));
    }

    f.render_widget(
        Paragraph::new(history_lines).block(
            create_panel_block("MOVE HISTORY", COLOR_HEADER_SECONDARY)
                .border_style(Style::default().fg(COLOR_DIM))
                .padding(Padding::horizontal(to_u16(SIDEBAR_HPAD).max(1))),
        ),
        area,
    );
}

/// Draws the chat history pane.
fn draw_chat_pane(f: &mut Frame, area: Rect, net: &NetworkContext) {
    let usable_width = to_usize(area.width)
        .saturating_sub(SIDEBAR_HPAD * 2 + 2)
        .max(1);
    let usable_height = to_usize(area.height).saturating_sub(2);

    let mut all_wrapped_lines = Vec::new();
    for msg in &net.chat_history {
        // Sanitize: replace tabs with spaces and remove newlines to prevent layout breaks
        let sanitized = msg.replace('\t', "    ").replace(['\n', '\r'], " ");

        let mut current_pos = 0;
        let msg_chars: Vec<char> = sanitized.chars().collect();
        if msg_chars.is_empty() {
            // Even empty messages (if they somehow get in) should probably show a blank line
            // or we can just skip them. Based on current history logic, we skip.
            continue;
        }

        while current_pos < msg_chars.len() {
            let end_pos = (current_pos + usable_width).min(msg_chars.len());
            let chunk: String = msg_chars[current_pos..end_pos].iter().collect();
            // Pad chunk to usable_width to ensure consistent background if one is added later
            all_wrapped_lines.push(Line::from(format!("{chunk:<usable_width$}")));
            current_pos = end_pos;
        }
    }

    // Only display the most recent lines that actually fit in the viewport
    let display_lines: Vec<Line> = all_wrapped_lines
        .into_iter()
        .rev()
        .take(usable_height)
        .rev()
        .collect();

    f.render_widget(
        Paragraph::new(display_lines).block(
            create_panel_block("CHAT", COLOR_HEADER_SECONDARY)
                .border_style(Style::default().fg(COLOR_DIM))
                .padding(Padding::horizontal(to_u16(SIDEBAR_HPAD).max(1))),
        ),
        area,
    );
}

/// Draws the main session status panel.
fn draw_status_panel(f: &mut Frame, area: Rect, app: &App, session: &UiGameSession) {
    let status_lines = render_player_stats(app, session);
    f.render_widget(
        Paragraph::new(status_lines).block(
            create_panel_block("SESSION", COLOR_HEADER_SECONDARY)
                .border_style(Style::default().fg(COLOR_DIM))
                .padding(Padding::horizontal(to_u16(SIDEBAR_HPAD).max(1))),
        ),
        area,
    );
}

/// Renders player statistics and session metadata for the status panel.
fn render_player_stats<'a>(app: &'a App, session: &'a UiGameSession) -> Vec<Line<'a>> {
    let mut lines = vec![];

    // 1. Turn Indicator
    lines.extend(render_turn_indicator(app, session));

    // 2. Player Chain Stats
    lines.extend(render_player_chain_stats(app, session));

    // 3. Session Metadata (Moves, Difficulty/Ping)
    lines.extend(render_session_metadata(app, session));

    lines
}

/// Renders the turn indicator showing whose move it is.
fn render_turn_indicator<'a>(app: &'a App, session: &'a UiGameSession) -> Vec<Line<'a>> {
    let is_local_player_turn = session.core.current_player == session.core.local_player;
    let turn_name = get_display_name(
        app,
        session,
        session.core.current_player,
        HEADER_NAME_LIMIT,
        true,
    );

    if session.core.game_mode == GameMode::LocalTwo
        || (session.core.game_mode == GameMode::Single && !is_local_player_turn)
    {
        let turn_color = match session.core.current_player {
            Player::Red => COLOR_TEAM_RED,
            Player::Yellow => COLOR_TEAM_YELLOW,
        };
        let mut spans = vec![Span::styled(
            format!("▶ {turn_name} Turn ◀"),
            Style::default()
                .fg(turn_color)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )];
        if session.core.is_ai_thinking {
            spans.push(Span::styled(" ⌛", Style::default().fg(COLOR_SELECTION)));
        }
        vec![Line::from(spans)]
    } else if is_local_player_turn {
        vec![Line::from(vec![Span::styled(
            "▶ Your Turn ◀",
            Style::default()
                .fg(COLOR_SUCCESS)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )])]
    } else {
        vec![Line::from(vec![Span::styled(
            format!("⌛ Waiting for {turn_name}"),
            Style::default().fg(COLOR_DIM),
        )])]
    }
}

/// Renders longest chain statistics for both players.
fn render_player_chain_stats<'a>(app: &'a App, session: &'a UiGameSession) -> Vec<Line<'a>> {
    let mut lines = vec![];
    let red_label = get_display_name(app, session, Player::Red, STATS_NAME_LIMIT, true);
    let yellow_label = get_display_name(app, session, Player::Yellow, STATS_NAME_LIMIT, true);

    let red_active = session.core.current_player == Player::Red;
    let yellow_active = session.core.current_player == Player::Yellow;

    let red_bg = if red_active {
        COLOR_TEAM_RED_BG
    } else {
        Color::Reset
    };
    let yellow_bg = if yellow_active {
        COLOR_TEAM_YELLOW_BG
    } else {
        Color::Reset
    };

    lines.push(Line::from(vec![
        Span::styled(
            format!("{red_label:<15}"),
            Style::default()
                .fg(COLOR_TEAM_RED)
                .bg(red_bg)
                .add_modifier(if red_active {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ),
        Span::styled(" L: ", Style::default().fg(COLOR_SECONDARY).bg(red_bg)),
        Span::styled(
            format!("{:>2}", session.core.board.stats().red_longest_chain),
            Style::default().fg(COLOR_TEAM_RED).bg(red_bg),
        ),
        Span::styled(" / 4 ", Style::default().fg(COLOR_DIM).bg(red_bg)),
    ]));

    lines.push(Line::from(vec![
        Span::styled(
            format!("{yellow_label:<15}"),
            Style::default()
                .fg(COLOR_TEAM_YELLOW)
                .bg(yellow_bg)
                .add_modifier(if yellow_active {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ),
        Span::styled(" L: ", Style::default().fg(COLOR_SECONDARY).bg(yellow_bg)),
        Span::styled(
            format!("{:>2}", session.core.board.stats().yellow_longest_chain),
            Style::default().fg(COLOR_TEAM_YELLOW).bg(yellow_bg),
        ),
        Span::styled(" / 4 ", Style::default().fg(COLOR_DIM).bg(yellow_bg)),
    ]));

    lines
}

/// Renders session-level metadata such as move count and difficulty/ping.
fn render_session_metadata<'a>(app: &'a App, session: &'a UiGameSession) -> Vec<Line<'a>> {
    let mut lines = vec![];

    lines.push(Line::from(vec![
        Span::styled("Moves: ", Style::default().fg(COLOR_SECONDARY)),
        Span::styled(
            format!("{:<2}", session.core.move_history.len()),
            Style::default()
                .fg(COLOR_SELECTION)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    if session.core.game_mode == GameMode::Single {
        lines.push(Line::from(vec![
            Span::styled("Difficulty: ", Style::default().fg(COLOR_SECONDARY)),
            Span::styled(
                ToLabel::label(&session.core.difficulty),
                Style::default()
                    .fg(COLOR_SELECTION)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    } else if session.core.game_mode == GameMode::Remote {
        let latency_str = app
            .network
            .as_ref()
            .and_then(|n| n.latency)
            .map_or("---".to_string(), |l| format!("{l}ms"));
        lines.push(Line::from(vec![
            Span::styled("Ping: ", Style::default().fg(COLOR_SECONDARY)),
            Span::styled(
                latency_str,
                Style::default()
                    .fg(COLOR_SELECTION)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines
}

/// Helper to get the current threat status and associated color.
fn get_threat_status(session: &UiGameSession) -> (&'static str, Color) {
    let player = session.core.current_player;
    let status = session.core.board.get_threat_status(player);
    match status {
        connect4::game::ThreatLevel::Critical => ("CRITICAL", COLOR_DANGER),
        connect4::game::ThreatLevel::Caution => ("CAUTION", COLOR_WARNING),
        connect4::game::ThreatLevel::Stable => ("STABLE", COLOR_SUCCESS),
    }
}

/// Main game board rendering function.
///
/// Handles the cabinet, indicators, and the grid itself.
fn draw_game(f: &mut Frame, ctx: &RenderContext, app: &App, session: &UiGameSession) {
    draw_cabinet_wings(
        f,
        ctx.board.area,
        ctx.board.area.height,
        ctx.board.area.width,
    );
    let cabinet_style = Style::default().fg(COLOR_CABINET);
    f.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Thick)
            .border_style(cabinet_style),
        ctx.board.area,
    );

    if app.state == AppState::Playing || app.state == AppState::ChatInput {
        let indicator_x = ctx.board.col_indicator_x(session.selected_column);

        let is_actionable = (session.core.game_mode == GameMode::LocalTwo
            || session.core.current_player == session.core.local_player)
            && session.falling_pieces.is_empty();

        let color = if is_actionable {
            match session.core.current_player {
                Player::Red => COLOR_TEAM_RED,
                Player::Yellow => COLOR_TEAM_YELLOW,
            }
        } else {
            COLOR_DIM
        };

        f.buffer_mut().set_string(
            indicator_x,
            ctx.board.area.y.saturating_sub(1),
            "▼",
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        );

        for c in 0..session.core.board.columns() {
            let label_x = ctx.board.col_label_x(c);
            let style = if c == session.selected_column {
                Style::default()
                    .fg(color)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
            } else {
                Style::default().fg(COLOR_DIM)
            };
            f.buffer_mut().set_string(
                label_x,
                ctx.board.area.y.saturating_sub(2),
                format!("{}", c + 1),
                style,
            );
        }
    }
    draw_board_grid(f, ctx, app, session, cabinet_style);
}

/// Draws decorative "wings" and foundation for the arcade cabinet.
fn draw_cabinet_wings(f: &mut Frame, board_area: Rect, board_height: u16, board_width: u16) {
    let left_margin = board_area.x;
    let right_margin = f.area().width.saturating_sub(board_area.x + board_width);

    // Ensure the wide foundation (5 chars per side) fits with a 1-char buffer
    if left_margin >= 6 && right_margin >= 6 {
        let rail_style = Style::default().fg(COLOR_WINGS);
        let connect_style = rail_style;

        // Render rails and foundation.
        // We extend 2 rows below the board for the thick plinth.
        for y_offset in 0..=(board_height + 1) {
            let abs_y = board_area.y + y_offset;
            let left_rail_x = board_area.x - 3;
            let right_rail_x = board_area.x + board_width + 2;

            if y_offset < board_height {
                // Regular vertical rails
                f.buffer_mut()
                    .set_string(left_rail_x, abs_y, "║", rail_style);
                f.buffer_mut()
                    .set_string(right_rail_x, abs_y, "║", rail_style);

                // Offset Connectors (only within board boundaries)
                if y_offset % CELL_HEIGHT == CELL_HEIGHT / 2 {
                    f.buffer_mut()
                        .set_string(board_area.x - 2, abs_y, "─╴", connect_style);
                    f.buffer_mut().set_string(
                        board_area.x + board_width,
                        abs_y,
                        "╶─",
                        connect_style,
                    );
                }
            } else if y_offset == board_height {
                // Tier 1 Foundation (Solid Plinth)
                f.buffer_mut()
                    .set_string(left_rail_x - 1, abs_y, "▟", rail_style);
                for x in left_rail_x..=right_rail_x {
                    f.buffer_mut().set_string(x, abs_y, "█", rail_style);
                }
                f.buffer_mut()
                    .set_string(right_rail_x + 1, abs_y, "▙", rail_style);
            } else if y_offset == board_height + 1 {
                // Tier 2 Foundation (Flared Floor - Continuous Top-Half bar)
                // We extend from left_rail_x - 2 to right_rail_x + 2 for a wide, solid base
                for x in (left_rail_x - 2)..=(right_rail_x + 2) {
                    f.buffer_mut().set_string(x, abs_y, "▀", rail_style);
                }
            }
        }
    }
}

/// Renders the main game grid, handling the 3-phase animation system.
///
/// # Animation Phases
/// 1. **Logic State**: The core game logic (`BoardState`) is updated immediately when a move is made.
///    This allows the AI and Network layers to proceed without waiting for animations.
/// 2. **Global Hide**: If a piece is currently "falling" (in `session.falling_pieces`), the target
///    cell in the grid is rendered as empty (hidden). This prevents "ghosting" where the piece
///    appears at the destination before the animation arrives.
/// 3. **Cosmetic Overlay**: The falling piece is drawn at its interpolated `current_y` position
///    over the top of the grid.
fn draw_board_grid(
    f: &mut Frame,
    ctx: &RenderContext,
    app: &App,
    session: &UiGameSession,
    cabinet_style: Style,
) {
    let (board_width, board_height) = (ctx.board.area.width, ctx.board.area.height);
    let min_y = ctx.board.area.y + 1;
    let max_y = ctx.board.area.y + board_height - 2;

    // 1. Draw static pieces and target guides
    draw_grid_cells(f, ctx, app, session, min_y, max_y);

    // 2. Render active cosmetic animation overlay
    if let Some(anim) = session.falling_pieces.front() {
        draw_animation_overlay(f, ctx, session, anim, min_y, max_y);
    }

    // 3. Draw the grid lines last so that pieces appear to be "behind" the cabinet structure.
    for i in 1..session.core.board.columns() {
        let x = ctx.board.area.x + u32_to_u16(i) * CELL_WIDTH;
        for y in ctx.board.area.y + 1..ctx.board.area.y + board_height - 1 {
            f.buffer_mut()[(x, y)]
                .set_char('│')
                .set_style(cabinet_style);
        }
    }
    for i in 1..session.core.board.rows() {
        let y = ctx.board.area.y + u32_to_u16(i) * CELL_HEIGHT;
        for x in ctx.board.area.x + 1..ctx.board.area.x + board_width - 1 {
            f.buffer_mut()[(x, y)]
                .set_char('─')
                .set_style(cabinet_style);
        }
    }
}

/// Renders the static grid cells and landing guides.
fn draw_grid_cells(
    f: &mut Frame,
    ctx: &RenderContext,
    app: &App,
    session: &UiGameSession,
    min_y: u16,
    max_y: u16,
) {
    for c in 0..session.core.board.columns() {
        // Calculate the landing row for the selected column to place the 'Target' anchor
        let mut target_row = None;
        if (app.state == AppState::Playing || app.state == AppState::ChatInput)
            && session.selected_column == c
        {
            target_row = session.core.board.get_first_empty_row(c);
        }

        for r in 0..session.core.board.rows() {
            // Global Hide logic: Calculate if this cell should be hidden because it's in the animation queue
            let mut is_hidden = false;
            for anim in &session.falling_pieces {
                if anim.col == c && anim.target_row == r {
                    is_hidden = true;
                    break;
                }
            }

            let cell_rect = ctx.board.cell_rect(c, r);
            let is_selected = (app.state == AppState::Playing || app.state == AppState::ChatInput)
                && session.selected_column == c;

            let is_occupied = matches!(session.core.board.get_cell(c, r), Cell::Occupied(_));

            if is_hidden {
                // Empty space for hidden pieces
            } else if let Cell::Occupied(player) = session.core.board.get_cell(c, r) {
                render_piece(
                    f,
                    cell_rect.x,
                    cell_rect.y,
                    player,
                    Style::default(),
                    min_y,
                    max_y,
                );
            } else {
                // Empty space for unoccupied cells
            }

            // Draw selection target anchor for the landing cell in the selected column
            if is_selected && (!is_occupied || is_hidden) && target_row == Some(r) {
                draw_landing_guide(f, cell_rect.x, cell_rect.y);
            }
        }
    }
}

/// Draws a visual landing guide (target anchor) for the current column.
fn draw_landing_guide(f: &mut Frame, x: u16, y: u16) {
    let style = Style::default().fg(COLOR_DIM);
    // 'High-Fidelity Trinity Orbit' - Perfectly continuous dot-symmetric circle.
    // Uses braille_dots helper to define characters by their bit-positions (1-8).

    // Line 0: Top Arc
    let l0_c1 = braille_dots(&[8]);
    let l0_c2 = braille_dots(&[5, 3]);
    let l0_c3 = braille_dots(&[2, 5]);
    let l0_c4 = braille_dots(&[2, 6]);
    let l0_c5 = braille_dots(&[7]);
    f.buffer_mut().set_string(
        x + 1,
        y,
        format!("{l0_c1}{l0_c2}{l0_c3}{l0_c4}{l0_c5}"),
        style,
    );

    // Line 1: Center Bulge
    let l1_c1 = braille_dots(&[1, 2, 3, 7]);
    let l1_c5 = braille_dots(&[4, 5, 6, 8]);
    f.buffer_mut()
        .set_string(x + 1, y + 1, format!("{l1_c1}   {l1_c5}"), style);

    // Line 2: Bottom Arc (Mirrored)
    let l2_c1 = braille_dots(&[4]);
    let l2_c2 = braille_dots(&[6, 2]);
    let l2_c3 = braille_dots(&[3, 6]);
    let l2_c4 = braille_dots(&[3, 5]);
    let l2_c5 = braille_dots(&[1]);
    f.buffer_mut().set_string(
        x + 1,
        y + 2,
        format!("{l2_c1}{l2_c2}{l2_c3}{l2_c4}{l2_c5}"),
        style,
    );
}

/// Draws an overlay for falling pieces during animations.
fn draw_animation_overlay(
    f: &mut Frame,
    ctx: &RenderContext,
    session: &UiGameSession,
    anim: &crate::ui_session::FallingPiece,
    min_y: u16,
    max_y: u16,
) {
    let rows_f64 = f64::from(session.core.board.rows());
    let cell_x = ctx.board.area.x + 1 + u32_to_u16(anim.col) * CELL_WIDTH;

    // Calculate Y position with row-level interpolation
    let display_y = rows_f64 - 1.0 - anim.current_y;

    // Calculate total character-level offset for smooth, non-snapping animation
    let total_offset_f64 = display_y * f64::from(CELL_HEIGHT);

    // Use rounded offset for smooth animation. Truncation is safe here as total_offset_f64
    // is physically bounded by the board height (rows * CELL_HEIGHT).
    #[allow(clippy::cast_possible_truncation)]
    let offset_i16 = total_offset_f64.round() as i16;

    let cell_y = ctx
        .board
        .area
        .y
        .saturating_add(1)
        .saturating_add_signed(offset_i16);

    render_piece(
        f,
        cell_x,
        cell_y,
        anim.player,
        Style::default(),
        min_y,
        max_y,
    );
}

/// Draws the game over screen in the sidebar.
fn draw_game_over(f: &mut Frame, area: Rect, app: &App, reason: &GameOverReason) {
    let mut lines = vec![Line::from("")];

    let header_style = Style::default().add_modifier(Modifier::BOLD);
    match reason {
        GameOverReason::Win(player) => {
            let color = match player {
                Player::Red => COLOR_TEAM_RED,
                Player::Yellow => COLOR_TEAM_YELLOW,
            };
            lines.push(Line::from(vec![
                Span::styled("Winner: ", header_style),
                Span::styled(player.to_string().to_uppercase(), header_style.fg(color)),
                Span::styled("!", header_style),
            ]));
        }
        GameOverReason::Stalemate => {
            lines.push(Line::from(vec![Span::styled(
                "DRAW / STALEMATE",
                header_style,
            )]));
        }
        GameOverReason::NetworkError(msg) => {
            lines.push(Line::from(vec![Span::styled(
                "DISCONNECTED:",
                header_style.fg(COLOR_DANGER),
            )]));
            lines.push(Line::from(vec![Span::styled(
                msg,
                Style::default().fg(COLOR_PRIMARY),
            )]));
        }
        GameOverReason::Desync => {
            lines.push(Line::from(vec![Span::styled(
                "STATE DESYNC DETECTED",
                header_style.fg(COLOR_DANGER),
            )]));
        }
    }

    let is_remote = app
        .session
        .as_ref()
        .is_some_and(|s| s.core.game_mode == GameMode::Remote);

    lines.push(Line::from(""));
    if is_remote {
        lines.push(Line::from(vec![Span::styled(
            "[Esc/Q] Main Menu",
            Style::default().fg(COLOR_SECONDARY),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "(Session Ended)",
            Style::default().fg(COLOR_DIM),
        )]));
    } else {
        lines.push(Line::from(vec![Span::styled(
            "[R] Restart Game",
            Style::default().fg(COLOR_SECONDARY),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "[S] Swap Sides",
            Style::default().fg(COLOR_SECONDARY),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "[Esc/Q] Main Menu",
            Style::default().fg(COLOR_SECONDARY),
        )]));
    }

    let is_local_win = if let GameOverReason::Win(p) = reason {
        app.session
            .as_ref()
            .is_some_and(|s| s.core.local_player == *p)
    } else {
        false
    };

    let border_color = match reason {
        GameOverReason::Win(_) if is_local_win => COLOR_SUCCESS,
        GameOverReason::Stalemate => COLOR_WARNING,
        _ => COLOR_DANGER,
    };

    let block = create_panel_block("GAME OVER", border_color)
        .padding(Padding::horizontal(to_u16(SIDEBAR_HPAD).max(1)));

    f.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Left)
            .wrap(ratatui::widgets::Wrap { trim: true })
            .block(block),
        area,
    );
}

/// Draws an error message when the terminal window is too small.
fn draw_size_error(f: &mut Frame, area: Rect) {
    let msg = vec![
        Line::from(vec![Span::styled(
            " TERMINAL TOO SMALL ",
            Style::default()
                .fg(COLOR_DANGER)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            format!(" Required: {MIN_UI_WIDTH}x{MIN_UI_HEIGHT}"),
            Style::default().fg(COLOR_PRIMARY),
        )]),
        Line::from(vec![Span::styled(
            format!(" Current:  {}x{}", area.width, area.height),
            Style::default().fg(COLOR_PRIMARY),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            " Please resize your terminal ",
            Style::default().fg(COLOR_SECONDARY),
        )]),
        Line::from(vec![Span::styled(
            " to continue. ",
            Style::default().fg(COLOR_SECONDARY),
        )]),
    ];

    let error_area = centered_rect(60, 10, area);
    f.render_widget(Clear, error_area);
    f.render_widget(
        Paragraph::new(msg)
            .alignment(Alignment::Center)
            .block(create_panel_block("SIZE ERROR", COLOR_DANGER)),
        error_area,
    );
}

/// Draws a generic network error screen.
fn draw_error_screen(f: &mut Frame, area: Rect, message: &str) {
    let error_area = centered_rect(60, 10, area);
    f.render_widget(Clear, error_area);

    let msg = vec![
        Line::from(vec![Span::styled(
            " NETWORK ERROR ",
            Style::default()
                .fg(COLOR_DANGER)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(message),
        Line::from(""),
        Line::from(vec![Span::styled(
            " [Esc/Q] Return to Main Menu ",
            Style::default().fg(COLOR_SECONDARY),
        )]),
    ];

    f.render_widget(
        Paragraph::new(msg)
            .alignment(Alignment::Center)
            .wrap(ratatui::widgets::Wrap { trim: true })
            .block(create_panel_block("NETWORK ERROR", COLOR_DANGER)),
        error_area,
    );
}

/// Renders a game piece at the specified coordinates.
///
/// Uses ASCII block characters to form a circular shape:
/// ```text
///  ▄███▄
/// ▐█████▌
///  ▀███▀
/// ```
fn render_piece(
    f: &mut Frame,
    x: u16,
    y: u16,
    player: Player,
    style: Style,
    min_y: u16,
    max_y: u16,
) {
    let color = match player {
        Player::Red => COLOR_TEAM_RED,
        Player::Yellow => COLOR_TEAM_YELLOW,
    };
    let piece_content = [" ▄███▄ ", "▐█████▌", " ▀███▀ "];
    for (i, line) in piece_content.iter().enumerate() {
        let target_y = y.saturating_add(to_u16(i));
        if target_y >= min_y && target_y <= max_y {
            f.buffer_mut()
                .set_string(x, target_y, line, Style::default().fg(color).patch(style));
        }
    }
}

/// Gets a formatted display name for a player, considering game mode and local player.
///
/// # Arguments
/// * `limit` - Maximum length for the name before truncation.
/// * `show_suffix` - Whether to append "[You]" to the local player's name in remote mode.
pub fn get_display_name(
    app: &App,
    session: &UiGameSession,
    player: Player,
    limit: usize,
    show_suffix: bool,
) -> String {
    let raw_name = if session.core.game_mode == GameMode::Single {
        if player == session.core.local_player {
            "You".to_string()
        } else {
            "AI".to_string()
        }
    } else if session.core.game_mode == GameMode::Remote {
        if player == session.core.local_player {
            app.network
                .as_ref()
                .map_or("You".to_string(), |n| n.local_name.clone())
        } else {
            app.network
                .as_ref()
                .map_or("Opponent".to_string(), |n| n.peer_name.clone())
        }
    } else {
        player.to_string()
    };

    let is_you_remote = show_suffix
        && session.core.game_mode == GameMode::Remote
        && player == session.core.local_player;
    let suffix = if is_you_remote { " [You]" } else { "" };
    let suffix_len = suffix.chars().count();

    // Ensure we don't truncate below zero
    let name_limit = limit.saturating_sub(suffix_len);

    let display_name = truncate_string(&raw_name, name_limit);

    format!("{display_name}{suffix}")
}

/// Truncates a string to a maximum length, appending an ellipsis if truncated.
/// Handles UTF-8 characters correctly.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.chars().count() > max_len {
        let mut t: String = s.chars().take(max_len.saturating_sub(1)).collect();
        t.push('…');
        t
    } else {
        s.to_string()
    }
}

/// Calculates a centered rectangle of a given size within a parent rectangle.
fn centered_rect(width: u16, height: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length((r.height.saturating_sub(height)) / 2),
            Constraint::Length(height),
            Constraint::Length((r.height.saturating_sub(height)) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length((r.width.saturating_sub(width)) / 2),
            Constraint::Length(width),
            Constraint::Length((r.width.saturating_sub(width)) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Generates a single Braille character from a list of dot positions (1-8).
///
/// Dot mapping:
/// 1  4
/// 2  5
/// 3  6
/// 7  8
fn braille_dots(dots: &[u8]) -> char {
    let mut mask = 0u8;
    for &dot in dots {
        if (1..=8).contains(&dot) {
            mask |= 1 << (dot - 1);
        }
    }
    // Braille Unicode range starts at 0x2800
    std::char::from_u32(0x2800 + u32::from(mask)).unwrap_or(' ')
}

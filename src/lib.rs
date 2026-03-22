//! # Connect 4 Library
//!
//! A high-performance implementation of Connect 4, featuring a sophisticated bitboard engine,
//! an advanced negamax AI with stochastic move selection, and LAN-based remote multiplayer
//! capabilities.
//!
//! This crate contains the core engine and library logic. For the TUI application, see the
//! accompanying binary.

pub mod ai;
pub mod config;
pub mod engine;
pub mod error;
pub mod game;
pub mod game_session;
pub mod network;
pub mod tt;
pub mod types;
pub mod zobrist;

pub use game_session::GameSession;
pub use types::{BoardSize, Difficulty, GameMode, Player};

//! # Error Handling
//!
//! Central library error handling via `Connect4Error`.
//! This module provides a unified error type for all library operations.

use thiserror::Error;

/// Errors that can occur during Connect 4 gameplay or engine operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Connect4Error {
    /// Attempted to place a piece in a column that is already full.
    #[error("Column is full")]
    ColumnFull,
    /// Attempted to access or play in a column index that is out of bounds.
    #[error("Invalid column index")]
    InvalidColumn,
}

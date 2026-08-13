//! Typed errors crossing the Rust ↔ frontend boundary. Mirrors the Python
//! bridge's error codes so the UI handles one vocabulary.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl BridgeError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            detail: None,
        }
    }

    pub fn with_detail(code: &str, message: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            detail: Some(detail.into()),
        }
    }

    pub fn engine_not_found(detail: impl Into<String>) -> Self {
        Self::with_detail(
            "ENGINE_NOT_FOUND",
            "The analysis engine could not be started.",
            detail,
        )
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new("INVALID_INPUT", message)
    }

    pub fn engine_crash(detail: impl Into<String>) -> Self {
        Self::with_detail(
            "ENGINE_CRASH",
            "The analysis engine stopped unexpectedly.",
            detail,
        )
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for BridgeError {}

pub type BridgeResult<T> = Result<T, BridgeError>;

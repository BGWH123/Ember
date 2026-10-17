pub mod ai;
pub mod application;
pub mod catalog;
pub mod storage;
pub use application::Application;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Busy(String),
    #[error("Database: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Evaluator(#[from] ember_evaluator::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

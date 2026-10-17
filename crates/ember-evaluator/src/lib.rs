pub mod installer;
pub mod local;
pub mod remote;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{future::Future, pin::Pin};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("Evaluator IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("Evaluator protocol: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Evaluator HTTP: {0}")]
    Http(#[from] reqwest::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Hello {
    pub protocol_version: u32,
    pub catalog_version: String,
    pub engine_version: String,
}
impl Hello {
    pub fn validate(&self, catalog: &str) -> Result<()> {
        if self.protocol_version != PROTOCOL_VERSION || self.catalog_version != catalog {
            return Err(Error::Invalid(
                "Evaluator protocol/catalog is incompatible; install a matching module".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Budget {
    pub max_cpu_ms: u64,
    pub max_memory_bytes: u64,
    pub max_output_bytes: u64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            max_cpu_ms: 10_000,
            max_memory_bytes: 1024 * 1024 * 1024,
            max_output_bytes: 16_000,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationRequest {
    pub request_id: String,
    pub task_id: String,
    pub code: String,
    pub test_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub resource_budget: Budget,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationResult {
    pub outcome: String,
    pub passed: u64,
    pub total: u64,
    #[serde(default)]
    pub results: Vec<Value>,
    pub total_time_ms: f64,
    pub error: Option<String>,
    pub error_code: Option<String>,
    #[serde(default)]
    pub timings: Value,
    #[serde(default)]
    pub resources: Value,
}
impl EvaluationResult {
    pub fn crashed(message: String) -> Self {
        Self {
            outcome: "crashed".into(),
            passed: 0,
            total: 0,
            results: vec![],
            total_time_ms: 0.0,
            error: Some(message),
            error_code: Some("evaluator_unavailable".into()),
            timings: Value::Null,
            resources: Value::Null,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if !["passed", "failed", "timeout", "crashed", "cancelled"].contains(&self.outcome.as_str())
            || self.passed > self.total
            || !self.total_time_ms.is_finite()
            || self.total_time_ms < 0.0
            || (self.outcome == "passed" && (self.total == 0 || self.passed != self.total))
        {
            return Err(Error::Invalid("Invalid evaluator verdict".into()));
        }
        Ok(())
    }
}
pub trait Evaluator: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        request: EvaluationRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EvaluationResult>> + Send + 'a>>;
}

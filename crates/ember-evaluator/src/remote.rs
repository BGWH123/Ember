//! Inactive HTTP boundary reserved for future online evaluation.
use crate::{Error, EvaluationRequest, EvaluationResult, Result};

/// Always rejects locally; this placeholder never transmits submitted code.
pub async fn evaluate_over_http(
    _client: &reqwest::Client,
    _endpoint: &str,
    _request: &EvaluationRequest,
) -> Result<EvaluationResult> {
    // TODO: 线上评测预留功能。后续在这里接入 HTTP；当前不可用，不发送请求。
    Err(Error::Invalid("Online evaluation is not available".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reserved_http_boundary_never_connects() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let request = EvaluationRequest {
            request_id: "reserved".into(),
            task_id: "relu".into(),
            code: "def relu(x): return x".into(),
            test_indices: None,
            resource_budget: Default::default(),
        };
        let error = evaluate_over_http(&reqwest::Client::new(), &endpoint, &request)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not available"));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }
}

use crate::{
    Error, EvaluationRequest, EvaluationResult, Evaluator, Hello, Result, MAX_MESSAGE_BYTES,
};
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

pub struct RemoteEvaluator {
    client: reqwest::Client,
    base: reqwest::Url,
    token: String,
    catalog: String,
}
impl RemoteEvaluator {
    pub fn new(base: &str, token: String, catalog: String) -> Result<Self> {
        let base = reqwest::Url::parse(&format!("{}/", base.trim_end_matches('/')))
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let loopback = matches!(base.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        if base.scheme() != "https" && !(base.scheme() == "http" && loopback) {
            return Err(Error::Invalid(
                "Remote evaluator requires HTTPS (HTTP allowed for loopback development)".into(),
            ));
        }
        if !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::Invalid("Invalid remote endpoint".into()));
        }
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            base,
            token,
            catalog,
        })
    }
    async fn read(response: reqwest::Response) -> Result<Value> {
        let mut stream = response.error_for_status()?.bytes_stream();
        let mut data = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if data.len() + chunk.len() > MAX_MESSAGE_BYTES {
                return Err(Error::Invalid("Remote response is too large".into()));
            }
            data.extend_from_slice(&chunk);
        }
        Ok(serde_json::from_slice(&data)?)
    }
}
impl Evaluator for RemoteEvaluator {
    fn evaluate<'a>(
        &'a self,
        request: EvaluationRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<EvaluationResult>> + Send + 'a>>
    {
        Box::pin(async move {
            let url = |path| {
                self.base
                    .join(path)
                    .map_err(|e| Error::Invalid(e.to_string()))
            };
            let hello: Hello = serde_json::from_value(
                Self::read(
                    self.client
                        .get(url("v1/hello")?)
                        .bearer_auth(&self.token)
                        .send()
                        .await?,
                )
                .await?,
            )?;
            hello.validate(&self.catalog)?;
            let response = Self::read(
                self.client
                    .post(url("v1/evaluate")?)
                    .bearer_auth(&self.token)
                    .json(&request)
                    .send()
                    .await?,
            )
            .await?;
            if response["requestId"] != request.request_id {
                return Err(Error::Invalid("Remote request ID mismatch".into()));
            }
            let result: EvaluationResult = serde_json::from_value(response["result"].clone())?;
            result.validate()?;
            Ok(result)
        })
    }
}

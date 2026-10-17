use crate::{Error, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};

pub async fn help(body: Value) -> Result<Value> {
    let config = &body["config"];
    let base = config["baseUrl"]
        .as_str()
        .unwrap_or("")
        .trim_end_matches('/');
    let key = config["apiKey"].as_str().unwrap_or("");
    let model = config["model"].as_str().unwrap_or("");
    if base.is_empty() || key.is_empty() || model.is_empty() {
        return Err(Error::Invalid("Missing baseUrl, apiKey or model".into()));
    }
    let url = reqwest::Url::parse(&format!("{base}/chat/completions"))
        .map_err(|e| Error::Invalid(e.to_string()))?;
    if !["https", "http"].contains(&url.scheme()) {
        return Err(Error::Invalid("AI endpoint must use HTTP(S)".into()));
    }
    let custom = body["customPrompt"].as_str().unwrap_or("");
    let language = if body["locale"] == "zh" {
        "请用中文回答，使用 Markdown。"
    } else {
        "Reply in English using Markdown."
    };
    let prompt=format!("{language}\nProblem: {}\nFunction: {}\nDescription:\n{}\nSample tests:\n{}\nCustom prompt:\n{}\nReference solution:\n{}\nLearner current code:\n{}",body["problemTitle"],body["functionName"],body["description"].as_str().unwrap_or(""),body["sampleTests"],custom,if custom.trim().is_empty(){body["solutionCode"].as_str().unwrap_or("")}else{""},body["userCode"].as_str().unwrap_or(""));
    if prompt.len() > 256_000 {
        return Err(Error::Invalid("AI context too large".into()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(ember_evaluator::Error::from)?;
    let response=client.post(url).bearer_auth(key).json(&json!({"model":model,"stream":false,"temperature":0.4,"messages":[{"role":"system","content":"You are a PyTorch tutor. Analyze learner code when present; explain the idea, mistakes and next validation steps. Follow the custom prompt. Give hints and small snippets, never a complete submit-ready implementation."},{"role":"user","content":prompt}]})).send().await.map_err(ember_evaluator::Error::from)?;
    let status = response.status();
    let mut stream = response.bytes_stream();
    let mut bytes = vec![];
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(ember_evaluator::Error::from)?;
        if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
            return Err(Error::Invalid("AI response too large".into()));
        }
        bytes.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        return Err(Error::Invalid(format!("AI endpoint returned {status}")));
    }
    let data: Value = serde_json::from_slice(&bytes)?;
    let content = &data["choices"][0]["message"]["content"];
    let guidance = content.as_str().map(str::to_owned).unwrap_or_else(|| {
        content
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter(|p| p["type"] == "text")
                    .filter_map(|p| p["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    });
    if guidance.trim().is_empty() {
        return Err(Error::Invalid("Model returned an empty response".into()));
    }
    Ok(json!({"guidance":guidance,"model":data["model"].as_str().unwrap_or(model)}))
}

//! Scriptable adapter to the same application core; no WebView/IPC memory is included.
use ember_core::Application;
use ember_evaluator::local::Launch;
use serde_json::{json, Value};
use std::{
    io::{BufRead, Write},
    path::PathBuf,
};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("EMBER_ROOT")?);
    let data = PathBuf::from(std::env::var("EMBER_DATA")?);
    let public_key = std::env::var("EMBER_EVALUATOR_PUBLIC_KEY").ok();
    let app = Application::open(&root.join("content/catalog"), &data, public_key.as_deref())?;
    if let Ok(python) = std::env::var("EMBER_DEV_PYTHON") {
        app.use_development_evaluator(Launch {
            executable: python.into(),
            args: vec![root
                .join("modules/evaluator/runner.py")
                .to_string_lossy()
                .into_owned()],
            working_dir: root.clone(),
        })?;
    }
    println!("{}", json!({"ready":true,"status":app.status()?}));
    std::io::stdout().flush()?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.len() > 2 * 1024 * 1024 {
            break;
        }
        let request: Value = serde_json::from_str(&line)?;
        let result = match request["command"].as_str().unwrap_or("request") {
            "status" => app.status(),
            "load_draft" => app
                .storage
                .draft(request["id"].as_str().unwrap_or_default())
                .map(|v| json!(v)),
            "save_draft" => app
                .storage
                .save_draft(
                    request["id"].as_str().unwrap_or_default(),
                    request["code"].as_str().unwrap_or_default(),
                )
                .map(|_| Value::Null),
            "install" => {
                app.install(request["url"].as_str().unwrap_or_default())
                    .await
            }
            "activate" => {
                app.activate(request["version"].as_str().unwrap_or_default())
                    .await
            }
            "shutdown" => {
                app.shutdown_local().await;
                Ok(json!({"stopped":true}))
            }
            _ => {
                app.request(
                    request["path"].as_str().unwrap_or(""),
                    request["payload"].clone(),
                )
                .await
            }
        };
        println!(
            "{}",
            match result {
                Ok(value) => json!({"ok":true,"body":value}),
                Err(error) => json!({"ok":false,"error":error.to_string()}),
            }
        );
        std::io::stdout().flush()?;
    }
    app.shutdown_local().await;
    Ok(())
}

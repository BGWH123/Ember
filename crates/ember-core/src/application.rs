use crate::{catalog::Catalog, storage::Storage, Error, Result};
#[cfg(any(debug_assertions, feature = "developer-tools"))]
use ember_evaluator::local::Launch;
use ember_evaluator::{
    installer::ModuleManager, local::LocalEvaluator, EvaluationRequest, EvaluationResult, Evaluator,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmitRequest {
    pub task_id: String,
    pub code: String,
    #[serde(default)]
    pub test_indices: Option<Vec<usize>>,
    pub idempotency_key: Option<String>,
}
struct Selection {
    provider: Option<Arc<dyn Evaluator>>,
    local: Option<Arc<LocalEvaluator>>,
    mode: String,
}
pub struct Application {
    pub catalog: Catalog,
    pub storage: Storage,
    pub modules: ModuleManager,
    selection: Mutex<Selection>,
    gate: tokio::sync::Mutex<()>,
}
impl Application {
    pub fn open(assets: &Path, data: &Path, key: Option<&str>) -> Result<Self> {
        std::fs::create_dir_all(data)?;
        let catalog = Catalog::load(assets)?;
        let modules = ModuleManager::new(data.join("evaluators"), catalog.version.clone(), key)?;
        let app = Self {
            catalog,
            storage: Storage::open(&data.join("ember-desktop.db"))?,
            modules,
            selection: Mutex::new(Selection {
                provider: None,
                local: None,
                mode: "local".into(),
            }),
            gate: tokio::sync::Mutex::new(()),
        };
        // An old/incompatible module must not prevent offline learning after a shell upgrade.
        let _ = app.load_active();
        Ok(app)
    }
    fn selection(&self) -> Result<std::sync::MutexGuard<'_, Selection>> {
        self.selection
            .lock()
            .map_err(|_| Error::Invalid("Evaluator selection poisoned".into()))
    }
    fn load_active(&self) -> Result<()> {
        let provider = self
            .modules
            .active()?
            .map(|(_, launch)| Arc::new(LocalEvaluator::new(launch, self.catalog.version.clone())));
        let mut selection = self.selection()?;
        selection.provider = provider.clone().map(|p| p as Arc<dyn Evaluator>);
        selection.local = provider;
        selection.mode = "local".into();
        Ok(())
    }
    #[cfg(any(debug_assertions, feature = "developer-tools"))]
    pub fn use_development_evaluator(&self, launch: Launch) -> Result<()> {
        self.select_launch(launch)
    }
    #[cfg(any(debug_assertions, feature = "developer-tools"))]
    fn select_launch(&self, launch: Launch) -> Result<()> {
        let local = Arc::new(LocalEvaluator::new(launch, self.catalog.version.clone()));
        let mut selection = self.selection()?;
        selection.provider = Some(local.clone());
        selection.local = Some(local);
        selection.mode = "development".into();
        Ok(())
    }
    pub fn status(&self) -> Result<Value> {
        let selection = self.selection()?;
        let active_state = self.modules.active();
        let module_error = active_state.as_ref().err().map(ToString::to_string);
        let active = active_state
            .ok()
            .flatten()
            .map(|(manifest, _)| manifest.version);
        Ok(
            json!({"moduleError":module_error,"mode":selection.mode,"available":selection.provider.is_some(),"activeVersion":active,"installed":self.modules.installed()?,"catalogVersion":self.catalog.version,"protocolVersion":1,"target":ember_evaluator::installer::target(),"shellVersion":env!("CARGO_PKG_VERSION")}),
        )
    }
    pub async fn shutdown_local(&self) {
        let local = self.selection().ok().and_then(|s| s.local.clone());
        if let Some(local) = local {
            local.shutdown().await;
        }
    }
    pub async fn reap_idle(&self) {
        let local = self.selection().ok().and_then(|s| s.local.clone());
        if let Some(local) = local {
            local.reap_if_idle().await;
        }
    }
    pub async fn install(&self, url: &str) -> Result<Value> {
        let _guard = self
            .gate
            .try_lock()
            .map_err(|_| Error::Busy("Wait for evaluation to finish".into()))?;
        let version = self.modules.download(url).await?;
        self.activate_verified(&version).await?;
        self.status()
    }
    pub async fn import_module(&self, manifest: PathBuf, archive: PathBuf) -> Result<Value> {
        let _guard = self
            .gate
            .try_lock()
            .map_err(|_| Error::Busy("Wait for evaluation to finish".into()))?;
        let manager = self.modules.clone();
        let version = tokio::task::spawn_blocking(move || manager.import(&manifest, &archive))
            .await
            .map_err(|e| Error::Invalid(e.to_string()))??;
        self.activate_verified(&version).await?;
        self.status()
    }
    pub async fn activate(&self, version: &str) -> Result<Value> {
        let _guard = self
            .gate
            .try_lock()
            .map_err(|_| Error::Busy("Wait for evaluation to finish".into()))?;
        self.activate_verified(version).await?;
        self.status()
    }
    async fn activate_verified(&self, version: &str) -> Result<()> {
        let candidate =
            LocalEvaluator::new(self.modules.launch(version)?, self.catalog.version.clone());
        candidate.probe().await?;
        self.modules.activate(version)?;
        self.shutdown_local().await;
        self.load_active()
    }
    pub async fn submit(&self, request: SubmitRequest, mode: &str) -> Result<Value> {
        if !["sample", "submit"].contains(&mode)
            || request.code.len() > 64000
            || request.code.trim().is_empty()
        {
            return Err(Error::Invalid(
                "Invalid submission mode or source (1–64000 bytes)".into(),
            ));
        }
        let problem = self.catalog.problem(&request.task_id)?;
        let count = problem["tests"].as_array().map_or(0, Vec::len);
        if mode == "submit" && request.test_indices.is_some() {
            return Err(Error::Invalid("Full submission must run all tests".into()));
        }
        let indices = if mode == "sample" {
            Some(request.test_indices.clone().unwrap_or_else(|| vec![0]))
        } else {
            None
        };
        if let Some(indices) = &indices {
            let mut unique = std::collections::HashSet::new();
            if indices.is_empty() || indices.iter().any(|i| *i >= count || !unique.insert(i)) {
                return Err(Error::Invalid("Select distinct valid tests".into()));
            }
        }
        let key = request
            .idempotency_key
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        if key.is_empty() || key.len() > 128 {
            return Err(Error::Invalid("Invalid idempotency key".into()));
        }
        let fingerprint = hex::encode(Sha256::digest(serde_json::to_vec(&json!([
            request.task_id,
            request.code,
            mode,
            indices
        ]))?));
        let _guard = self
            .gate
            .try_lock()
            .map_err(|_| Error::Busy("An evaluation is already running".into()))?;
        let provider = self.selection()?.provider.clone().ok_or_else(|| {
            Error::Invalid(
                "No local evaluator installed. Open Settings to install a local module.".into(),
            )
        })?;
        let (id, created) = self.storage.accept(
            &uuid::Uuid::new_v4().to_string(),
            &request.task_id,
            &request.code,
            mode,
            &key,
            &fingerprint,
        )?;
        if created {
            self.storage.running(&id)?;
            let started = std::time::Instant::now();
            let evaluation = EvaluationRequest {
                request_id: id.clone(),
                task_id: request.task_id,
                code: request.code,
                test_indices: indices,
                resource_budget: Default::default(),
            };
            let mut result = provider
                .evaluate(evaluation)
                .await
                .unwrap_or_else(|e| EvaluationResult::crashed(e.to_string()));
            // A provider cannot claim a partial test set as a successful full submission.
            let expected = if mode == "submit" {
                count
            } else {
                request.test_indices.as_ref().map_or(1, Vec::len)
            } as u64;
            if result.outcome == "passed" && result.total != expected {
                result = EvaluationResult::crashed(
                    "Evaluator test count does not match requested tests".into(),
                );
            }
            if !result.timings.is_object() {
                result.timings = json!({});
            }
            result.timings["desktopEvaluationMs"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            self.storage.finish(&id, &result)?;
        }
        let mut result = self.storage.result(&id)?;
        result["submission_id"] = json!(id);
        Ok(result)
    }
    // Compatibility presentation facade: keeps existing UI response shapes while domain work stays typed.
    pub async fn request(&self, path: &str, payload: Value) -> Result<Value> {
        if path.len() > 1024 {
            return Err(Error::Invalid("Invalid command path".into()));
        }
        match path {
            "/api/stats" => Ok(self.catalog.stats()),
            "/api/problems" => Ok(self.catalog.list()),
            "/api/progress" => Ok(json!({"progress":self.storage.progress()?})),
            "/api/paths" => Ok(self.catalog.paths(&self.storage.progress()?)),
            "/api/run" => {
                self.submit(serde_json::from_value(payload)?, "sample")
                    .await
            }
            "/api/submit" => {
                self.submit(serde_json::from_value(payload)?, "submit")
                    .await
            }
            "/api/ai-help" => crate::ai::help(payload).await,
            "/api/ai-help/status" => Ok(json!({"configured":false})),
            _ => {
                if let Some(id) = path.strip_prefix("/api/problems/") {
                    return self.catalog.problem(id);
                }
                if let Some(id) = path.strip_prefix("/api/solutions/") {
                    return self.catalog.solution(id);
                }
                if let Some(id) = path.strip_prefix("/api/paths/") {
                    return self.catalog.path(id, &self.storage.progress()?);
                }
                if let Some(id) = path.strip_prefix("/api/submissions/status/") {
                    return self.storage.result(id);
                }
                if let Some(id) = path.strip_prefix("/api/submissions/") {
                    return self.storage.history(id);
                }
                Err(Error::NotFound("Unknown desktop operation".into()))
            }
        }
    }
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use ember_core::Application;
use serde_json::Value;
use std::sync::Arc;
use tauri::{Manager, State};
type AppState = Arc<Application>;
#[tauri::command]
async fn request(app: State<'_, AppState>, path: String, payload: Value) -> Result<Value, String> {
    app.request(&path, payload).await.map_err(|e| e.to_string())
}
#[tauri::command]
fn evaluator_status(app: State<'_, AppState>) -> Result<Value, String> {
    app.status().map_err(|e| e.to_string())
}
#[tauri::command]
async fn configure_evaluator(
    app: State<'_, AppState>,
    mode: String,
    url: String,
    token: String,
) -> Result<Value, String> {
    app.configure(&mode, &url, token)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
async fn install_evaluator(app: State<'_, AppState>, url: String) -> Result<Value, String> {
    app.install(&url).await.map_err(|e| e.to_string())
}
#[tauri::command]
async fn import_evaluator(
    app: State<'_, AppState>,
    manifest: String,
    archive: String,
) -> Result<Value, String> {
    app.import_module(manifest.into(), archive.into())
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
async fn activate_evaluator(app: State<'_, AppState>, version: String) -> Result<Value, String> {
    app.activate(&version).await.map_err(|e| e.to_string())
}
#[tauri::command]
async fn load_draft(app: State<'_, AppState>, id: String) -> Result<Option<String>, String> {
    let core = app.inner().clone();
    tokio::task::spawn_blocking(move || core.storage.draft(&id))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}
#[tauri::command]
async fn save_draft(app: State<'_, AppState>, id: String, code: String) -> Result<(), String> {
    let core = app.inner().clone();
    tokio::task::spawn_blocking(move || core.storage.save_draft(&id, &code))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}
#[tauri::command]
async fn legacy_users(path: String) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || {
        ember_core::storage::Storage::legacy_users(std::path::Path::new(&path))
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}
#[tauri::command]
async fn import_legacy(
    app: State<'_, AppState>,
    path: String,
    user: String,
) -> Result<Value, String> {
    let core = app.inner().clone();
    tokio::task::spawn_blocking(move || {
        core.storage
            .import_legacy(std::path::Path::new(&path), &user)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}
#[tauri::command]
async fn check_shell_update(handle: tauri::AppHandle) -> Result<Value, String> {
    if !handle.config().plugins.0.contains_key("updater") {
        return Err("This build has no shell update channel configured".into());
    }
    use tauri_plugin_updater::UpdaterExt;
    let update = handle
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?;
    Ok(update
        .map(|u| serde_json::json!({"version":u.version,"body":u.body}))
        .unwrap_or(Value::Null))
}
#[tauri::command]
async fn install_shell_update(handle: tauri::AppHandle) -> Result<(), String> {
    if !handle.config().plugins.0.contains_key("updater") {
        return Err("This build has no shell update channel configured".into());
    }
    use tauri_plugin_updater::UpdaterExt;
    if let Some(update) = handle
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?
    {
        update
            .download_and_install(|_, _| {}, || {})
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn main() {
    tauri::Builder::default()
        .setup(|app| {
            if app.config().plugins.0.contains_key("updater") {
                app.handle()
                    .plugin(tauri_plugin_updater::Builder::new().build())?;
            }
            let assets = app.path().resource_dir()?.join("catalog");
            #[cfg(debug_assertions)]
            let assets = if assets.is_dir() {
                assets
            } else {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../content/catalog")
            };
            let core = Arc::new(Application::open(
                &assets,
                &app.path().app_data_dir()?,
                option_env!("EMBER_EVALUATOR_PUBLIC_KEY"),
            )?);
            #[cfg(debug_assertions)]
            if let Ok(python) = std::env::var("EMBER_DEV_PYTHON") {
                let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
                core.use_development_evaluator(ember_evaluator::local::Launch {
                    executable: python.into(),
                    args: vec![root
                        .join("modules/evaluator/runner.py")
                        .to_string_lossy()
                        .into_owned()],
                    working_dir: root,
                })?;
            }
            let idle = core.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                    idle.reap_idle().await;
                }
            });
            app.manage(core);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            request,
            evaluator_status,
            configure_evaluator,
            install_evaluator,
            import_evaluator,
            activate_evaluator,
            load_draft,
            save_draft,
            legacy_users,
            import_legacy,
            check_shell_update,
            install_shell_update
        ])
        .build(tauri::generate_context!())
        .expect("Failed to initialize Ember")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                if let Some(core) = app.try_state::<AppState>() {
                    tauri::async_runtime::block_on(core.shutdown_local());
                }
            }
        });
}

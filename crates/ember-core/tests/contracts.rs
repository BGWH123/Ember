use ember_core::{catalog::Catalog, storage::Storage, Application};
use ember_evaluator::EvaluationResult;
use serde_json::json;
use std::path::PathBuf;
fn assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../content/catalog")
}
fn verdict(outcome: &str) -> EvaluationResult {
    EvaluationResult {
        outcome: outcome.into(),
        passed: 1,
        total: 1,
        results: vec![],
        total_time_ms: 12.5,
        error: None,
        error_code: None,
        timings: json!({}),
        resources: json!({}),
    }
}
#[test]
fn catalog_and_paths_preserve_public_data() {
    let c = Catalog::load(&assets()).unwrap();
    let stats = c.stats();
    assert_eq!(
        stats["total"].as_u64(),
        Some(
            stats["easy"].as_u64().unwrap()
                + stats["medium"].as_u64().unwrap()
                + stats["hard"].as_u64().unwrap()
        )
    );
    let source: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets().join("problems.json")).unwrap()).unwrap();
    assert_eq!(
        c.list()["total"],
        source["problems"].as_array().unwrap().len()
    );
    for p in source["problems"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let detail = c.problem(id).unwrap();
        assert_eq!(detail["tests"], p["tests"]);
        assert!(detail["starterCode"]
            .as_str()
            .unwrap()
            .contains("import torch"));
    }
    assert!(c.problem("missing").is_err());
    let paths = c.paths(&json!({}));
    let id = paths["paths"][0]["id"].as_str().unwrap();
    let path = c.path(id, &json!({"relu":{"status":"solved"}})).unwrap();
    assert_eq!(path["total"], path["problems"].as_array().unwrap().len());
}
#[test]
fn transactions_idempotency_and_sample_progress() {
    let dir = tempfile::tempdir().unwrap();
    let db = Storage::open(&dir.path().join("test.db")).unwrap();
    assert!(
        db.accept("one", "relu", "code", "sample", "key", "a")
            .unwrap()
            .1
    );
    assert!(
        !db.accept("other", "relu", "code", "sample", "key", "a")
            .unwrap()
            .1
    );
    assert!(db
        .accept("other", "relu", "code", "sample", "key", "b")
        .is_err());
    assert!(db
        .accept("two", "relu", "code", "submit", "two", "two")
        .is_err());
    db.running("one").unwrap();
    db.finish("one", &verdict("passed")).unwrap();
    assert_eq!(db.progress().unwrap()["relu"]["status"], "attempted");
    db.accept("two", "relu", "code", "submit", "two", "two")
        .unwrap();
    db.finish("two", &verdict("passed")).unwrap();
    db.finish("two", &verdict("passed")).unwrap();
    assert_eq!(db.progress().unwrap()["relu"]["attempts"], 2);
    assert_eq!(db.progress().unwrap()["relu"]["status"], "solved");
    db.accept("three", "relu", "code", "submit", "three", "three")
        .unwrap();
    db.finish("three", &verdict("failed")).unwrap();
    assert_eq!(db.progress().unwrap()["relu"]["status"], "solved");
    assert_eq!(db.history("relu").unwrap().as_array().unwrap().len(), 3);
    db.save_draft("relu", "draft").unwrap();
    assert_eq!(db.draft("relu").unwrap().as_deref(), Some("draft"));
}
#[test]
fn history_indexes_migrate_existing_database_and_preserve_ties() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    drop(Storage::open(&path).unwrap());
    let mut conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch("DROP INDEX submissions_problem_created; DROP INDEX submissions_pending;")
        .unwrap();
    let tx = conn.transaction().unwrap();
    for i in 0..120 {
        let id = format!("row-{i}");
        tx.execute("INSERT INTO submissions VALUES(?,'relu','code','submit',?,?,'completed',NULL,'same-time')", rusqlite::params![id,id,id]).unwrap();
    }
    tx.commit().unwrap();
    let db = Storage::open(&path).unwrap();
    let history = db.history("relu").unwrap();
    let rows = history.as_array().unwrap();
    assert_eq!(rows.len(), 100);
    assert_eq!(rows[0]["id"], "row-119");
    assert_eq!(rows[99]["id"], "row-20");
    // EXPLAIN does not execute the schema-cookie check on the fixture connection.
    let conn = rusqlite::Connection::open(&path).unwrap();
    for query in [
        "SELECT id,code,result,created_at FROM submissions WHERE problem_id='relu' ORDER BY created_at DESC,rowid DESC LIMIT 100",
        "SELECT count(*) FROM submissions WHERE status IN ('queued','running')",
    ] {
        let details: Vec<String> = conn.prepare(&format!("EXPLAIN QUERY PLAN {query}")).unwrap()
            .query_map([], |r| r.get(3)).unwrap().collect::<Result<_,_>>().unwrap();
        assert!(details.iter().any(|d| d.contains("INDEX submissions_")), "{details:?}");
        assert!(!details.iter().any(|d| d.contains("TEMP B-TREE") || d == "SCAN submissions"), "{details:?}");
    }
}
#[test]
fn restart_marks_pending_as_crashed_without_executing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    {
        let db = Storage::open(&path).unwrap();
        db.accept("one", "relu", "code", "submit", "key", "a")
            .unwrap();
        db.running("one").unwrap();
    }
    let db = Storage::open(&path).unwrap();
    let result = db.result("one").unwrap();
    assert_eq!(result["status"], "completed");
    assert_eq!(result["outcome"], "crashed");
    assert_eq!(db.progress().unwrap(), json!({}));
}
#[tokio::test]
async fn learning_without_evaluator_and_invalid_submissions() {
    let dir = tempfile::tempdir().unwrap();
    let app = Application::open(&assets(), dir.path(), None).unwrap();
    assert_eq!(app.status().unwrap()["available"], false);
    assert_eq!(app.status().unwrap()["mode"], "local");
    let unavailable = app
        .request(
            "/api/submit",
            json!({"taskId":"relu","code":"def relu(x): return x"}),
        )
        .await
        .unwrap_err();
    assert!(unavailable.to_string().contains("local evaluator"));
    assert_eq!(
        app.request("/api/submissions/relu", json!(null))
            .await
            .unwrap(),
        json!([])
    );
    assert!(app.request("/api/problems", json!(null)).await.is_ok());
    assert!(app
        .request("/api/solutions/relu", json!(null))
        .await
        .is_ok());
    for payload in [
        json!({"taskId":"relu","code":"x","testIndices":[]}),
        json!({"taskId":"relu","code":"x","testIndices":[0,0]}),
        json!({"taskId":"relu","code":"x","testIndices":[999]}),
        json!({"taskId":"relu","code":""}),
    ] {
        assert!(app.request("/api/run", payload).await.is_err());
    }
    assert!(app
        .request(
            "/api/submit",
            json!({"taskId":"relu","code":"x","testIndices":[0]})
        )
        .await
        .is_err());
}
#[test]
fn legacy_import_is_readonly_scoped_and_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.db");
    let source = rusqlite::Connection::open(&path).unwrap();
    source.execute_batch("CREATE TABLE progress(problem_id TEXT,status TEXT,attempts INTEGER,best_time_ms REAL,solved_at TEXT,user_id TEXT); INSERT INTO progress VALUES('relu','solved',3,12,NULL,'alice'); INSERT INTO progress VALUES('softmax','solved',1,10,NULL,'bob'); CREATE TABLE submission(id TEXT,user_id TEXT,status TEXT,problem_id TEXT,source_code TEXT,mode TEXT,created_at TEXT); CREATE TABLE judge_job(id TEXT,submission_id TEXT); CREATE TABLE judge_result(job_id TEXT,outcome TEXT,passed_count INTEGER,total_count INTEGER,total_ms REAL,details_json TEXT,error TEXT,error_code TEXT);").unwrap();
    drop(source);
    let before = std::fs::read(&path).unwrap();
    let db = Storage::open(&dir.path().join("new.db")).unwrap();
    db.import_legacy(&path, "alice").unwrap();
    assert!(db.import_legacy(&path, "alice").is_err());
    assert_eq!(db.progress().unwrap()["relu"]["attempts"], 3);
    assert!(db.progress().unwrap().get("softmax").is_none());
    assert_eq!(before, std::fs::read(path).unwrap());
}

#[tokio::test]
async fn broken_or_incompatible_active_module_does_not_block_learning() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("evaluators")).unwrap();
    std::fs::write(
        dir.path().join("evaluators/active.json"),
        r#"{"payload":"{}","signature":"00"}"#,
    )
    .unwrap();
    let app = Application::open(&assets(), dir.path(), None).unwrap();
    assert_eq!(app.status().unwrap()["available"], false);
    assert!(app.status().unwrap()["moduleError"].is_string());
    assert!(app.request("/api/problems/relu", json!(null)).await.is_ok());
}

#[tokio::test]
async fn ai_proxy_uses_custom_context_and_parses_text_parts() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let n = stream.read(&mut buffer).unwrap();
            assert!(n > 0);
            request.extend_from_slice(&buffer[..n]);
            if let Some(end) = request.windows(4).position(|p| p == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]);
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap();
                if request.len() >= end + 4 + length {
                    let payload: serde_json::Value =
                        serde_json::from_slice(&request[end + 4..end + 4 + length]).unwrap();
                    let body = r#"{"model":"fixture-model","choices":[{"message":{"content":[{"type":"text","text":"Check the tensor shape."}]}}]}"#;
                    write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
                    return payload;
                }
            }
        }
    });
    let response=ember_core::ai::help(json!({"config":{"baseUrl":format!("http://{address}/v1"),"apiKey":"fixture-token","model":"fixture-model"},"problemTitle":"ReLU","functionName":"relu","description":"Test description","sampleTests":[],"customPrompt":"Explain dimensions","solutionCode":"REFERENCE_SHOULD_BE_OMITTED","userCode":"USER_CODE_MARKER","locale":"zh"})).await.unwrap();
    assert_eq!(response["guidance"], "Check the tensor shape.");
    let payload = server.join().unwrap();
    let prompt = payload["messages"][1]["content"].as_str().unwrap();
    assert!(prompt.contains("Explain dimensions"));
    assert!(prompt.contains("USER_CODE_MARKER"));
    assert!(!prompt.contains("REFERENCE_SHOULD_BE_OMITTED"));
    assert!(ember_core::ai::help(json!({})).await.is_err());
}

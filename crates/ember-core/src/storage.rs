use crate::{Error, Result};
use ember_evaluator::EvaluationResult;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::{path::Path, sync::Mutex};

pub struct Storage {
    // Drop the reader first so the final writable close can checkpoint/clean WAL.
    reader: Mutex<Connection>,
    connection: Mutex<Connection>,
}

fn stamp() -> String {
    chrono::Utc::now().to_rfc3339()
}
impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS desktop_schema(version INTEGER NOT NULL);
            INSERT INTO desktop_schema SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM desktop_schema);
            CREATE TABLE IF NOT EXISTS submissions(id TEXT PRIMARY KEY,problem_id TEXT NOT NULL,code TEXT NOT NULL,mode TEXT NOT NULL,idempotency_key TEXT NOT NULL UNIQUE,fingerprint TEXT NOT NULL,status TEXT NOT NULL,result TEXT,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS progress(problem_id TEXT PRIMARY KEY,status TEXT NOT NULL,attempts INTEGER NOT NULL,best_time_ms REAL,solved_at TEXT);
            CREATE TABLE IF NOT EXISTS drafts(problem_id TEXT PRIMARY KEY,code TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS imports(source TEXT NOT NULL,user_id TEXT NOT NULL,PRIMARY KEY(source,user_id));")?;
        let version: i64 =
            connection.query_row("SELECT version FROM desktop_schema", [], |r| r.get(0))?;
        if version != 1 {
            return Err(Error::Invalid("Unsupported desktop database schema".into()));
        }
        // Additive migration for existing v1 databases. Reverse traversal also orders
        // equal timestamps by rowid DESC without a temporary sorting table.
        connection.execute_batch(
            "CREATE INDEX IF NOT EXISTS submissions_problem_created ON submissions(problem_id,created_at);
             CREATE INDEX IF NOT EXISTS submissions_pending ON submissions(status) WHERE status IN ('queued','running');",
        )?;
        connection.execute("UPDATE submissions SET status='completed',result=? WHERE status IN ('queued','running')",[serde_json::to_string(&EvaluationResult::crashed("Application stopped before evaluation completed; submit again".into()))?])?;
        // WAL readers see committed snapshots while a writer is waiting for fsync.
        // Keep writes serialized and durable without lending their connection lock
        // to history/progress requests on the async runtime.
        let reader = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        reader.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(Self {
            connection: Mutex::new(connection),
            reader: Mutex::new(reader),
        })
    }
    fn conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| Error::Invalid("Database lock poisoned".into()))
    }
    fn read_conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.reader
            .lock()
            .map_err(|_| Error::Invalid("Database reader lock poisoned".into()))
    }
    pub fn accept(
        &self,
        id: &str,
        problem: &str,
        code: &str,
        mode: &str,
        key: &str,
        fingerprint: &str,
    ) -> Result<(String, bool)> {
        let conn = self.conn()?;
        let prior: Option<(String, String)> = conn
            .query_row(
                "SELECT id,fingerprint FROM submissions WHERE idempotency_key=?",
                [key],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((id, prior)) = prior {
            if prior != fingerprint {
                return Err(Error::Invalid(
                    "Idempotency key reused with different input".into(),
                ));
            }
            return Ok((id, false));
        }
        let pending: i64 = conn.query_row(
            "SELECT count(*) FROM submissions WHERE status IN ('queued','running')",
            [],
            |r| r.get(0),
        )?;
        if pending >= 1 {
            return Err(Error::Busy("An evaluation is already running".into()));
        }
        conn.execute(
            "INSERT INTO submissions VALUES(?,?,?,?,?,?,'queued',NULL,?)",
            params![id, problem, code, mode, key, fingerprint, stamp()],
        )?;
        Ok((id.into(), true))
    }
    pub fn running(&self, id: &str) -> Result<()> {
        self.conn()?.execute(
            "UPDATE submissions SET status='running' WHERE id=? AND status='queued'",
            [id],
        )?;
        Ok(())
    }
    pub fn finish(&self, id: &str, result: &EvaluationResult) -> Result<()> {
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        let (problem, mode, status): (String, String, String) = tx.query_row(
            "SELECT problem_id,mode,status FROM submissions WHERE id=?",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        if status == "completed" {
            return Ok(());
        }
        tx.execute(
            "UPDATE submissions SET status='completed',result=? WHERE id=?",
            params![serde_json::to_string(result)?, id],
        )?;
        let solved = mode == "submit" && result.outcome == "passed";
        // Infrastructure failure is recorded but does not change learning progress.
        if result.outcome != "crashed" {
            tx.execute("INSERT INTO progress VALUES(?,?,1,?,?) ON CONFLICT(problem_id) DO UPDATE SET status=CASE WHEN progress.status='solved' OR excluded.status='solved' THEN 'solved' ELSE 'attempted' END,attempts=progress.attempts+1,best_time_ms=CASE WHEN excluded.best_time_ms IS NULL THEN progress.best_time_ms WHEN progress.best_time_ms IS NULL THEN excluded.best_time_ms ELSE min(progress.best_time_ms,excluded.best_time_ms) END,solved_at=coalesce(progress.solved_at,excluded.solved_at)",params![problem,if solved{"solved"}else{"attempted"},if solved{Some(result.total_time_ms)}else{None},if solved{Some(stamp())}else{None}])?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn result(&self, id: &str) -> Result<Value> {
        let conn = self.read_conn()?;
        let row: Option<(String, Option<String>)> = conn
            .query_row(
                "SELECT status,result FROM submissions WHERE id=?",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (status, result) = row.ok_or_else(|| Error::NotFound("Submission not found".into()))?;
        let mut value = if let Some(result) = result {
            serde_json::from_str::<Value>(&result)?
        } else {
            json!({"passed":0,"total":0,"results":[],"totalTimeMs":0})
        };
        value["submissionId"] = json!(id);
        value["status"] = json!(status);
        value["allPassed"] = json!(value["outcome"] == "passed");
        Ok(value)
    }
    pub fn progress(&self) -> Result<Value> {
        let conn = self.read_conn()?;
        let mut stmt =
            conn.prepare("SELECT problem_id,status,attempts,best_time_ms,solved_at FROM progress")?;
        let rows=stmt.query_map([],|r|Ok((r.get::<_,String>(0)?,json!({"status":r.get::<_,String>(1)?,"attempts":r.get::<_,i64>(2)?,"bestTimeMs":r.get::<_,Option<f64>>(3)?,"solvedAt":r.get::<_,Option<String>>(4)?}))))?;
        let mut map = serde_json::Map::new();
        for row in rows {
            let (key, value) = row?;
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    }
    pub fn history(&self, problem: &str) -> Result<Value> {
        let conn = self.read_conn()?;
        let mut stmt=conn.prepare("SELECT id,code,result,created_at FROM submissions WHERE problem_id=? ORDER BY created_at DESC,rowid DESC LIMIT 100")?;
        let rows = stmt.query_map([problem], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut values = vec![];
        for row in rows {
            let (id, code, result, created) = row?;
            let result = result
                .and_then(|v| serde_json::from_str::<Value>(&v).ok())
                .unwrap_or_default();
            values.push(json!({"id":id,"code":code,"passed":result["outcome"]=="passed","execTimeMs":result["totalTimeMs"],"submittedAt":created}));
        }
        Ok(json!(values))
    }
    pub fn draft(&self, id: &str) -> Result<Option<String>> {
        Ok(self
            .read_conn()?
            .query_row("SELECT code FROM drafts WHERE problem_id=?", [id], |r| {
                r.get(0)
            })
            .optional()?)
    }
    pub fn save_draft(&self, id: &str, code: &str) -> Result<()> {
        if code.len() > 64000 {
            return Err(Error::Invalid("Draft exceeds 64KB".into()));
        }
        self.conn()?.execute("INSERT INTO drafts VALUES(?,?) ON CONFLICT(problem_id) DO UPDATE SET code=excluded.code",params![id,code])?;
        Ok(())
    }
    pub fn legacy_users(path: &Path) -> Result<Value> {
        let source = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut stmt = source.prepare(
            "SELECT DISTINCT user_id FROM progress UNION SELECT DISTINCT user_id FROM submission",
        )?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(json!(rows))
    }
    pub fn import_legacy(&self, path: &Path, user: &str) -> Result<Value> {
        let source = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let source_name = path.canonicalize()?.to_string_lossy().to_string();
        let mut conn = self.conn()?;
        let tx = conn.transaction()?;
        if tx.query_row(
            "SELECT count(*) FROM imports WHERE source=? AND user_id=?",
            params![source_name, user],
            |r| r.get::<_, i64>(0),
        )? != 0
        {
            return Err(Error::Invalid(
                "This legacy user was already imported".into(),
            ));
        }
        let mut count = 0;
        let mut stmt=source.prepare("SELECT problem_id,status,attempts,best_time_ms,solved_at FROM progress WHERE user_id=?")?;
        let mut rows = stmt.query([user])?;
        while let Some(row) = rows.next()? {
            tx.execute("INSERT INTO progress VALUES(?,?,?,?,?) ON CONFLICT(problem_id) DO UPDATE SET status=CASE WHEN progress.status='solved' OR excluded.status='solved' THEN 'solved' ELSE 'attempted' END,attempts=progress.attempts+excluded.attempts,best_time_ms=CASE WHEN excluded.best_time_ms IS NULL THEN progress.best_time_ms WHEN progress.best_time_ms IS NULL THEN excluded.best_time_ms ELSE min(progress.best_time_ms,excluded.best_time_ms) END,solved_at=coalesce(progress.solved_at,excluded.solved_at)",params![row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,i64>(2)?,row.get::<_,Option<f64>>(3)?,row.get::<_,Option<String>>(4)?])?;
            count += 1;
        }
        let mut stmt=source.prepare("SELECT s.id,s.problem_id,s.source_code,s.mode,s.created_at,r.outcome,r.passed_count,r.total_count,r.total_ms,r.details_json,r.error,r.error_code FROM submission s JOIN judge_job j ON j.submission_id=s.id JOIN judge_result r ON r.job_id=j.id WHERE s.user_id=? AND s.status='completed'")?;
        let mut rows = stmt.query([user])?;
        let mut submissions = 0;
        while let Some(row) = rows.next()? {
            let id = format!("legacy-{}", row.get::<_, String>(0)?);
            let result = json!({"outcome":row.get::<_,String>(5)?,"passed":row.get::<_,i64>(6)?,"total":row.get::<_,i64>(7)?,"totalTimeMs":row.get::<_,f64>(8)?,"results":serde_json::from_str::<Value>(&row.get::<_,String>(9)?)?,"error":row.get::<_,Option<String>>(10)?,"errorCode":row.get::<_,Option<String>>(11)?});
            submissions += tx.execute(
                "INSERT OR IGNORE INTO submissions VALUES(?,?,?,?,?,?,'completed',?,?)",
                params![
                    id,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    id,
                    id,
                    result.to_string(),
                    row.get::<_, String>(4)?
                ],
            )?;
        }
        if count == 0 && submissions == 0 {
            return Err(Error::Invalid("Legacy user has no records".into()));
        }
        tx.execute(
            "INSERT INTO imports VALUES(?,?)",
            params![source_name, user],
        )?;
        tx.commit()?;
        Ok(json!({"progress":count,"submissions":submissions}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{mpsc, Arc},
        time::Duration,
    };

    #[test]
    fn wal_reader_is_independent_of_writer_and_only_sees_commits() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Storage::open(&dir.path().join("test.db")).unwrap());
        let mut writer = db.conn().unwrap();
        let tx = writer.transaction().unwrap();
        tx.execute(
            "INSERT INTO progress VALUES('relu','attempted',1,NULL,NULL)",
            [],
        )
        .unwrap();
        let (send, receive) = mpsc::channel();
        let reader = db.clone();
        let handle = std::thread::spawn(move || send.send(reader.progress()).unwrap());
        let snapshot = receive.recv_timeout(Duration::from_secs(2));
        tx.commit().unwrap();
        drop(writer);
        handle.join().unwrap();
        assert_eq!(
            snapshot
                .expect("reader must not wait for writer lock")
                .unwrap(),
            json!({})
        );
        assert_eq!(db.progress().unwrap()["relu"]["attempts"], 1);
        drop(db);
        assert!(!dir.path().join("test.db-wal").exists());
    }
}

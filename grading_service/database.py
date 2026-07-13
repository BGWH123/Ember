"""SQLite persistence and migrations.  Schema work happens only at startup."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent.parent / "data" / "ember.db"))
LOCAL_TENANT_ID = "local"
LOCAL_USER_PREFIX = "local-user-"


def now() -> str:
    return datetime.now(UTC).isoformat()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def initialize() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        # The previous application used `progress` for a non-tenant schema.
        # Preserve it before creating the contract table; this is idempotent.
        old_progress = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='progress'").fetchone()
        if old_progress and "tenant_id" not in (old_progress["sql"] or ""):
            conn.execute("ALTER TABLE progress RENAME TO legacy_progress")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migration (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tenant (id TEXT PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, plan TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS user (id TEXT PRIMARY KEY, external_subject TEXT, display_name TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS membership (tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(tenant_id,user_id), FOREIGN KEY(tenant_id) REFERENCES tenant(id), FOREIGN KEY(user_id) REFERENCES user(id));
        CREATE TABLE IF NOT EXISTS session (id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS quota_policy (tenant_id TEXT PRIMARY KEY, max_running_jobs INTEGER NOT NULL, max_queued_jobs INTEGER NOT NULL, max_user_running_jobs INTEGER NOT NULL, rate_limit_per_minute INTEGER NOT NULL, max_source_bytes INTEGER NOT NULL, max_cpu_ms INTEGER NOT NULL, max_memory_bytes INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS problem_version (problem_id TEXT NOT NULL, version TEXT NOT NULL, content_hash TEXT NOT NULL, runtime_key TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(problem_id,version));
        CREATE TABLE IF NOT EXISTS submission (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, problem_id TEXT NOT NULL, problem_version TEXT NOT NULL, mode TEXT NOT NULL, source_code TEXT NOT NULL, idempotency_key TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, accepted_at TEXT, completed_at TEXT, UNIQUE(tenant_id,user_id,idempotency_key));
        CREATE TABLE IF NOT EXISTS judge_job (id TEXT PRIMARY KEY, submission_id TEXT UNIQUE NOT NULL, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, priority INTEGER NOT NULL, queue_name TEXT NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL, available_at TEXT NOT NULL, lease_until TEXT, worker_id TEXT, queued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, last_error_code TEXT);
        CREATE TABLE IF NOT EXISTS judge_result (id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL, tenant_id TEXT NOT NULL, outcome TEXT NOT NULL, passed_count INTEGER NOT NULL, total_count INTEGER NOT NULL, error_code TEXT, error TEXT, queue_wait_ms INTEGER NOT NULL, worker_startup_ms INTEGER NOT NULL, setup_ms INTEGER NOT NULL, user_code_ms INTEGER NOT NULL, test_execution_ms INTEGER NOT NULL, persist_ms INTEGER NOT NULL, total_ms INTEGER NOT NULL, details_json TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, problem_id TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL, best_time_ms INTEGER, solved_at TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(tenant_id,user_id,problem_id));
        CREATE INDEX IF NOT EXISTS idx_submission_tenant_user_created ON submission(tenant_id,user_id,created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_job_ready ON judge_job(status,available_at,priority);
        CREATE INDEX IF NOT EXISTS idx_job_tenant_status ON judge_job(tenant_id,status,queued_at);
        CREATE INDEX IF NOT EXISTS idx_progress_tenant_user ON progress(tenant_id,user_id,updated_at DESC);
        """)
        # Additive compatibility migration for databases created by early v0 builds.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(judge_job)")}
        if "test_indices_json" not in columns:
            conn.execute("ALTER TABLE judge_job ADD COLUMN test_indices_json TEXT")
        stamp = now()
        conn.execute("INSERT OR IGNORE INTO tenant VALUES (?,?,?,?,?,?,?)", (LOCAL_TENANT_ID, "local", "Local workspace", "active", "local", stamp, stamp))
        conn.execute("INSERT OR IGNORE INTO quota_policy VALUES (?,?,?,?,?,?,?,?,?)", (LOCAL_TENANT_ID, 1, 32, 1, 30, 64_000, 10_000, 1_024 * 1_024 * 1_024, stamp))
        _migrate_legacy(conn)
    finally:
        conn.close()


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    """One-time, repeatable migration from the original local SQLite tables."""
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "users" not in tables:
        return
    stamp = now()
    legacy_users = conn.execute("SELECT id,session_token,created_at FROM users").fetchall()
    for legacy in legacy_users:
        user_id = f"legacy-user-{legacy['id']}"
        created = legacy["created_at"] or stamp
        conn.execute("INSERT OR IGNORE INTO user VALUES (?,?,?,?,?,?)", (user_id, None, "Anonymous", "active", created, stamp))
        conn.execute("INSERT OR IGNORE INTO membership VALUES (?,?,?,?,?,?)", (LOCAL_TENANT_ID, user_id, "member", "active", created, stamp))
        conn.execute("INSERT OR IGNORE INTO session VALUES (?,?,?,?,?,?,?)", (f"legacy-session-{legacy['id']}", token_hash(legacy["session_token"]), LOCAL_TENANT_ID, user_id, "2099-01-01T00:00:00+00:00", created, stamp))
    if "legacy_progress" in tables:
        for row in conn.execute("SELECT user_id,task_id,status,best_time_ms,attempts,solved_at FROM legacy_progress"):
            conn.execute("INSERT OR IGNORE INTO progress VALUES (?,?,?,?,?,?,?,?)", (LOCAL_TENANT_ID, f"legacy-user-{row['user_id']}", row["task_id"], row["status"], row["attempts"] or 0, row["best_time_ms"], row["solved_at"], stamp))
    if "submissions" in tables:
        for row in conn.execute("SELECT id,user_id,task_id,code,passed,exec_time_ms,submitted_at FROM submissions"):
            submission_id = f"legacy-submission-{row['id']}"
            created = row["submitted_at"] or stamp
            status = "completed"
            conn.execute("INSERT OR IGNORE INTO problem_version VALUES (?,?,?,?,?,?,?)", (row["task_id"], "v0", "catalog", "python-torch", "published", "{}", created))
            conn.execute("INSERT OR IGNORE INTO submission VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (submission_id, LOCAL_TENANT_ID, f"legacy-user-{row['user_id']}", row["task_id"], "v0", "submit", row["code"][:64_000], f"legacy-{row['id']}", status, created, created, created))


def resolve_local_session(session_token: str) -> tuple[str, str]:
    """Create a compatible anonymous tenant/user/session once, at the boundary."""
    digest = token_hash(session_token)
    with transaction() as conn:
        row = conn.execute("SELECT tenant_id,user_id FROM session WHERE token_hash=?", (digest,)).fetchone()
        if row:
            conn.execute("UPDATE session SET last_seen_at=? WHERE token_hash=?", (now(), digest))
            return row["tenant_id"], row["user_id"]
        user_id = LOCAL_USER_PREFIX + str(uuid.uuid4())
        stamp = now()
        conn.execute("INSERT INTO user VALUES (?,?,?,?,?,?)", (user_id, None, "Anonymous", "active", stamp, stamp))
        conn.execute("INSERT INTO membership VALUES (?,?,?,?,?,?)", (LOCAL_TENANT_ID, user_id, "member", "active", stamp, stamp))
        conn.execute("INSERT INTO session VALUES (?,?,?,?,?,?,?)", (str(uuid.uuid4()), digest, LOCAL_TENANT_ID, user_id, "2099-01-01T00:00:00+00:00", stamp, stamp))
        return LOCAL_TENANT_ID, user_id

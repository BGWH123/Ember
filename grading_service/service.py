"""Application service: admission, fair local queue, persistence and result mapping."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from typing import Any

from .database import now, transaction
from .domain import JudgeResult, JobStatus, Outcome, ResourceBudget, SubmissionStatus
from .judge import execute, prewarm


class AdmissionError(Exception):
    def __init__(self, code: str, message: str, status: int = 429):
        self.code, self.message, self.status = code, message, status


class SubmissionService:
    def __init__(self) -> None:
        self._ready: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._work, name="ember-judge-worker", daemon=True)

    def start(self) -> None:
        prewarm()
        self._worker.start()

    def stop(self) -> None:
        self._stop.set(); self._ready.put("")
        self._worker.join(timeout=2)

    def create(self, tenant_id: str, user_id: str, problem_id: str, code: str, mode: str, idempotency_key: str, test_indices: list[int] | None = None) -> dict[str, Any]:
        if mode not in {"sample", "submit"}:
            raise AdmissionError("invalid_mode", "mode must be sample or submit", 422)
        with transaction() as conn:
            existing = conn.execute("SELECT id,status FROM submission WHERE tenant_id=? AND user_id=? AND idempotency_key=?", (tenant_id, user_id, idempotency_key)).fetchone()
            if existing:
                return {"submission_id": existing["id"], "status": existing["status"], "idempotent": True}
            membership = conn.execute("SELECT 1 FROM membership WHERE tenant_id=? AND user_id=? AND status='active'", (tenant_id,user_id)).fetchone()
            quota = conn.execute("SELECT * FROM quota_policy WHERE tenant_id=?", (tenant_id,)).fetchone()
            if not membership or not quota:
                raise AdmissionError("forbidden", "An active tenant membership is required", 403)
            if len(code.encode()) > quota["max_source_bytes"]:
                raise AdmissionError("source_too_large", "Source code exceeds the tenant limit", 413)
            queued = conn.execute("SELECT count(*) FROM judge_job WHERE tenant_id=? AND status IN ('queued','leased','running')", (tenant_id,)).fetchone()[0]
            if queued >= quota["max_queued_jobs"]:
                raise AdmissionError("tenant_queue_full", "Tenant queue is full; retry later")
            running = conn.execute("SELECT count(*) FROM judge_job WHERE tenant_id=? AND user_id=? AND status IN ('leased','running')", (tenant_id,user_id)).fetchone()[0]
            if running >= quota["max_user_running_jobs"]:
                raise AdmissionError("user_concurrency_limited", "User already has a running evaluation")
            submission_id, job_id, stamp = str(uuid.uuid4()), str(uuid.uuid4()), now()
            # Task versions are content-addressable in future builds; v0 compatibility uses current catalog version.
            conn.execute("INSERT OR IGNORE INTO problem_version VALUES (?,?,?,?,?,?,?)", (problem_id,"v0","catalog", "python-torch", "published", "{}", stamp))
            conn.execute("INSERT INTO submission VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (submission_id,tenant_id,user_id,problem_id,"v0",mode,code,idempotency_key,SubmissionStatus.QUEUED,stamp,stamp,None))
            conn.execute("INSERT INTO judge_job (id,submission_id,tenant_id,user_id,priority,queue_name,status,attempt,available_at,queued_at,test_indices_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (job_id,submission_id,tenant_id,user_id,0,"local",JobStatus.QUEUED,0,stamp,stamp,json.dumps(test_indices) if test_indices is not None else None))
        self._ready.put(job_id)
        return {"submission_id": submission_id, "status": "queued", "idempotent": False, "test_indices": test_indices}

    def _work(self) -> None:
        while not self._stop.is_set():
            job_id = self._ready.get()
            if not job_id:
                continue
            self._run(job_id)

    def _run(self, job_id: str) -> None:
        with transaction() as conn:
            row = conn.execute("SELECT j.*,s.problem_id,s.source_code,s.mode FROM judge_job j JOIN submission s ON s.id=j.submission_id WHERE j.id=? AND j.status='queued'", (job_id,)).fetchone()
            if not row: return
            conn.execute("UPDATE judge_job SET status='running',attempt=attempt+1,started_at=?,worker_id=? WHERE id=? AND status='queued'", (now(), threading.current_thread().name, job_id))
            conn.execute("UPDATE submission SET status='running' WHERE id=?", (row["submission_id"],))
            quota = conn.execute("SELECT * FROM quota_policy WHERE tenant_id=?", (row["tenant_id"],)).fetchone()
        test_indices = json.loads(row["test_indices_json"]) if row["test_indices_json"] else None
        result = execute(row["problem_id"], row["source_code"], test_indices, ResourceBudget(max_source_bytes=quota["max_source_bytes"], max_cpu_ms=quota["max_cpu_ms"], max_memory_bytes=quota["max_memory_bytes"]))
        result.queue_wait_ms = max(0, int((time.time() - __import__('datetime').datetime.fromisoformat(row["queued_at"]).timestamp()) * 1000))
        self._persist(row, result)

    def _persist(self, row: Any, result: JudgeResult) -> None:
        started = time.perf_counter(); stamp = now()
        with transaction() as conn:
            job_status = JobStatus.SUCCEEDED if result.outcome == Outcome.PASSED else JobStatus.TIMEOUT if result.outcome == Outcome.TIMEOUT else JobStatus.FAILED
            conn.execute("INSERT INTO judge_result VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(uuid.uuid4()),row["id"],row["tenant_id"],result.outcome,result.passed_count,result.total_count,result.error_code,result.error,result.queue_wait_ms,result.worker_startup_ms,result.setup_ms,result.user_code_ms,result.test_execution_ms,0,result.total_ms,json.dumps(result.details)[:16_000],stamp))
            conn.execute("UPDATE judge_job SET status=?,finished_at=?,last_error_code=? WHERE id=? AND status='running'", (job_status,stamp,result.error_code,row["id"]))
            conn.execute("UPDATE submission SET status='completed',completed_at=? WHERE id=?", (stamp,row["submission_id"]))
            prior = conn.execute("SELECT status,best_time_ms,attempts FROM progress WHERE tenant_id=? AND user_id=? AND problem_id=?", (row["tenant_id"],row["user_id"],row["problem_id"])).fetchone()
            solved = result.outcome == Outcome.PASSED and row["mode"] == "submit"
            status = "solved" if solved else (prior["status"] if prior and prior["status"] == "solved" else "attempted")
            best = min(prior["best_time_ms"], result.total_ms) if prior and prior["best_time_ms"] is not None and solved else (result.total_ms if solved else (prior["best_time_ms"] if prior else None))
            conn.execute("INSERT INTO progress VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(tenant_id,user_id,problem_id) DO UPDATE SET status=excluded.status,attempts=progress.attempts+1,best_time_ms=excluded.best_time_ms,solved_at=COALESCE(progress.solved_at,excluded.solved_at),updated_at=excluded.updated_at", (row["tenant_id"],row["user_id"],row["problem_id"],status,(prior["attempts"] + 1 if prior else 1),best,(stamp if solved else None),stamp))
        result.persist_ms = int((time.perf_counter()-started)*1000)

    def get(self, tenant_id: str, submission_id: str) -> dict[str, Any] | None:
        with transaction() as conn:
            row = conn.execute("SELECT s.*,r.* FROM submission s LEFT JOIN judge_job j ON j.submission_id=s.id LEFT JOIN judge_result r ON r.job_id=j.id WHERE s.id=? AND s.tenant_id=?", (submission_id,tenant_id)).fetchone()
        if not row: return None
        details = json.loads(row["details_json"] or "[]")
        return {"submissionId": row["id"], "status": row["status"], "passed": row["passed_count"] or 0, "total": row["total_count"] or 0, "allPassed": row["outcome"] == "passed", "results": details, "totalTimeMs": row["total_ms"] or 0, "error": row["error"], "timings": {k: row[k] or 0 for k in ("queue_wait_ms","worker_startup_ms","setup_ms","user_code_ms","test_execution_ms","persist_ms","total_ms")}}

    def progress(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        with transaction() as conn: rows = conn.execute("SELECT * FROM progress WHERE tenant_id=? AND user_id=?", (tenant_id,user_id)).fetchall()
        return {r["problem_id"]: {"status":r["status"],"bestTimeMs":r["best_time_ms"],"attempts":r["attempts"],"solvedAt":r["solved_at"]} for r in rows}

    def history(self, tenant_id: str, user_id: str, problem_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        with transaction() as conn: rows=conn.execute("SELECT s.id,s.source_code,s.created_at,r.outcome,r.total_ms FROM submission s LEFT JOIN judge_job j ON j.submission_id=s.id LEFT JOIN judge_result r ON r.job_id=j.id WHERE s.tenant_id=? AND s.user_id=? AND s.problem_id=? ORDER BY s.created_at DESC LIMIT ? OFFSET ?", (tenant_id,user_id,problem_id,min(limit,100),offset)).fetchall()
        return [{"id":r["id"],"passed":r["outcome"]=="passed","execTimeMs":r["total_ms"],"submittedAt":r["created_at"],"code":r["source_code"]} for r in rows]

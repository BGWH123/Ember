"""Admission and a durable SQLite queue for a single local judge scheduler."""
from __future__ import annotations

import fcntl
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from . import database
from .database import now, transaction
from .domain import JudgeResult, JobStatus, Outcome, ResourceBudget, SubmissionStatus
from .judge import execute, warm_runtime
from .timing import Timings, event, measure, recording
from torch_judge.tasks import get_task

logger = logging.getLogger(__name__)


class AdmissionError(Exception):
    def __init__(self, code: str, message: str, status: int = 429):
        self.code, self.message, self.status = code, message, status


class SubmissionService:
    def __init__(self) -> None:
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock_file = None
        self._last_error: str | None = None

    @property
    def healthy(self) -> bool:
        return bool(self._worker and self._worker.is_alive() and not self._stop.is_set() and not self._last_error)

    def start(self) -> None:
        # Recovery is valid only when no other scheduler owns this database.
        lock_path = Path(database.DB_PATH).resolve().with_suffix(".judge.lock")
        self._lock_file = lock_path.open("a")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            warm_runtime()
            with transaction() as conn:
                recovered = conn.execute("UPDATE judge_job SET status='queued',worker_id=NULL,started_at=NULL WHERE status IN ('leased','running') AND NOT EXISTS (SELECT 1 FROM judge_result r WHERE r.job_id=judge_job.id)").rowcount
                conn.execute("UPDATE submission SET status='queued' WHERE id IN (SELECT submission_id FROM judge_job WHERE status='queued')")
            self._stop.clear()
            self._last_error = None
            self._worker = threading.Thread(target=self._work, name="ember-judge-worker", daemon=True)
            self._worker.start()
            event("scheduler.started", recovered_jobs=recovered)
        except BaseException:
            self._lock_file.close()
            self._lock_file = None
            raise

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker:
            # Give the current bounded evaluation time to finish and persist.
            self._worker.join(timeout=15)
            if self._worker.is_alive():
                logger.error("Judge scheduler did not stop within 15 seconds")
                return  # retain ownership; a second scheduler must not recover its job
        if self._lock_file:
            self._lock_file.close()
            self._lock_file = None

    def create_for_session(self, session_token: str, problem_id: str, code: str, mode: str,
                           idempotency_key: str, test_indices: list[int] | None = None) -> dict[str, Any]:
        with measure("admission_ms", cpu=True), transaction() as conn:
            with measure("session_ms"):
                tenant_id, user_id = database.resolve_local_session(session_token, connection=conn)
            result = self.create(tenant_id, user_id, problem_id, code, mode, idempotency_key, test_indices, connection=conn)
        # Notify only after COMMIT so the worker cannot miss an uncommitted job.
        self._wake.set()
        return result

    def create(self, tenant_id: str, user_id: str, problem_id: str, code: str, mode: str,
               idempotency_key: str, test_indices: list[int] | None = None, *,
               connection: sqlite3.Connection | None = None) -> dict[str, Any]:
        task = get_task(problem_id)
        if task is None:
            raise AdmissionError("problem_unavailable", "Problem not found", 404)
        if mode not in {"sample", "submit"}:
            raise AdmissionError("invalid_mode", "mode must be sample or submit", 422)
        if not idempotency_key or len(idempotency_key) > 128:
            raise AdmissionError("invalid_idempotency_key", "Use an idempotency key of 1 to 128 characters", 422)
        if mode == "submit" and test_indices is not None:
            raise AdmissionError("invalid_test_indices", "Full submissions must run every test", 422)
        if mode == "sample" and test_indices is None:
            test_indices = [0]
        if test_indices is not None and (not test_indices or len(set(test_indices)) != len(test_indices)
                or any(type(i) is not int or i < 0 or i >= len(task.get("tests", [])) for i in test_indices)):
            raise AdmissionError("invalid_test_indices", "Select at least one distinct valid test", 422)
        admission_scope = measure("admission_ms", cpu=True) if connection is None else nullcontext()
        database_scope = transaction() if connection is None else nullcontext(connection)
        with admission_scope, database_scope as conn:
            existing = conn.execute("SELECT id,status,problem_id,source_code,mode FROM submission WHERE tenant_id=? AND user_id=? AND idempotency_key=?", (tenant_id, user_id, idempotency_key)).fetchone()
            if existing:
                prior_indices = conn.execute("SELECT test_indices_json FROM judge_job WHERE submission_id=?", (existing["id"],)).fetchone()
                if (existing["problem_id"], existing["source_code"], existing["mode"]) != (problem_id, code, mode) or (prior_indices and json.loads(prior_indices[0] or "null") != test_indices):
                    raise AdmissionError("idempotency_conflict", "This key was already used for a different submission", 409)
                return {"submission_id": existing["id"], "status": existing["status"], "idempotent": True}
            membership = conn.execute("SELECT 1 FROM membership WHERE tenant_id=? AND user_id=? AND status='active'", (tenant_id, user_id)).fetchone()
            quota = conn.execute("SELECT * FROM quota_policy WHERE tenant_id=?", (tenant_id,)).fetchone()
            if not membership or not quota:
                raise AdmissionError("forbidden", "An active tenant membership is required", 403)
            if len(code.encode()) > quota["max_source_bytes"]:
                raise AdmissionError("source_too_large", "Source code exceeds the tenant limit", 413)
            queued = conn.execute("SELECT count(*) FROM judge_job WHERE tenant_id=? AND status IN ('queued','leased','running')", (tenant_id,)).fetchone()[0]
            if queued >= quota["max_queued_jobs"]:
                raise AdmissionError("tenant_queue_full", "Tenant queue is full; retry later")
            running = conn.execute("SELECT count(*) FROM judge_job WHERE tenant_id=? AND user_id=? AND status IN ('queued','leased','running')", (tenant_id, user_id)).fetchone()[0]
            if running >= quota["max_user_running_jobs"]:
                raise AdmissionError("user_concurrency_limited", "User already has an outstanding evaluation")
            submission_id, job_id, stamp = str(uuid.uuid4()), str(uuid.uuid4()), now()
            conn.execute("INSERT OR IGNORE INTO problem_version VALUES (?,?,?,?,?,?,?)", (problem_id, "v0", "catalog", "python-torch", "published", "{}", stamp))
            conn.execute("INSERT INTO submission (id,tenant_id,user_id,problem_id,problem_version,mode,source_code,idempotency_key,status,created_at,accepted_at,completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (submission_id, tenant_id, user_id, problem_id, "v0", mode, code, idempotency_key, SubmissionStatus.QUEUED, stamp, stamp, None))
            conn.execute("INSERT INTO judge_job (id,submission_id,tenant_id,user_id,priority,queue_name,status,attempt,available_at,queued_at,test_indices_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (job_id, submission_id, tenant_id, user_id, 0, "local", JobStatus.QUEUED, 0, stamp, stamp, json.dumps(test_indices) if test_indices is not None else None))
        if connection is None:
            self._wake.set()
        return {"submission_id": submission_id, "status": "queued", "idempotent": False, "test_indices": test_indices}

    def _work(self) -> None:
        failed_job: str | None = None
        while not self._stop.is_set():
            self._wake.clear()
            job_id = failed_job
            try:
                if failed_job:
                    self._fail_job(failed_job)
                    failed_job = None
                with transaction(write=False) as conn:
                    row = conn.execute("SELECT id FROM judge_job WHERE status='queued' AND available_at<=? ORDER BY priority DESC,queued_at,id LIMIT 1", (now(),)).fetchone()
                self._last_error = None
                if row is None:
                    self._wake.wait(timeout=1)
                    continue
                job_id = row["id"]
                self._run(job_id)
            except Exception:
                # Keep the scheduler alive and durably close the failed job. A DB
                # outage leaves it pending here until persistence works again.
                logger.exception("Judge scheduler failed for job %s", job_id)
                self._last_error = "scheduler_error"
                failed_job = job_id
                self._stop.wait(timeout=0.2)

    def _fail_job(self, job_id: str) -> None:
        with transaction(write=False) as conn:
            row = conn.execute("SELECT j.*,s.problem_id,s.mode FROM judge_job j JOIN submission s ON s.id=j.submission_id WHERE j.id=? AND j.status IN ('queued','running')", (job_id,)).fetchone()
        if row:
            self._persist(row, JudgeResult(Outcome.CRASHED, 0, 0, error_code="scheduler_error", error="The evaluation could not be completed"))

    def _run(self, job_id: str) -> None:
        claim = Timings()
        with recording(claim), measure("claim_ms"), transaction() as conn:
            row = conn.execute("SELECT j.*,s.problem_id,s.source_code,s.mode FROM judge_job j JOIN submission s ON s.id=j.submission_id WHERE j.id=? AND j.status='queued'", (job_id,)).fetchone()
            if not row:
                return
            started_at = now()
            conn.execute("UPDATE judge_job SET status='running',attempt=attempt+1,started_at=?,worker_id=? WHERE id=? AND status='queued'", (started_at, threading.current_thread().name, job_id))
            conn.execute("UPDATE submission SET status='running' WHERE id=?", (row["submission_id"],))
            quota = conn.execute("SELECT * FROM quota_policy WHERE tenant_id=?", (row["tenant_id"],)).fetchone()
        execution = time.perf_counter()
        try:
            indices = json.loads(row["test_indices_json"]) if row["test_indices_json"] else None
            result = execute(row["problem_id"], row["source_code"], indices, ResourceBudget(max_source_bytes=quota["max_source_bytes"], max_cpu_ms=quota["max_cpu_ms"], max_memory_bytes=quota["max_memory_bytes"]))
        except Exception:
            logger.exception("Judge execution failed for job %s", job_id)
            result = JudgeResult(Outcome.CRASHED, 0, 0, error_code="worker_crashed", error="The judge process could not complete", total_ms=round((time.perf_counter() - execution) * 1000, 3))
        result.queue_wait_ms = max(0, round((datetime.fromisoformat(started_at) - datetime.fromisoformat(row["queued_at"])).total_seconds() * 1000, 3))
        result.timings.update({("claim_" + key if key.startswith("db_") else key): value for key, value in claim.snapshot().items()})
        for attempt in range(3):
            try:
                self._persist(row, result)
                break
            except sqlite3.OperationalError:
                if attempt == 2:
                    raise
                self._stop.wait(timeout=0.05 * (attempt + 1))

    def _persist(self, row: Any, result: JudgeResult) -> None:
        persisted = Timings()
        stamp = now()
        with recording(persisted), transaction() as conn:
            job_status = JobStatus.SUCCEEDED if result.outcome == Outcome.PASSED else JobStatus.TIMEOUT if result.outcome == Outcome.TIMEOUT else JobStatus.FAILED
            # Truncate individual output/error fields in the worker; slicing the
            # serialized JSON would make a legitimate result impossible to read.
            conn.execute("INSERT INTO judge_result (id,job_id,tenant_id,outcome,passed_count,total_count,error_code,error,queue_wait_ms,worker_startup_ms,setup_ms,user_code_ms,test_execution_ms,persist_ms,total_ms,details_json,created_at,timings_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), row["id"], row["tenant_id"], result.outcome, result.passed_count, result.total_count, result.error_code, result.error, result.queue_wait_ms, result.worker_startup_ms, result.setup_ms, result.user_code_ms, result.test_execution_ms, 0, result.total_ms, json.dumps(result.details, ensure_ascii=False), stamp, self._timing_json(result)))
            conn.execute("UPDATE judge_job SET status=?,finished_at=?,last_error_code=? WHERE id=? AND status IN ('queued','running')", (job_status, stamp, result.error_code, row["id"]))
            conn.execute("UPDATE submission SET status='completed',completed_at=? WHERE id=?", (stamp, row["submission_id"]))
            prior = conn.execute("SELECT status,best_time_ms,attempts FROM progress WHERE tenant_id=? AND user_id=? AND problem_id=?", (row["tenant_id"], row["user_id"], row["problem_id"])).fetchone()
            solved = result.outcome == Outcome.PASSED and row["mode"] == "submit"
            status = "solved" if solved else (prior["status"] if prior and prior["status"] == "solved" else "attempted")
            best = min(prior["best_time_ms"], result.total_ms) if prior and prior["best_time_ms"] is not None and solved else (result.total_ms if solved else (prior["best_time_ms"] if prior else None))
            conn.execute("INSERT INTO progress VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(tenant_id,user_id,problem_id) DO UPDATE SET status=excluded.status,attempts=progress.attempts+1,best_time_ms=excluded.best_time_ms,solved_at=COALESCE(progress.solved_at,excluded.solved_at),updated_at=excluded.updated_at", (row["tenant_id"], row["user_id"], row["problem_id"], status, (prior["attempts"] + 1 if prior else 1), best, (stamp if solved else None), stamp))
        result.persist_ms = round((time.perf_counter() - persisted.started) * 1000, 3)
        result.timings.update({"persist_" + key: value for key, value in persisted.snapshot().items()})
        result.timings["persist_ms"] = result.persist_ms
        result.timings["server_e2e_ms"] = max(0, round((datetime.fromisoformat(now()) - datetime.fromisoformat(row["queued_at"])).total_seconds() * 1000, 3))
        # Commit can only be timed after it returns. The verdict above is atomic;
        # this small second write stores its completed timing, including COMMIT.
        try:
            with transaction() as conn:
                conn.execute("UPDATE judge_result SET persist_ms=?,timings_json=? WHERE job_id=?", (result.persist_ms, self._timing_json(result), row["id"]))
        except sqlite3.OperationalError:
            logger.exception("Could not save timing metadata for job %s", row["id"])
        event("judge.completed", job_id=row["id"], submission_id=row["submission_id"], task_id=row["problem_id"],
              mode=row["mode"], outcome=result.outcome, error_code=result.error_code,
              timings={**result.timings, "queue_wait_ms": result.queue_wait_ms, "total_ms": result.total_ms}, resources=result.resources)

    @staticmethod
    def _timing_json(result: JudgeResult) -> str:
        return json.dumps({"timings": result.timings, "resources": result.resources}, ensure_ascii=False)

    def get(self, tenant_id: str, submission_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with measure("result_read_ms", cpu=True), transaction(write=False) as conn:
            row = conn.execute("SELECT s.*,r.* FROM submission s LEFT JOIN judge_job j ON j.submission_id=s.id LEFT JOIN judge_result r ON r.job_id=j.id WHERE s.id=? AND s.tenant_id=? AND (? IS NULL OR s.user_id=?)", (submission_id, tenant_id, user_id, user_id)).fetchone()
        if not row:
            return None
        extra = json.loads(row["timings_json"] or "{}")
        timings = {key: row[key] for key in ("queue_wait_ms", "worker_startup_ms", "setup_ms", "user_code_ms", "test_execution_ms", "persist_ms", "total_ms") if row[key] is not None}
        if "persist_ms" not in extra.get("timings", {}):
            timings.pop("persist_ms", None)  # COMMIT timing has not been saved yet
        timings.update(extra.get("timings", {}))
        return {"submissionId": row["id"], "status": row["status"], "passed": row["passed_count"] or 0,
                "total": row["total_count"] or 0, "allPassed": row["outcome"] == "passed", "outcome": row["outcome"],
                "results": json.loads(row["details_json"] or "[]"), "totalTimeMs": row["total_ms"] or 0,
                "error": row["error"], "errorCode": row["error_code"], "timings": timings, "resources": extra.get("resources", {})}

    def progress(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        with measure("progress_read_ms", cpu=True), transaction(write=False) as conn:
            rows = conn.execute("SELECT * FROM progress WHERE tenant_id=? AND user_id=?", (tenant_id, user_id)).fetchall()
        return {r["problem_id"]: {"status": r["status"], "bestTimeMs": r["best_time_ms"], "attempts": r["attempts"], "solvedAt": r["solved_at"]} for r in rows}

    def history(self, tenant_id: str, user_id: str, problem_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        with measure("history_read_ms", cpu=True), transaction(write=False) as conn:
            rows = conn.execute("SELECT s.id,s.source_code,s.created_at,r.outcome,r.total_ms FROM submission s LEFT JOIN judge_job j ON j.submission_id=s.id LEFT JOIN judge_result r ON r.job_id=j.id WHERE s.tenant_id=? AND s.user_id=? AND s.problem_id=? ORDER BY s.created_at DESC LIMIT ? OFFSET ?", (tenant_id, user_id, problem_id, max(1, min(limit, 100)), max(0, offset))).fetchall()
        return [{"id": r["id"], "passed": r["outcome"] == "passed", "execTimeMs": r["total_ms"], "submittedAt": r["created_at"], "code": r["source_code"]} for r in rows]

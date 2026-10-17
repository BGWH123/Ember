"""Regression tests against real SQLite, disposable processes and HTTP boundaries."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grading_service import database
from grading_service.domain import JudgeResult, Outcome, ResourceBudget
from grading_service.judge import execute
from grading_service.service import AdmissionError, SubmissionService


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regression.db"))
    database.initialize()
    return database.resolve_local_session("owner")


def wait_result(service, tenant, submission_id):
    deadline = time.monotonic() + 5
    wait = threading.Event()
    while time.monotonic() < deadline:
        result = service.get(tenant, submission_id)
        if result["status"] == "completed":
            return result
        wait.wait(0.01)
    pytest.fail("Submission did not reach a terminal state within 5s")


@pytest.mark.parametrize("fixture", json.loads((Path(__file__).parents[1] / "scripts/loadtest/workload.json").read_text()), ids=lambda item: item["taskId"])
def test_real_correct_solutions_pass_default_memory_budget_with_browser_imports(fixture):
    code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n" + fixture["code"]
    result = execute(fixture["taskId"], code, None, ResourceBudget())
    assert result.outcome == Outcome.PASSED, (result.error, result.details)
    assert result.total_count > 0 and result.passed_count == result.total_count
    assert result.total_ms >= result.test_execution_ms > 0
    assert result.worker_startup_ms > 0 and result.setup_ms > 0
    assert 0 < result.resources["peak_rss_bytes"] < ResourceBudget().max_memory_bytes
    assert result.resources["torch_threads"] == 1


@pytest.mark.parametrize("code,indices,budget,error", [
    ("def relu(:", None, ResourceBudget(), "syntax_error"),
    ("import os\ndef relu(x): return x", None, ResourceBudget(), "forbidden_import"),
    ("def relu(x): return x", [], ResourceBudget(), "invalid_test_indices"),
    ("def relu(x): return x", [-1], ResourceBudget(), "invalid_test_indices"),
    ("while True: pass", None, ResourceBudget(max_cpu_ms=200), "timeout"),
    ("buf = bytearray(2 * 1024 * 1024 * 1024)", None, ResourceBudget(), "memory_limit"),
])
def test_real_process_failures_are_bounded_and_next_process_can_succeed(code, indices, budget, error):
    result = execute("relu", code, indices, budget)
    assert result.error_code == error, result
    if error not in {"timeout", "memory_limit"}:
        assert result.resources["exit_code"] == 0  # a reported error exits normally
    following = execute("relu", "def relu(x): return x.clamp(min=0)", [0], ResourceBudget())
    assert following.outcome == Outcome.PASSED


def test_user_system_exit_returns_a_terminal_result_not_an_invalid_payload():
    result = execute("relu", "from typing import sys\nsys.exit(0)", None, ResourceBudget())
    assert result.outcome == Outcome.CRASHED and result.error_code == "worker_crashed"
    assert result.resources["exit_code"] == 0


def test_large_unicode_output_is_bounded_drained_and_persisted_as_valid_json(store):
    tenant, user = store
    service = SubmissionService()
    code = "print('测' * 20000)\ndef relu(x): return x.clamp(min=0)"
    created = service.create(tenant, user, "relu", code, "submit", "large-output")
    with database.transaction(write=False) as conn:
        job_id = conn.execute("SELECT id FROM judge_job").fetchone()[0]
    service._run(job_id)
    result = service.get(tenant, created["submission_id"], user)
    assert result["allPassed"], result
    assert sum(len((detail.get("output") or "").encode()) for detail in result["results"]) <= 16_000
    with database.transaction(write=False) as conn:
        serialized = conn.execute("SELECT details_json FROM judge_result").fetchone()[0]
    assert len(serialized.encode()) > 16_000  # output plus valid JSON metadata
    assert json.loads(serialized) == result["results"]


def test_existing_session_progress_and_poll_read_while_writer_holds_wal(store):
    tenant, user = store
    service = SubmissionService()
    created = service.create(tenant, user, "relu", "def relu(x): return x", "submit", "reader")
    with database.transaction() as writer:
        writer.execute("UPDATE judge_job SET status='running'")
        assert database.resolve_local_session("owner") == (tenant, user)
        assert service.get(tenant, created["submission_id"], user)["status"] == "queued"
        assert service.progress(tenant, user) == {}
        assert len(service.history(tenant, user, "relu")) == 1


def test_keep_open_retains_wal_without_holding_a_reader_snapshot(store):
    with database.keep_open():
        for number in range(3):
            with database.transaction() as conn:
                conn.execute("UPDATE tenant SET name=? WHERE id='local'", (f"name-{number}",))
            assert Path(database.DB_PATH + "-wal").exists()
        with closing(database.connect()) as conn:
            assert conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        with database.transaction(write=False) as conn:
            assert conn.execute("SELECT name FROM tenant WHERE id='local'").fetchone()[0] == "name-2"


def test_concurrent_first_requests_share_one_anonymous_user(store):
    with ThreadPoolExecutor(max_workers=20) as pool:
        identities = list(pool.map(database.resolve_local_session, ["new-session"] * 20))
    assert len(set(identities)) == 1
    with database.transaction(write=False) as conn:
        assert conn.execute("SELECT count(*) FROM user").fetchone()[0] == 2


def test_admission_rejects_empty_or_partial_full_tests_and_preserves_idempotency(store):
    tenant, user = store
    service = SubmissionService()
    code = "def relu(x): return x.clamp(min=0)"
    for mode, indices in (("sample", []), ("sample", [-1]), ("sample", [0, 0]), ("submit", [0])):
        with pytest.raises(AdmissionError) as failure:
            service.create(tenant, user, "relu", code, mode, "invalid", indices)
        assert failure.value.status == 422
    first = service.create(tenant, user, "relu", code, "sample", "stable-key")
    second = service.create(tenant, user, "relu", code, "sample", "stable-key")
    assert first["submission_id"] == second["submission_id"] and second["idempotent"]
    with pytest.raises(AdmissionError) as conflict:
        service.create(tenant, user, "relu", code + "\n", "sample", "stable-key")
    assert conflict.value.status == 409
    with pytest.raises(AdmissionError) as busy:
        service.create(tenant, user, "relu", code, "sample", "second-job")
    assert busy.value.code == "user_concurrency_limited"


def test_rejected_first_submission_rolls_back_session_and_user(store):
    service = SubmissionService()
    with pytest.raises(AdmissionError):
        service.create_for_session("rejected-new-session", "relu", "def relu(x): return x", "sample", "bad", [])
    with database.transaction(write=False) as conn:
        assert conn.execute("SELECT count(*) FROM user").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM submission").fetchone()[0] == 0


def test_restart_recovers_running_job_and_prevents_two_schedulers(store, monkeypatch):
    tenant, user = store
    service = SubmissionService()
    created = service.create(tenant, user, "relu", "def relu(x): return x", "submit", "restart")
    with database.transaction() as conn:
        conn.execute("UPDATE judge_job SET status='running',attempt=1")
        conn.execute("UPDATE submission SET status='running'")
    monkeypatch.setattr("grading_service.service.warm_runtime", lambda: None)
    monkeypatch.setattr("grading_service.service.execute", lambda *_args: JudgeResult(Outcome.PASSED, 1, 1))
    service.start()
    try:
        duplicate = SubmissionService()
        with pytest.raises(BlockingIOError):
            duplicate.start()
        result = wait_result(service, tenant, created["submission_id"])
        assert result["allPassed"] and service.healthy
        with database.transaction(write=False) as conn:
            assert conn.execute("SELECT attempt FROM judge_job").fetchone()[0] == 2
    finally:
        service.stop()


def test_unexpected_execute_error_does_not_kill_scheduler(store, monkeypatch):
    tenant, user = store
    service = SubmissionService()
    calls = 0

    def worker(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("test unexpected failure")
        return JudgeResult(Outcome.PASSED, 1, 1)

    monkeypatch.setattr("grading_service.service.warm_runtime", lambda: None)
    monkeypatch.setattr("grading_service.service.execute", worker)
    service.start()
    try:
        first = service.create(tenant, user, "relu", "def relu(x): return x", "submit", "crash")
        assert wait_result(service, tenant, first["submission_id"])["errorCode"] == "worker_crashed"
        second = service.create(tenant, user, "relu", "def relu(x): return x", "submit", "following")
        assert wait_result(service, tenant, second["submission_id"])["allPassed"]
        assert service.healthy
    finally:
        service.stop()


def test_http_admission_is_async_timings_propagate_and_other_user_cannot_read(store, monkeypatch):
    from grading_service import main
    service = SubmissionService()
    release = threading.Event()
    entered = threading.Event()

    def worker(*_args):
        entered.set()
        if not release.wait(timeout=3):
            raise RuntimeError("test worker was not released")
        return JudgeResult(Outcome.PASSED, 1, 1)

    monkeypatch.setattr(main, "service", service)
    monkeypatch.setattr("grading_service.service.warm_runtime", lambda: None)
    monkeypatch.setattr("grading_service.service.execute", worker)
    try:
        with TestClient(main.app) as client:
            response = client.post("/submissions", json={"taskId": "relu", "code": "def relu(x): return x", "idempotencyKey": "http"},
                                   headers={"X-Session-Token": "owner", "X-Request-ID": "regression-request"})
            assert response.status_code == 202 and entered.wait(timeout=1)
            assert response.headers["x-request-id"] == "regression-request"
            for point in ("backend;dur=", "dispatch;dur=", "session;dur=", "db_lock;dur=", "admission;dur="):
                assert point in response.headers["server-timing"]
            status = f"/submissions/{response.json()['submission_id']}"
            assert client.get(status, headers={"X-Session-Token": "different-user"}).status_code == 404
            assert client.get("/health").status_code == 200
            release.set()
            result = wait_result(service, store[0], response.json()["submission_id"])
            assert result["allPassed"]
    finally:
        release.set()

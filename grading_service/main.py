"""HTTP adapter for Ember's queued, tenant-scoped grading service."""
from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, StrictInt

from .database import SessionExpired, initialize, keep_open, resolve_local_session
from .service import AdmissionError, SubmissionService
from .timing import TimingMiddleware, configure_logging, handler_started, measure
from torch_judge.tasks import get_task

service = SubmissionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    initialize()
    with keep_open():
        service.start()
        try:
            yield
        finally:
            service.stop()


app = FastAPI(title="Ember Grading Service", lifespan=lifespan)
app.add_middleware(TimingMiddleware)


class SubmissionRequest(BaseModel):
    taskId: str
    code: str = Field(max_length=64_000)
    mode: str = "submit"
    idempotencyKey: str | None = Field(default=None, min_length=1, max_length=128)
    testIndices: list[StrictInt] | None = None


class UserRequest(BaseModel):
    sessionToken: str


def context(session_token: str | None) -> tuple[str, str]:
    handler_started()
    _require_session_token(session_token)
    try:
        with measure("session_ms"):
            return resolve_local_session(session_token)
    except SessionExpired as exc:
        raise HTTPException(401, "Session expired") from exc


def _require_session_token(session_token: str | None) -> None:
    if not session_token:
        raise HTTPException(401, "A session token is required")
    if len(session_token) > 256:
        raise HTTPException(422, "Session token is too long")


def create(request: SubmissionRequest, session_token: str | None) -> dict:
    handler_started()
    if get_task(request.taskId) is None:
        raise HTTPException(404, f"Task '{request.taskId}' not found")
    _require_session_token(session_token)
    try:
        return service.create_for_session(session_token, request.taskId, request.code, request.mode, request.idempotencyKey or str(uuid.uuid4()), request.testIndices)
    except AdmissionError as exc:
        raise HTTPException(exc.status, {"code": exc.code, "message": exc.message}) from exc
    except SessionExpired as exc:
        raise HTTPException(401, "Session expired") from exc


@app.post("/submissions", status_code=202)
def create_submission(request: SubmissionRequest, x_session_token: str | None = Header(default=None)) -> dict:
    return create(request, x_session_token)


@app.get("/submissions/{submission_id}")
def get_submission(submission_id: str, x_session_token: str | None = Header(default=None)) -> dict:
    tenant_id, user_id = context(x_session_token)
    result = service.get(tenant_id, submission_id, user_id)
    if result is None: raise HTTPException(404, "Submission not found")
    return result


# Compatibility endpoints retain existing browser clients while using the same queue.
@app.post("/grade", status_code=202)
def grade(request: SubmissionRequest, x_session_token: str | None = Header(default=None)) -> dict:
    request.mode = "submit"; return create(request, x_session_token)


@app.post("/run", status_code=202)
def run(request: SubmissionRequest, x_session_token: str | None = Header(default=None)) -> dict:
    request.mode = "sample"; return create(request, x_session_token)


@app.post("/users")
def user(request: UserRequest) -> dict:
    _, user_id = context(request.sessionToken)
    return {"userId": user_id}


@app.get("/progress")
def progress(x_session_token: str | None = Header(default=None)) -> dict:
    tenant_id, user_id = context(x_session_token)
    return service.progress(tenant_id, user_id)


@app.get("/progress/{_user_id}")
def legacy_progress(_user_id: str, x_session_token: str | None = Header(default=None)) -> dict:
    tenant_id, user_id = context(x_session_token)
    return service.progress(tenant_id, user_id)


@app.get("/submissions/history/{task_id}")
def history(task_id: str, limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0), x_session_token: str | None = Header(default=None)) -> list[dict]:
    tenant_id, user_id = context(x_session_token)
    return service.history(tenant_id, user_id, task_id, limit, offset)


@app.get("/tasks/{task_id}/notebook")
def get_notebook(task_id: str) -> dict:
    task = get_task(task_id)
    if task is None or not task.get("solution"): raise HTTPException(404, "Notebook not found")
    return {"cells": [{"type":"code","source":task["solution"].strip(),"role":"solution"}]}


@app.get("/tasks/{task_id}/solution")
def get_solution(task_id: str) -> dict[str, str]:
    task = get_task(task_id)
    if task is None or not task.get("solution"): raise HTTPException(404, "Solution not found")
    return {"solution": task["solution"]}


@app.get("/health")
def health() -> dict[str, str]:
    handler_started()
    if not service.healthy:
        raise HTTPException(503, "Judge scheduler is unavailable")
    return {"status": "ok"}

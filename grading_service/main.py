"""HTTP adapter for Ember's queued, tenant-scoped grading service."""
from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .database import initialize, resolve_local_session
from .service import AdmissionError, SubmissionService
from torch_judge.tasks import get_task

service = SubmissionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    service.start()
    yield
    service.stop()


app = FastAPI(title="Ember Grading Service", lifespan=lifespan)


class SubmissionRequest(BaseModel):
    taskId: str
    code: str = Field(max_length=64_000)
    mode: str = "submit"
    idempotencyKey: str | None = None
    testIndices: list[int] | None = None


class UserRequest(BaseModel):
    sessionToken: str


def context(session_token: str | None) -> tuple[str, str]:
    if not session_token:
        raise HTTPException(401, "A session token is required")
    return resolve_local_session(session_token)


def create(request: SubmissionRequest, session_token: str | None) -> dict:
    if get_task(request.taskId) is None:
        raise HTTPException(404, f"Task '{request.taskId}' not found")
    tenant_id, user_id = context(session_token)
    try:
        return service.create(tenant_id, user_id, request.taskId, request.code, request.mode, request.idempotencyKey or str(uuid.uuid4()), request.testIndices)
    except AdmissionError as exc:
        raise HTTPException(exc.status, {"code": exc.code, "message": exc.message}) from exc


@app.post("/submissions", status_code=202)
def create_submission(request: SubmissionRequest, x_session_token: str | None = Header(default=None)) -> dict:
    return create(request, x_session_token)


@app.get("/submissions/{submission_id}")
def get_submission(submission_id: str, x_session_token: str | None = Header(default=None)) -> dict:
    tenant_id, _ = context(x_session_token)
    result = service.get(tenant_id, submission_id)
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
    _, user_id = resolve_local_session(request.sessionToken)
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
def history(task_id: str, limit: int = 20, offset: int = 0, x_session_token: str | None = Header(default=None)) -> list[dict]:
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
def health() -> dict[str, str]: return {"status": "ok"}

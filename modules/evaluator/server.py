"""Loopback reference adapter for protocol interoperability; not a public multi-tenant service."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
import secrets
import threading
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from ember_evaluator.protocol import evaluate, hello, MAX_MESSAGE

app = FastAPI(title="Ember evaluator protocol v1")
_lock = threading.Lock()


def authorize(authorization: str | None) -> None:
    token = os.environ.get("EMBER_REMOTE_TOKEN")
    if not token or not secrets.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "Configure EMBER_REMOTE_TOKEN and supply Bearer authentication")


@app.get("/v1/hello")
def handshake(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return hello()


@app.post("/v1/evaluate")
async def run(request: Request, authorization: str | None = Header(default=None)):
    authorize(authorization)
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_MESSAGE:
            raise HTTPException(413, "Request too large")
    import json
    try:
        body = json.loads(data)
        request_id = body["requestId"]
        if not isinstance(request_id, str) or len(request_id) > 128:
            raise ValueError("Invalid request ID")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, "Invalid protocol request") from exc
    if not _lock.acquire(blocking=False):
        raise HTTPException(429, "Evaluator busy")
    try:
        result = await run_in_threadpool(evaluate, body)
        return {"requestId": request_id, "result": result}
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        _lock.release()

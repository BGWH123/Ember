"""Bounded protocol adapter. Only this package owns protocol stdout."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from .domain import ResourceBudget
from .judge import execute
from torch_judge.tasks import get_task

VERSION = "0.1.0"
PROTOCOL_VERSION = 1
MAX_MESSAGE = 2 * 1024 * 1024


def hello() -> dict:
    catalog = os.environ.get("EMBER_CATALOG_PATH")
    if catalog:
        version = hashlib.sha256(Path(catalog).read_bytes()).hexdigest()
    else:
        bundled = Path(__file__).resolve().parent.parent / "catalog-version.txt"
        if bundled.exists():
            version = bundled.read_text().strip()
        else:
            root = Path(__file__).resolve().parents[3]
            version = hashlib.sha256((root / "content/catalog/problems.json").read_bytes()).hexdigest()
    return {"protocolVersion": PROTOCOL_VERSION, "catalogVersion": version, "engineVersion": VERSION}


def evaluate(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Expected an evaluation request")
    task_id, code = request.get("taskId"), request.get("code")
    if not isinstance(task_id, str) or get_task(task_id) is None:
        raise ValueError("Unknown task")
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 64_000:
        raise ValueError("Expected 1–64000 bytes of source")
    indices = request.get("testIndices")
    count = len(get_task(task_id)["tests"])
    if indices is not None and (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 or index >= count for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Select distinct valid tests")
    budget = request.get("resourceBudget", {})
    limits = {"maxCpuMs": (100, 10_000), "maxMemoryBytes": (256 * 1024**2, 1024**3), "maxOutputBytes": (1, 16_000)}
    for key, (minimum, maximum) in limits.items():
        value = budget.get(key, maximum)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {key}")
    result = execute(task_id, code, indices, ResourceBudget(
        max_cpu_ms=budget.get("maxCpuMs", 10_000),
        max_memory_bytes=budget.get("maxMemoryBytes", 1024**3),
        max_output_bytes=budget.get("maxOutputBytes", 16_000)))
    return {"outcome": result.outcome, "passed": result.passed_count, "total": result.total_count,
            "results": result.details, "totalTimeMs": result.total_ms, "error": result.error,
            "errorCode": result.error_code, "timings": result.timings, "resources": result.resources}


def serve() -> None:
    while True:
        line = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
        if not line:
            return
        if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
            raise ValueError("Protocol message exceeds limit")
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            if not isinstance(request_id, str) or len(request_id) > 128:
                raise ValueError("Invalid request ID")
            if request["method"] == "hello":
                result = hello()
            elif request["method"] == "evaluate":
                result = evaluate(request["params"])
            else:
                raise ValueError("Unknown method")
            response = {"id": request_id, "result": result}
        except Exception as exc:  # Protocol process boundary: report, keep accepting independent requests.
            response = {"id": request_id, "error": {"code": "evaluation_error", "message": str(exc)[:1000]}}
        encoded = json.dumps(response, ensure_ascii=False)
        if len(encoded.encode()) > MAX_MESSAGE:
            encoded = json.dumps({"id": request_id, "error": {"code": "response_limit", "message": "Result too large"}})
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()

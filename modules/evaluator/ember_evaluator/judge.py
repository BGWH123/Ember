"""Supervise disposable judge processes without importing PyTorch into the API."""
from __future__ import annotations

import json
import multiprocessing as mp
import signal
import time
import psutil
from typing import Any

from .domain import JudgeResult, Outcome, ResourceBudget
from torch_judge.tasks import TASKS

_COMPILED_TESTS: dict[str, list[tuple[str, object]]] = {}
_METHOD = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
_CONTEXT = mp.get_context(_METHOD)
if _METHOD == "forkserver":
    # The helper, rather than a multithreaded HTTP server, owns the Torch runtime.
    _CONTEXT.set_forkserver_preload(["ember_evaluator.worker_runtime"])


def prewarm() -> None:
    """Validate every catalog test; the forkserver also caches these code objects."""
    if not _COMPILED_TESTS:
        for task_id, task in TASKS.items():
            _COMPILED_TESTS[task_id] = [
                (test["name"], compile(test["code"].replace("{fn}", task["function_name"]),
                                       f"<test:{test['name']}>", "exec"))
                for test in task.get("tests", [])
            ]


def warm_runtime() -> None:
    """Finish cold imports before admitting traffic, and verify the memory budget."""
    prewarm()
    result = execute("relu", "def relu(x): return torch.clamp(x, min=0)", [0], ResourceBudget())
    if result.outcome != Outcome.PASSED:
        raise RuntimeError(f"Judge warmup failed: {result.error or result.details}")


def _child(code: str, task_id: str, test_indices: list[int] | None, budget: ResourceBudget, conn: Any) -> None:
    entered = time.perf_counter()
    try:
        conn.send_bytes(json.dumps({"event": "started", "entered": entered}).encode())
        from .worker_runtime import run
        run(code, task_id, test_indices, budget, conn, entered)
    except BaseException as exc:  # one disposable process is the task boundary
        conn.send_bytes(json.dumps({"error_code": "worker_crashed", "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}).encode())
    finally:
        conn.close()


def _rss_bytes(pid: int) -> int:
    try:
        return psutil.Process(pid).memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def execute(task_id: str, code: str, test_indices: list[int] | None, budget: ResourceBudget, *, context=None) -> JudgeResult:
    """Drain IPC before joining; bound wall time, CPU time, resident memory and output.

    Linux RSS is sampled every 10ms. An address-space growth guard and the service
    container limit complement it; this is not a hostile-code security sandbox.
    """
    ctx = context or _CONTEXT
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(code, task_id, test_indices, budget, child), daemon=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    resources: dict[str, Any] = {"start_method": ctx.get_start_method()}
    payload: dict[str, Any] = {}
    peak_rss = 0
    launched = False
    received_result = False
    try:
        launch = time.perf_counter()
        process.start()
        launched = True
        timings["process_launch_ms"] = (time.perf_counter() - launch) * 1000
        child.close()
        deadline = started + budget.max_cpu_ms / 1000
        while True:
            peak_rss = max(peak_rss, _rss_bytes(process.pid))
            if peak_rss > budget.max_memory_bytes:
                payload = {"error_code": "memory_limit", "error": "Evaluation exceeded the resident memory limit"}
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                payload = {"error_code": "timeout", "error": "Evaluation exceeded the wall time limit"}
                break
            if parent.poll(min(0.01, remaining)):
                try:
                    raw = parent.recv_bytes(maxlength=256_000)
                except EOFError:
                    break
                received = decoding = time.perf_counter()
                message = json.loads(raw)
                timings["result_decode_ms"] = timings.get("result_decode_ms", 0) + (time.perf_counter() - decoding) * 1000
                if message.get("event") == "started":
                    timings["worker_startup_ms"] = (message["entered"] - started) * 1000
                elif message.get("event") == "ready":
                    resources.update(message["resources"])
                else:
                    payload = message
                    received_result = True
                    timings["result_transfer_ms"] = max(0, (received - message.get("sent_at", received)) * 1000)
                    break
            elif not process.is_alive():
                if not parent.poll():
                    break
        reaping = time.perf_counter()
        process.join(timeout=min(0.2, max(0, deadline - time.perf_counter())) if received_result else 0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        timings["process_reap_ms"] = (time.perf_counter() - reaping) * 1000
        resources["exit_code"] = process.exitcode
        if not payload:
            code_ = "timeout" if process.exitcode == -getattr(signal, "SIGXCPU", 24) else "worker_crashed"
            payload = {"error_code": code_, "error": f"Judge child exited without a result (exit code {process.exitcode})"}
    finally:
        child.close()
        parent.close()
        if launched and process.is_alive():
            process.kill()
            process.join(timeout=1)
        process.close()

    timings.update(payload.get("timings", {}))
    timings["worker_total_ms"] = (time.perf_counter() - started) * 1000
    resources["peak_rss_bytes"] = max(peak_rss, payload.get("peak_rss_bytes", 0))
    resources["cpu_ms"] = payload.get("cpu_ms")
    resources["output_truncated"] = payload.get("output_truncated", False)
    error_code = payload.get("error_code")
    if error_code:
        outcome = Outcome.TIMEOUT if error_code == "timeout" else Outcome.CRASHED if error_code == "worker_crashed" else Outcome.FAILED
    else:
        outcome = Outcome(payload["outcome"])
    return JudgeResult(
        outcome, payload.get("passed_count", 0), payload.get("total_count", 0), payload.get("details", []),
        error_code=error_code, error=f"{error_code}: {payload.get('error', '')}" if error_code else None,
        worker_startup_ms=round(timings.get("worker_startup_ms", 0), 3),
        setup_ms=round(timings.get("setup_ms", 0), 3), user_code_ms=round(timings.get("user_code_ms", 0), 3),
        test_execution_ms=round(timings.get("test_execution_ms", 0), 3),
        total_ms=round(timings["worker_total_ms"], 3),
        timings={key: round(value, 3) for key, value in timings.items()}, resources=resources,
    )

"""Prewarmed judge runtime. User code always runs in a fresh child process."""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
import multiprocessing as mp
import os
import resource
import time
from typing import Any

import numpy as np
import torch

from .domain import JudgeResult, Outcome, ResourceBudget
from torch_judge.tasks import TASKS, get_task

_COMPILED_TESTS: dict[str, list[tuple[str, object]]] = {}
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "getattr": getattr, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "max": max, "min": min,
    "range": range, "reversed": reversed, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "RuntimeError": RuntimeError,
    "__build_class__": builtins.__build_class__, "object": object, "property": property,
    "super": super,
}


def prewarm() -> None:
    for task_id, task in TASKS.items():
        _COMPILED_TESTS[task_id] = [(test["name"], compile(test["code"].replace("{fn}", task["function_name"]), f"<test:{test['name']}>", "exec")) for test in task.get("tests", [])]


def _child(code: str, task_id: str, test_indices: list[int] | None, output_limit: int, memory_limit: int, conn: Any) -> None:
    start = time.perf_counter()
    try:
        # The process is disposable; RLIMIT prevents accidental allocation growth on Linux.
        if os.name == "posix":
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        task = get_task(task_id)
        if task is None:
            conn.send({"error_code": "problem_unavailable", "error": "Task not found"}); return
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            conn.send({"error_code": "syntax_error", "error": str(exc)}); return
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            conn.send({"error_code": "forbidden_import", "error": "Imports are not allowed; torch, np, nn, F and math are preloaded"}); return
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "__name__": "submission", "torch": torch, "Tensor": torch.Tensor, "nn": torch.nn, "F": torch.nn.functional, "np": np, "math": math}
        user_start = time.perf_counter()
        exec(compile(tree, "<submission>", "exec"), namespace)  # noqa: S102
        user_code_ms = int((time.perf_counter() - user_start) * 1000)
        fn_name = task["function_name"]
        if fn_name not in namespace:
            conn.send({"error_code": "missing_function", "error": f"Function '{fn_name}' not found", "user_code_ms": user_code_ms}); return
        compiled = _COMPILED_TESTS[task_id]
        selected = [(i, item) for i, item in enumerate(compiled) if test_indices is None or i in test_indices]
        details, passed, tests_ms = [], 0, 0
        for _, (name, test) in selected:
            stream = io.StringIO()
            tick = time.perf_counter()
            try:
                with contextlib.redirect_stdout(stream):
                    exec(test, {"torch": torch, "Tensor": torch.Tensor, "nn": torch.nn, "F": torch.nn.functional, "np": np, "math": math, fn_name: namespace[fn_name]})  # noqa: S102
                elapsed = int((time.perf_counter() - tick) * 1000); passed += 1
                details.append({"name": name, "passed": True, "execTimeMs": elapsed, "output": stream.getvalue()[:output_limit] or None})
            except AssertionError as exc:
                elapsed = int((time.perf_counter() - tick) * 1000); details.append({"name": name, "passed": False, "execTimeMs": elapsed, "error": str(exc), "output": stream.getvalue()[:output_limit] or None})
            except Exception as exc:  # user code error
                elapsed = int((time.perf_counter() - tick) * 1000); details.append({"name": name, "passed": False, "execTimeMs": elapsed, "error": f"{type(exc).__name__}: {exc}", "output": stream.getvalue()[:output_limit] or None})
            tests_ms += elapsed
        conn.send({"outcome": "passed" if passed == len(selected) else "failed", "passed_count": passed, "total_count": len(selected), "details": details, "user_code_ms": user_code_ms, "test_execution_ms": tests_ms, "total_ms": int((time.perf_counter()-start)*1000)})
    except BaseException as exc:
        conn.send({"error_code": "worker_crashed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


def execute(task_id: str, code: str, test_indices: list[int] | None, budget: ResourceBudget) -> JudgeResult:
    parent, child = mp.Pipe(duplex=False)
    process = mp.Process(target=_child, args=(code, task_id, test_indices, budget.max_output_bytes, budget.max_memory_bytes, child), daemon=True)
    started = time.perf_counter(); process.start(); child.close()
    process.join(budget.max_cpu_ms / 1000)
    if process.is_alive():
        process.kill(); process.join()
        return JudgeResult(Outcome.TIMEOUT, 0, 0, error_code="timeout", error="Evaluation exceeded the time limit", total_ms=int((time.perf_counter()-started)*1000))
    payload = parent.recv() if parent.poll() else {"error_code": "worker_crashed", "error": "Judge child exited without a result"}
    if payload.get("error_code"):
        return JudgeResult(Outcome.FAILED if payload["error_code"] != "worker_crashed" else Outcome.CRASHED, 0, 0, error_code=payload["error_code"], error=payload["error_code"] + ": " + payload.get("error", ""), user_code_ms=payload.get("user_code_ms", 0), total_ms=int((time.perf_counter()-started)*1000))
    return JudgeResult(Outcome(payload["outcome"]), payload["passed_count"], payload["total_count"], payload["details"], user_code_ms=payload["user_code_ms"], test_execution_ms=payload["test_execution_ms"], total_ms=payload["total_ms"])

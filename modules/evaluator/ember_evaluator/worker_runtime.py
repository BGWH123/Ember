"""Preload only in the clean forkserver; never run user code in this helper."""
from __future__ import annotations

import ast
import builtins
import contextlib
import json
import math
import os
try:
    import resource
except ImportError:  # Windows is supervised using psutil in the parent.
    resource = None
import psutil
import sys
import time
import typing
from pathlib import Path

# Avoid importing threaded BLAS pools into the process that forks children.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import torch

from .domain import ResourceBudget
from .judge import _COMPILED_TESTS, prewarm
from torch_judge.tasks import get_task

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
_ = torch.tensor([0])
prewarm()

_MODULES = {"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": torch.nn.functional,
            "numpy": np, "math": math, "typing": typing}


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in _MODULES:
        raise ImportError("Only torch, torch.nn, torch.nn.functional, numpy, math and typing are available")
    return _MODULES[name] if fromlist else _MODULES[name.split(".")[0]]


_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "bytes", "bytearray", "complex", "dict", "enumerate", "float",
    "getattr", "hasattr", "int", "isinstance", "iter", "len", "list", "max", "min", "next", "pow",
    "print", "range", "reversed", "round", "set", "setattr", "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "TypeError", "RuntimeError", "NotImplementedError", "AssertionError",
    "__build_class__", "object", "property", "super", "staticmethod", "classmethod",
)}
_BUILTINS["__import__"] = _import


class BoundedOutput:
    """One byte budget for all stdout/stderr, including top-level user code."""
    def __init__(self, limit: int):
        self.remaining = limit
        self.parts: list[str] = []
        self.truncated = False

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8", errors="replace")
        kept = encoded[:self.remaining]
        if kept:
            self.parts.append(kept.decode("utf-8", errors="ignore"))
        self.remaining -= len(kept)
        self.truncated |= len(kept) < len(encoded)
        return len(value)

    def flush(self) -> None:
        pass

    def take(self) -> str | None:
        value = "".join(self.parts)
        self.parts.clear()
        return value or None


def _memory() -> tuple[int, int]:
    memory = psutil.Process().memory_info()
    return memory.vms, memory.rss


def run(code: str, task_id: str, test_indices: list[int] | None, budget: ResourceBudget, conn, entered: float) -> None:
    setup = time.perf_counter()
    timings = {"runtime_import_ms": (setup - entered) * 1000}
    payload: dict = {}
    output = BoundedOutput(budget.max_output_bytes)
    try:
        vms, rss = _memory()
        conn.send_bytes(json.dumps({"event": "ready", "resources": {
            "baseline_vms_bytes": vms, "baseline_rss_bytes": rss, "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__, "python_version": sys.version.split()[0],
        }}).encode())
        if rss > budget.max_memory_bytes:
            raise MemoryError("The runtime already exceeds the resident memory budget")
        # Mapped Torch libraries are not resident allocations. Allow only remaining
        # resident headroom as address-space growth; parent monitors total RSS.
        address_limit = vms + budget.max_memory_bytes - rss
        if resource is not None and sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        cpu_seconds = max(1, math.ceil(budget.max_cpu_ms / 1000))
        if resource is not None:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        task = get_task(task_id)
        if task is None:
            payload.update(error_code="problem_unavailable", error="Task not found")
            return
        parsing = time.perf_counter()
        tree = ast.parse(code)
        bytecode = compile(tree, "<submission>", "exec")
        timings["parse_compile_ms"] = (time.perf_counter() - parsing) * 1000
        compiled = _COMPILED_TESTS[task_id]
        indices = list(range(len(compiled))) if test_indices is None else test_indices
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(compiled) for i in indices):
            payload.update(error_code="invalid_test_indices", error="Select at least one distinct valid test")
            return
        namespace = {"__builtins__": _BUILTINS, "__name__": "submission", "torch": torch, "Tensor": torch.Tensor,
                     "nn": torch.nn, "F": torch.nn.functional, "np": np, "math": math}
        timings["setup_ms"] = (time.perf_counter() - setup) * 1000
        user_start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(bytecode, namespace)  # intentional judge execution in disposable process
        finally:
            timings["user_code_ms"] = (time.perf_counter() - user_start) * 1000
        fn_name = task["function_name"]
        if not callable(namespace.get(fn_name)):
            payload.update(error_code="missing_function", error=f"Callable '{fn_name}' not found")
            return
        initial_output = output.take()
        details, passed = [], 0
        tests_start = time.perf_counter()
        for position, index in enumerate(indices):
            name, test = compiled[index]
            tick = time.perf_counter()
            detail = {"name": name, "passed": False}
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(test, {"torch": torch, "Tensor": torch.Tensor, "nn": torch.nn, "F": torch.nn.functional,
                                "np": np, "math": math, fn_name: namespace[fn_name]})
                passed += 1
                detail["passed"] = True
            except Exception as exc:  # user verdict, not a scheduler failure
                detail["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            detail.update(execTimeMs=round((time.perf_counter() - tick) * 1000, 3), output=output.take())
            if position == 0 and initial_output:
                detail["output"] = initial_output + (detail["output"] or "")
            details.append(detail)
        timings["test_execution_ms"] = (time.perf_counter() - tests_start) * 1000
        payload.update(outcome="passed" if passed == len(indices) else "failed", passed_count=passed,
                       total_count=len(indices), details=details)
    except SyntaxError as exc:
        payload.update(error_code="syntax_error", error=str(exc)[:1000])
    except ImportError as exc:
        payload.update(error_code="forbidden_import", error=str(exc)[:1000])
    except MemoryError:
        payload.update(error_code="memory_limit", error="Evaluation exceeded the memory budget")
    except Exception as exc:
        payload.update(error_code="user_code_error", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    except BaseException as exc:  # SystemExit/interrupts are terminal for this disposable task
        payload.update(error_code="worker_crashed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    finally:
        process = psutil.Process()
        usage = process.cpu_times()
        peak = process.memory_info().rss
        if resource is not None:
            peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1))
        if peak > budget.max_memory_bytes:
            payload.update(error_code="memory_limit", error="Evaluation exceeded the resident memory limit")
        timings["child_total_ms"] = (time.perf_counter() - entered) * 1000
        payload.update(timings=timings, peak_rss_bytes=peak, cpu_ms=(usage.user + usage.system) * 1000,
                       output_truncated=output.truncated)
        encoding = time.perf_counter()
        json.dumps(payload, ensure_ascii=False)
        timings["result_encode_ms"] = (time.perf_counter() - encoding) * 1000
        payload["sent_at"] = time.perf_counter()
        conn.send_bytes(json.dumps(payload, ensure_ascii=False).encode())

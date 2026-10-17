"""Request and job timings, in milliseconds, without logging source or tokens."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


logger = logging.getLogger("ember.timings")


@dataclass
class Timings:
    started: float = field(default_factory=time.perf_counter)
    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, milliseconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + max(0.0, milliseconds)

    def snapshot(self) -> dict[str, float]:
        return {key: round(value, 3) for key, value in self.values.items()}


_current: ContextVar[Timings | None] = ContextVar("ember_timings", default=None)


@contextmanager
def recording(timings: Timings) -> Iterator[Timings]:
    token = _current.set(timings)
    try:
        yield timings
    finally:
        _current.reset(token)


@contextmanager
def measure(name: str, *, cpu: bool = False) -> Iterator[None]:
    current = _current.get()
    started = time.perf_counter()
    cpu_started = time.thread_time() if cpu and current is not None else None
    try:
        yield
    finally:
        if current is not None:
            current.add(name, (time.perf_counter() - started) * 1000)
            if cpu_started is not None:
                current.add(name.removesuffix("_ms") + "_cpu_ms", (time.thread_time() - cpu_started) * 1000)


def handler_started() -> None:
    """Include validation and thread-pool dispatch before the sync handler."""
    current = _current.get()
    if current is not None and "dispatch_ms" not in current.values:
        current.add("dispatch_ms", (time.perf_counter() - current.started) * 1000)


def event(name: str, **fields: object) -> None:
    logger.info(json.dumps({"event": name, **fields}, ensure_ascii=False, separators=(",", ":")))


def configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TimingMiddleware:
    """Use pure ASGI so context reaches FastAPI's worker threads and error paths."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        supplied = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied) else str(uuid.uuid4())
        timings = Timings()
        status = 500

        async def timed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                timings.add("backend_ms", (time.perf_counter() - timings.started) * 1000)
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-request-id", request_id.encode()),
                    (b"server-timing", ", ".join(
                        f"{name.removesuffix('_ms')};dur={value:.3f}"
                        for name, value in timings.snapshot().items()
                    ).encode()),
                ])
                message = {**message, "headers": headers}
            await send(message)

        with recording(timings):
            try:
                await self.app(scope, receive, timed_send)
            finally:
                timings.add("http_total_ms", (time.perf_counter() - timings.started) * 1000)
                route = scope.get("route")
                event("http.completed", request_id=request_id, method=scope["method"],
                      route=getattr(route, "path", "unmatched"), status=status, timings=timings.snapshot())

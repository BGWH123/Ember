"""Stable domain contracts shared by the API, scheduler and judge worker."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SubmissionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ResourceBudget:
    max_source_bytes: int = 64_000
    max_cpu_ms: int = 10_000
    max_memory_bytes: int = 1_024 * 1_024 * 1_024
    max_output_bytes: int = 16_000


@dataclass
class JudgeResult:
    outcome: Outcome
    passed_count: int
    total_count: int
    details: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error: str | None = None
    queue_wait_ms: int = 0
    worker_startup_ms: int = 0
    setup_ms: int = 0
    user_code_ms: int = 0
    test_execution_ms: int = 0
    persist_ms: int = 0
    total_ms: int = 0

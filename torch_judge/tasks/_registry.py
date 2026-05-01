"""Auto-discovery registry for task definitions."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

TASKS: dict[str, dict[str, Any]] = {}

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__package__}.{_info.name}")
    if hasattr(_mod, "TASK"):
        TASKS[_info.name] = _mod.TASK


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)


def list_tasks() -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        TASKS.items(),
        key=lambda t: DIFFICULTY_ORDER.get(t[1]["difficulty"], 9),
    )

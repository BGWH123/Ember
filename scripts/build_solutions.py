#!/usr/bin/env python3
"""Build solutions.json from torch_judge task definitions for the frontend."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "web" / "src" / "lib" / "solutions.json"

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS, list_tasks


def build_solution_entry(task: dict) -> dict:
    cells = [{"type": "code", "source": task["solution"].strip(), "role": "solution"}]
    if "explanation" in task:
        cells.append({"type": "markdown", "source": task["explanation"].strip(), "role": "explanation"})
    if "demo" in task:
        cells.append({"type": "code", "source": task["demo"].strip(), "role": "demo"})
    return {"cells": cells}


def main():
    result = {}
    for task_id, task in list_tasks():
        if "solution" not in task:
            print(f"WARNING: {task_id} has no solution field, skipping")
            continue
        result[task_id] = build_solution_entry(task)

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {len(result)} solutions to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

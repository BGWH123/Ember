#!/usr/bin/env python3
"""Build starters.json from task definitions while preserving legacy starters."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "content" / "catalog" / "starters.json"

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS


def _without_imports(source: str) -> str:
    """The editor prelude already provides Torch, nn, F, NumPy, and math."""
    return re.sub(r"^(?:(?:import |from )[^\n]+\n\s*)+", "", source.strip())


def main() -> None:
    starters = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    for task_id, task in TASKS.items():
        if "starter_code" in task:
            starters[task_id] = _without_imports(task["starter_code"])

    ordered = dict(sorted(starters.items()))
    OUTPUT.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Written {len(ordered)} starters to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

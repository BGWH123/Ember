#!/usr/bin/env python3
"""Build solutions.json from torch_judge task definitions for the frontend."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "content" / "catalog" / "solutions.json"

sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS, list_tasks

CTRL_MAP = {
    "\x07": r"\a",
    "\x08": r"\b",
    "\t": r"\t",
    "\x0b": r"\v",
    "\x0c": r"\f",
    "\r": r"\r",
}

KEY_ORDER = (
    "title",
    "title_zh",
    "difficulty",
    "category",
    "function_name",
    "starter_code",
    "hint",
    "hint_zh",
    "description_en",
    "description_zh",
    "theory_en",
    "theory_zh",
    "diagram_en",
    "diagram_zh",
    "tests",
    "solution",
    "demo",
    "explanation",
)


def _fix_ctrl(text: str) -> str:
    for ctrl, escaped in CTRL_MAP.items():
        text = text.replace(ctrl, escaped)
    return text


def _fmt_str(val: str) -> str:
    if "\n" not in val and len(val) < 100:
        return repr(val)
    if '"""' not in val and not val.endswith('"') and not val.endswith("\\"):
        prefix = "r" if "\\" in val else ""
        return f'{prefix}"""{val}"""'
    if "'''" not in val and not val.endswith("'") and not val.endswith("\\"):
        prefix = "r" if "\\" in val else ""
        return f"{prefix}'''{val}'''"
    return repr(val)


def normalize_task_files() -> None:
    tasks_dir = ROOT / "torch_judge" / "tasks"
    for path in sorted(tasks_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        task_id = path.stem
        task = dict(TASKS[task_id])
        for key in (
            "explanation",
            "theory_en",
            "theory_zh",
            "description_en",
            "description_zh",
            "hint",
            "hint_zh",
        ):
            if key in task and isinstance(task[key], str):
                task[key] = _fix_ctrl(task[key]).strip()
        for key in ("starter_code", "solution", "demo"):
            if key in task and isinstance(task[key], str):
                task[key] = task[key].strip()
        TASKS[task_id].update(task)

        lines = [f'"""{task["title"]} task."""', "", "TASK = {"]
        keys = [k for k in KEY_ORDER if k in task] + [
            k for k in task if k not in KEY_ORDER
        ]
        for key in keys:
            val = task[key]
            if key == "tests":
                lines.append('    "tests": [')
                for test in val:
                    tname = repr(test["name"])
                    tcode = _fmt_str(test["code"].strip())
                    lines.append("        {")
                    lines.append(f'            "name": {tname},')
                    lines.append(f'            "code": {tcode},')
                    lines.append("        },")
                lines.append("    ],")
            elif isinstance(val, str):
                lines.append(f"    {repr(key)}: {_fmt_str(val)},")
            else:
                lines.append(f"    {repr(key)}: {repr(val)},")
        lines.append("}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")


def build_solution_entry(task: dict) -> dict:
    cells = [{"type": "code", "source": task["solution"].strip(), "role": "solution"}]
    if "explanation" in task:
        cells.append({"type": "markdown", "source": _fix_ctrl(task["explanation"]).strip(), "role": "explanation"})
    if "demo" in task:
        cells.append({"type": "code", "source": task["demo"].strip(), "role": "demo"})
    return {"cells": cells}


def main():
    normalize_task_files()
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

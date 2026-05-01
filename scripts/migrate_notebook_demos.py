#!/usr/bin/env python3
"""One-time migration: extract demo cells from solution notebooks and inject into task definitions."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOLUTIONS_DIR = ROOT / "solutions"
TASKS_DIR = ROOT / "torch_judge" / "tasks"

SKIP_PATTERNS = ("google.colab", "torch_judge", "get_ipython", "colab.research.google.com")
SOLUTION_MARKER = re.compile(r"#\s*✅\s*SOLUTION")


def strip_comment_lines(src: str) -> str:
    lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
    return "\n".join(lines).strip()


def strip_imports(src: str) -> str:
    lines = [l for l in src.splitlines() if not l.startswith("import ") and not l.startswith("from ")]
    return "\n".join(lines).strip()


def extract_demo(nb_path: Path) -> str | None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    demo_parts = []
    for c in nb.get("cells", []):
        src = "".join(c["source"])
        if not src.strip() or any(s in src for s in SKIP_PATTERNS):
            continue
        if c["cell_type"] == "code":
            if not SOLUTION_MARKER.search(src):
                stripped = strip_imports(strip_comment_lines(src))
                if stripped:
                    demo_parts.append(stripped)
    return "\n\n".join(demo_parts) if demo_parts else None


def inject_demo(task_file: Path, demo: str) -> bool:
    content = task_file.read_text(encoding="utf-8")
    if '"demo"' in content or "'demo'" in content:
        return False

    # Find the closing of the "solution" field and insert demo after it
    # The TASK dict ends with }  — we insert before the final }
    # Strategy: find the last } that closes the TASK dict
    lines = content.split("\n")
    insert_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            insert_idx = i
            break

    if insert_idx is None:
        print(f"  WARNING: could not find closing brace in {task_file.name}")
        return False

    # Build the demo field
    demo_field = f'    "demo": """{demo}""",\n'
    lines.insert(insert_idx, demo_field)
    task_file.write_text("\n".join(lines), encoding="utf-8")
    return True


def main():
    demos = {}
    for nb_path in sorted(SOLUTIONS_DIR.glob("*_solution.ipynb")):
        task_id = re.sub(r"^\d+_", "", nb_path.stem).replace("_solution", "")
        demo = extract_demo(nb_path)
        if demo:
            demos[task_id] = demo

    print(f"Extracted demos from {len(demos)} notebooks")

    injected = 0
    for task_id, demo in demos.items():
        task_file = TASKS_DIR / f"{task_id}.py"
        if not task_file.exists():
            print(f"  SKIP: {task_id}.py not found")
            continue
        if inject_demo(task_file, demo):
            injected += 1
            print(f"  OK: {task_id}")
        else:
            print(f"  SKIP: {task_id} (already has demo)")

    print(f"\nInjected demo into {injected} task files")


if __name__ == "__main__":
    main()

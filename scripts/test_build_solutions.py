"""Tests for build_solutions.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from build_solutions import build_solution_entry


# --- unit tests for build_solution_entry ---


def test_solution_only():
    task = {"solution": "def foo(): pass"}
    entry = build_solution_entry(task)
    assert entry == {"cells": [{"type": "code", "source": "def foo(): pass", "role": "solution"}]}


def test_solution_with_demo():
    task = {"solution": "def foo(): pass", "demo": "print(foo())"}
    entry = build_solution_entry(task)
    assert len(entry["cells"]) == 2
    assert entry["cells"][0]["role"] == "solution"
    assert entry["cells"][1] == {"type": "code", "source": "print(foo())", "role": "demo"}


def test_solution_with_explanation_and_demo():
    task = {"solution": "def foo(): pass", "explanation": "Does foo.", "demo": "foo()"}
    entry = build_solution_entry(task)
    assert len(entry["cells"]) == 3
    assert entry["cells"][0]["role"] == "solution"
    assert entry["cells"][1] == {"type": "markdown", "source": "Does foo.", "role": "explanation"}
    assert entry["cells"][2]["role"] == "demo"


def test_strips_whitespace():
    task = {"solution": "  def foo(): pass  \n", "demo": "\n  foo()\n  "}
    entry = build_solution_entry(task)
    assert entry["cells"][0]["source"] == "def foo(): pass"
    assert entry["cells"][1]["source"] == "foo()"


# --- integration: verify solutions.json ---

SOLUTIONS_JSON = Path(__file__).parent.parent / "web" / "src" / "lib" / "solutions.json"


@pytest.mark.skipif(not SOLUTIONS_JSON.exists(), reason="solutions.json not generated yet")
def test_solutions_json_has_all_tasks():
    from torch_judge.tasks import TASKS

    data = json.loads(SOLUTIONS_JSON.read_text())
    assert len(data) == len(TASKS)


@pytest.mark.skipif(not SOLUTIONS_JSON.exists(), reason="solutions.json not generated yet")
def test_solutions_json_relu_shape():
    data = json.loads(SOLUTIONS_JSON.read_text())
    assert "relu" in data
    cells = data["relu"]["cells"]
    assert any(c["role"] == "solution" for c in cells)
    for c in cells:
        assert "type" in c and "source" in c and "role" in c
        assert c["role"] in ("solution", "demo", "explanation")


@pytest.mark.skipif(not SOLUTIONS_JSON.exists(), reason="solutions.json not generated yet")
def test_solutions_json_every_task_has_solution_cell():
    data = json.loads(SOLUTIONS_JSON.read_text())
    missing = [tid for tid, entry in data.items() if not any(c["role"] == "solution" for c in entry["cells"])]
    assert missing == [], f"Tasks missing solution cell: {missing}"

#!/usr/bin/env python3
"""Audit task and content catalogs for accidental duplicate identifiers."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torch_judge.tasks import TASKS  # noqa: E402


errors: list[str] = []


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalized_text(value: Any) -> str:
    return "".join(character for character in normalized(value) if character.isalnum())


def duplicate_values(items: Iterable[Any]) -> list[Any]:
    counts = Counter(items)
    return sorted(value for value, count in counts.items() if count > 1)


def load_json(relative_path: str) -> Any:
    path = ROOT / relative_path

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        duplicates = duplicate_values(key for key, _ in pairs)
        if duplicates:
            errors.append(f"{relative_path}: duplicate JSON keys {duplicates}")
        return dict(pairs)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def check_unique_records(
    label: str, records: list[dict[str, Any]], fields: Iterable[str]
) -> None:
    for field in fields:
        locations: dict[str, list[str]] = defaultdict(list)
        for index, record in enumerate(records):
            raw_value = record.get(field)
            value = (
                normalized(raw_value)
                if field in {"id", "functionName", "function_name"}
                else normalized_text(raw_value)
            )
            if value:
                locations[value].append(str(record.get("id", index)))
        duplicates = {
            value: ids for value, ids in locations.items() if len(ids) > 1
        }
        if duplicates:
            errors.append(f"{label}: duplicate {field} values {duplicates}")


def audit_tasks() -> None:
    task_records = [{"id": task_id, **task} for task_id, task in TASKS.items()]
    check_unique_records(
        "torch_judge/tasks",
        task_records,
        ("id", "function_name", "title", "title_zh"),
    )
    for task_id, task in TASKS.items():
        names = [normalized(test.get("name")) for test in task.get("tests", [])]
        duplicates = duplicate_values(name for name in names if name)
        if duplicates:
            errors.append(f"task {task_id}: duplicate test names {duplicates}")


def audit_catalogs() -> None:
    problem_files = (
        "content/catalog/problems.json",
        "apps/desktop/public/problems-list.json",
    )
    problem_sets: dict[str, set[str]] = {}
    for filename in problem_files:
        records = load_json(filename)["problems"]
        check_unique_records(
            filename, records, ("id", "title", "titleZh", "functionName")
        )
        problem_sets[filename] = {record["id"] for record in records}
        for problem in records:
            names = [
                normalized(test.get("name")) for test in problem.get("tests", [])
            ]
            duplicates = duplicate_values(name for name in names if name)
            if duplicates:
                errors.append(
                    f"{filename} problem {problem['id']}: "
                    f"duplicate test names {duplicates}"
                )

    keyed_files = (
        "content/catalog/starters.json",
        "content/catalog/solutions.json",
    )
    for filename in keyed_files:
        problem_sets[filename] = set(load_json(filename))

    expected = set(TASKS)
    for filename, problem_ids in problem_sets.items():
        if problem_ids != expected:
            errors.append(
                f"{filename}: ID set differs from tasks "
                f"(missing={sorted(expected - problem_ids)}, "
                f"extra={sorted(problem_ids - expected)})"
            )


def audit_paths() -> None:
    for path in load_json("content/catalog/paths.json")["paths"]:
        duplicates = duplicate_values(path.get("problems", []))
        if duplicates:
            errors.append(
                f"learning path {path['id']}: duplicate problem IDs {duplicates}"
            )


def audit_curriculum() -> None:
    placements: dict[str, list[str]] = defaultdict(list)
    curriculum = load_json("apps/desktop/public/curriculum.json")
    for stage in curriculum["stages"]:
        for module in stage["modules"]:
            location = f"{stage['id']}/{module['id']}"
            for field in ("problems", "hotCategories", "baguChapters"):
                duplicates = duplicate_values(module.get(field, []))
                if duplicates:
                    errors.append(
                        f"curriculum {location}: duplicate {field} {duplicates}"
                    )
            for problem_id in module.get("problems", []):
                placements[problem_id].append(location)
            if module.get("status") == "hands-on" and not module.get("problems"):
                errors.append(f"curriculum {location}: empty hands-on module")

    duplicates = {
        problem_id: locations
        for problem_id, locations in placements.items()
        if len(locations) > 1
    }
    if duplicates:
        errors.append(f"curriculum: problems in multiple modules {duplicates}")


def audit_questions() -> None:
    sources = (
        ("apps/desktop/public/hot-questions.json", "question"),
        ("apps/desktop/public/handbook/questions.json", "title"),
    )
    combined: list[dict[str, Any]] = []
    for filename, question_field in sources:
        questions = load_json(filename)["questions"]
        canonical = [
            {
                "id": question["id"],
                "question": question.get(question_field),
                "source": filename,
            }
            for question in questions
        ]
        check_unique_records(filename, canonical, ("id", "question"))
        combined.extend(canonical)
    check_unique_records("question files combined", combined, ("id", "question"))


def main() -> int:
    audit_tasks()
    audit_catalogs()
    audit_paths()
    audit_curriculum()
    audit_questions()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print(
        f"Deduplication audit passed: {len(TASKS)} tasks and all content catalogs "
        "contain unique identifiers and canonical curriculum placements."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

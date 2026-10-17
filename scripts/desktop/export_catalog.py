#!/usr/bin/env python3
"""Refresh the desktop catalog while retaining authored presentation fields."""
from pathlib import Path
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_problems import export_problem_catalog


def main():
    data = export_problem_catalog()
    for problem in data['problems']:
        for test in problem['tests']:
            ast.parse(test['code'].replace('{fn}', problem['functionName']))
    print(f"Exported {len(data['problems'])} desktop problems")


if __name__ == '__main__':
    main()

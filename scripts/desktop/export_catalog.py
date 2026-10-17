#!/usr/bin/env python3
"""Refresh the desktop execution catalog without changing the legacy baseline.
Preserve authored presentation fields; take executable tests/function names from task sources.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from torch_judge.tasks import TASKS


def main():
    destination=ROOT/'content/catalog';destination.mkdir(parents=True,exist_ok=True)
    source=destination/'problems.json'
    if not source.exists():source=ROOT/'web/src/lib/problems.json'
    data=json.loads(source.read_text())
    existing={p['id']:p for p in data['problems']}
    for task_id,task in TASKS.items():
        problem=existing.get(task_id,{'id':task_id,'title':task['title'],'titleZh':task['title_zh'],
            'difficulty':task['difficulty'],'hint':task['hint'],'hintZh':task['hint_zh'],
            'descriptionEn':task['description_en'],'descriptionZh':task['description_zh']})
        problem['functionName']=task['function_name'];problem['tests']=task['tests']
        for test in problem['tests']:ast.parse(test['code'].replace('{fn}',problem['functionName']))
        existing[task_id]=problem
    data['problems']=[problem for key,problem in existing.items() if key in TASKS]
    (destination/'problems.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for name in ['starters.json','solutions.json','paths.json']:
        if not (destination/name).exists():shutil.copy2(ROOT/'web/src/lib'/name,destination/name)
    print(json.dumps({'problems':len(data['problems']),'destination':str(destination.relative_to(ROOT))}))


if __name__=='__main__':main()

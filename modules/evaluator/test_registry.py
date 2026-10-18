"""Validate the task registry used by the standalone evaluator."""
from torch_judge.tasks import TASKS
import pytest


@pytest.mark.parametrize('task_id', TASKS)
def test_registered_tasks_have_valid_tests(task_id):
    task = TASKS[task_id]
    for field in ('title', 'title_zh', 'difficulty', 'function_name', 'tests', 'solution'):
        assert task.get(field), (task_id, field)
    assert isinstance(task['tests'], list)
    assert all(test.get('code', '').strip() for test in task['tests']), task_id
    for test in task['tests']:
        assert test.get('name') and '{fn}' in test['code'], task_id
        compile(test['code'].replace('{fn}', task['function_name']), f"<test:{task_id}>", 'exec')

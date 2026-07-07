import sys
sys.path.insert(0, r'd:\BGWH_Code\pyre-code-main')
from torch_judge.tasks import get_task, TASKS

print(f'Total tasks loaded: {len(TASKS)}')

# Verify all tasks have required fields and valid structure
errors = []
for tid, task in TASKS.items():
    required = ['title', 'title_zh', 'difficulty', 'function_name', 'tests', 'solution']
    for field in required:
        if field not in task:
            errors.append(f'{tid}: missing field {field}')
    if not isinstance(task.get('tests', []), list):
        errors.append(f'{tid}: tests is not a list')
    if not task.get('function_name'):
        errors.append(f'{tid}: empty function_name')
    # Check test code format
    for i, test in enumerate(task.get('tests', [])):
        if 'code' not in test or 'name' not in test:
            errors.append(f'{tid}: test {i} missing code or name')
        elif '{fn}' not in test['code']:
            errors.append(f'{tid}: test {i} missing {{fn}} placeholder')

if errors:
    print(f'Found {len(errors)} errors:')
    for e in errors[:20]:
        print(f'  {e}')
else:
    print('All tasks have valid structure!')

# Show summary of new tasks
print('\nNew tasks summary:')
for tid in ['transformer_encoder', 'transformer_decoder', 'clip_model', 'einsum_ops', 'broadcasting']:
    t = get_task(tid)
    if t:
        print(f'  {tid}: {t["title"]} ({t["difficulty"]}) - {len(t["tests"])} tests - cat={t.get("category", "N/A")}')
    else:
        print(f'  {tid}: MISSING!')

# Show case-sensitive fix summary
print('\nCase-sensitive fix check (mha, gqa, cross_attention):')
for tid in ['mha', 'gqa', 'cross_attention']:
    t = get_task(tid)
    tests = t.get('tests', [])
    fallback_count = sum(1 for test in tests if 'hasattr' in test['code'] and 'else' in test['code'])
    print(f'  {tid}: {fallback_count}/{len(tests)} tests have case fallback')

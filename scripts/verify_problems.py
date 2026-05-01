import json
p = json.load(open(r'd:\BGWH_Code\pyre-code-main\web\src\lib\problems.json', encoding='utf-8'))
print(f'Total problems: {len(p["problems"])}')

for tid in ['transformer_encoder', 'transformer_decoder', 'clip_model', 'einsum_ops', 'broadcasting']:
    prob = next((x for x in p['problems'] if x['id'] == tid), None)
    if prob:
        print(f'{tid}: OK, cat={prob.get("category", "N/A")}, theory={bool(prob.get("theoryEn"))}, diagram={bool(prob.get("diagramEn"))}')
    else:
        print(f'{tid}: MISSING')

from collections import Counter
cats = Counter(x.get('category', 'Other') for x in p['problems'])
print('\nCategory distribution:')
for cat, count in cats.most_common():
    print(f'  {cat}: {count}')

# Verify theory/diagram coverage
with_theory = sum(1 for x in p['problems'] if x.get('theoryEn'))
with_diagram = sum(1 for x in p['problems'] if x.get('diagramEn'))
print(f'\nProblems with theory: {with_theory}/{len(p["problems"])}')
print(f'Problems with diagram: {with_diagram}/{len(p["problems"])}')

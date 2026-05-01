import json
import re

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

all_text = '\n'.join(p['text'] for p in pages)
lines = all_text.split('\n')

# Find all unique main sections and their sub-sections
main_sections = []
current_main = None
current_subs = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    # Main sections
    m = re.match(r'^大模型[（(]LLMs[)）]\s*(.+)[面篇]$', line)
    if m:
        if current_main:
            main_sections.append({'name': current_main, 'subs': current_subs})
        current_main = line
        current_subs = []
    # Sub sections - Chinese numbered
    elif re.match(r'^[一二三四五六七八九十]+、', line) and len(line) < 100:
        current_subs.append(line)
    # Numbered questions
    elif re.match(r'^\d+\s+', line) and len(line) < 100 and not line.startswith('1.') and not line.startswith('2.'):
        current_subs.append(line)
    # Sub sections like 3.1, 4.2
    elif re.match(r'^\d+\.\d+\s+', line) and len(line) < 100:
        current_subs.append('  ' + line)
    # Deep sections
    elif re.match(r'^\d+\.\d+\.\d+\s+', line) and len(line) < 100:
        current_subs.append('    ' + line)

if current_main:
    main_sections.append({'name': current_main, 'subs': current_subs})

# Print structure
for sec in main_sections:
    print(f"\n{'='*60}")
    print(f"【{sec['name']}】")
    print(f"{'='*60}")
    for sub in sec['subs'][:60]:  # limit output
        print(sub)
    if len(sec['subs']) > 60:
        print(f"  ... and {len(sec['subs'])-60} more ...")

# Save structure
with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\structure.json', 'w', encoding='utf-8') as f:
    json.dump(main_sections, f, ensure_ascii=False, indent=2)

print(f"\n\nTotal main sections: {len(main_sections)}")

import json
import re

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

# Concatenate all text
all_text = '\n'.join(p['text'] for p in pages)

# Find chapter/section headers
# Patterns: "大模型（LLMs）基础面", "一、xxx", "1 xxx", "1.1 xxx", "3.3.1 xxx"
lines = all_text.split('\n')

# Collect potential headers
headers = []
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    # Main sections like "大模型（LLMs）基础面"
    if re.match(r'^大模型[（(]LLMs[)）].*[面篇]$', line):
        headers.append(('MAIN', line, i))
    # Numbered sections like "1 xxx", "2 xxx"
    elif re.match(r'^\d+\s+', line) and len(line) < 80 and not line.startswith('1.') and not line.startswith('2.'):
        headers.append(('NUM', line, i))
    # Chinese numbered sections like "一、", "二、"
    elif re.match(r'^[一二三四五六七八九十]+、', line):
        headers.append(('CN_NUM', line, i))
    # Sub sections like "3.1 xxx", "4.2 xxx"
    elif re.match(r'^\d+\.\d+\s+', line) and len(line) < 80:
        headers.append(('SUB', line, i))
    # Deep sub sections like "3.3.1 xxx"
    elif re.match(r'^\d+\.\d+\.\d+\s+', line) and len(line) < 80:
        headers.append(('DEEP', line, i))

print(f"Found {len(headers)} headers\n")
for htype, text, idx in headers[:80]:
    print(f"[{htype:8s}] {text[:70]}")

import json
import re
import os

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

all_text = '\n'.join(p['text'] for p in pages)
lines = all_text.split('\n')

# Chapter patterns
MAIN_PATTERN = re.compile(r'^大模型[（(]LLMs[)）]\s*(.+)[面篇]$')
NUM_PATTERN = re.compile(r'^(\d+)\s+(.+)$')
CN_PATTERN = re.compile(r'^([一二三四五六七八九十]+)、\s*(.+)$')
SUB_PATTERN = re.compile(r'^(\d+\.\d+)\s+(.+)$')
DEEP_PATTERN = re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)$')

def get_level(line):
    line = line.strip()
    if not line:
        return None, None
    m = MAIN_PATTERN.match(line)
    if m:
        return 0, line
    m = CN_PATTERN.match(line)
    if m and len(line) < 120:
        return 1, line
    m = NUM_PATTERN.match(line)
    if m and len(line) < 120 and not line.startswith('1.') and not line.startswith('2.') and not line.startswith('3.') and not line.startswith('4.') and not line.startswith('5.') and not line.startswith('6.') and not line.startswith('7.') and not line.startswith('8.') and not line.startswith('9.'):
        return 1, line
    m = SUB_PATTERN.match(line)
    if m and len(line) < 120:
        return 2, line
    m = DEEP_PATTERN.match(line)
    if m and len(line) < 120:
        return 3, line
    return None, line

# Build chapter tree
chapters = []
current_chapter = None
current_section = None
current_subsection = None

for line in lines:
    level, text = get_level(line)
    if level is None:
        # Content line
        if current_subsection:
            current_subsection['content'].append(text)
        elif current_section:
            current_section['content'].append(text)
        elif current_chapter:
            current_chapter['content'].append(text)
        continue
    
    if level == 0:
        if current_chapter:
            chapters.append(current_chapter)
        current_chapter = {'title': text, 'sections': [], 'content': []}
        current_section = None
        current_subsection = None
    elif level == 1:
        if current_chapter is None:
            continue
        current_section = {'title': text, 'subsections': [], 'content': []}
        current_chapter['sections'].append(current_section)
        current_subsection = None
    elif level == 2:
        if current_section is None:
            continue
        current_subsection = {'title': text, 'subsubsections': [], 'content': []}
        current_section['subsections'].append(current_subsection)
    elif level == 3:
        if current_subsection is None:
            continue
        current_subsection['subsubsections'].append({'title': text, 'content': []})

if current_chapter:
    chapters.append(current_chapter)

# Clean content: join lines and remove empty
for ch in chapters:
    ch['content'] = '\n'.join(l for l in ch['content'] if l.strip()).strip()
    for sec in ch['sections']:
        sec['content'] = '\n'.join(l for l in sec['content'] if l.strip()).strip()
        for sub in sec['subsections']:
            sub['content'] = '\n'.join(l for l in sub['content'] if l.strip()).strip()
            for subsub in sub['subsubsections']:
                subsub['content'] = '\n'.join(l for l in subsub['content'] if l.strip()).strip()

# Save
output_path = r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\chapters.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(chapters, f, ensure_ascii=False, indent=2)

print(f"Saved {len(chapters)} chapters to {output_path}")
for ch in chapters:
    sec_count = len(ch['sections'])
    print(f"  {ch['title']}: {sec_count} sections")

import json
import re
import os

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

all_text = '\n'.join(p['text'] for p in pages)
lines = all_text.split('\n')

# More comprehensive patterns
MAIN_PATTERN = re.compile(r'^大模型[（(]LLMs[)）]\s*(.+)[面篇]$')
# Section patterns
CN_PATTERN = re.compile(r'^([一二三四五六七八九十百千]+)、\s*(.+)')
NUM_Q_PATTERN = re.compile(r'^(\d+)\s+(.+)')  # "1 目前主流..."
SUB_PATTERN = re.compile(r'^(\d+\.\d+)\s+(.+)')
DEEP_PATTERN = re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)')
# Also match list-style questions like "1. 为什么要..."
LIST_PATTERN = re.compile(r'^(\d+)\.\s+(.+)')

def get_level(line):
    line = line.strip()
    if not line:
        return None, None
    # Main chapter
    m = MAIN_PATTERN.match(line)
    if m:
        return 0, line
    # Chinese numbered section
    m = CN_PATTERN.match(line)
    if m and len(line) < 150:
        return 1, line
    # Numbered question (not sub-section like 1.1)
    m = NUM_Q_PATTERN.match(line)
    if m and len(line) < 150:
        num = m.group(1)
        # Exclude if it looks like a sub-section (e.g., "6.1", "3.2")
        if not re.match(r'^\d+\.\d+', line):
            return 1, line
    # Sub section
    m = SUB_PATTERN.match(line)
    if m and len(line) < 150:
        return 2, line
    # Deep section
    m = DEEP_PATTERN.match(line)
    if m and len(line) < 150:
        return 3, line
    return None, line

# Build tree
chapters = []
current_chapter = None
current_section = None
current_subsection = None

for line in lines:
    level, text = get_level(line)
    if level is None:
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
            if current_chapter is None:
                continue
            # Create a default section
            current_section = {'title': '', 'subsections': [], 'content': []}
            current_chapter['sections'].append(current_section)
        current_subsection = {'title': text, 'subsubsections': [], 'content': []}
        current_section['subsections'].append(current_subsection)
    elif level == 3:
        if current_subsection is None:
            if current_section is None:
                continue
            current_subsection = {'title': '', 'subsections': [], 'content': []}
            current_section['subsections'].append(current_subsection)
        current_subsection['subsubsections'].append({'title': text, 'content': []})

if current_chapter:
    chapters.append(current_chapter)

# Clean content
def clean_content(items):
    for item in items:
        if 'content' in item:
            item['content'] = '\n'.join(l for l in item['content'] if l.strip()).strip()
        if 'sections' in item:
            clean_content(item['sections'])
        if 'subsections' in item:
            clean_content(item['subsections'])
        if 'subsubsections' in item:
            clean_content(item['subsubsections'])

clean_content(chapters)

# Merge duplicate chapters (same title)
merged = {}
for ch in chapters:
    key = ch['title']
    if key not in merged:
        merged[key] = ch
    else:
        # Merge sections and content
        merged[key]['sections'].extend(ch['sections'])
        if ch['content']:
            merged[key]['content'] = (merged[key]['content'] + '\n\n' + ch['content']).strip()

chapters = list(merged.values())

# Save
output_path = r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\chapters_v2.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(chapters, f, ensure_ascii=False, indent=2)

print(f"Saved {len(chapters)} chapters")
for ch in chapters:
    sec_count = len(ch['sections'])
    sub_count = sum(len(s.get('subsections', [])) for s in ch['sections'])
    content_len = len(ch.get('content', ''))
    print(f"  {ch['title'][:35]:35s} | sec={sec_count:3d} | sub={sub_count:3d} | content={content_len:5d}")

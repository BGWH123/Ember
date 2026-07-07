import json
import re

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\chapters_v2.json', 'r', encoding='utf-8') as f:
    chapters = json.load(f)

# For chapters with 0 sections but with content, split into Q&A pairs
def split_qa(content):
    """Split content into question-answer pairs based on number patterns."""
    if not content:
        return []
    # Pattern: number followed by question
    pattern = re.compile(r'(?:^|\n)\s*(\d+)[\.、\s]+([^\n]+)\n')
    matches = list(pattern.finditer(content))
    if len(matches) < 2:
        return [{'question': '', 'answer': content}]
    
    result = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(content)
        block = content[start:end].strip()
        lines = block.split('\n', 1)
        q = lines[0].strip()
        a = lines[1].strip() if len(lines) > 1 else ''
        result.append({'question': q, 'answer': a})
    return result

# Build final gossip data
gossip_data = []
for ch in chapters:
    title = ch['title']
    # Skip empty chapters
    has_content = bool(ch.get('content', '').strip()) or any(
        s.get('content', '').strip() or s.get('subsections', []) 
        for s in ch.get('sections', [])
    )
    if not has_content:
        continue
    
    chapter_item = {
        'id': re.sub(r'[^\w]', '_', title),
        'title': title,
        'sections': []
    }
    
    # Process sections
    sections = ch.get('sections', [])
    if not sections and ch.get('content'):
        # Split chapter content into Q&A
        qa_pairs = split_qa(ch['content'])
        for qa in qa_pairs:
            chapter_item['sections'].append({
                'title': qa['question'],
                'content': qa['answer'],
                'subsections': []
            })
    else:
        for sec in sections:
            sec_item = {
                'title': sec.get('title', ''),
                'content': sec.get('content', ''),
                'subsections': []
            }
            for sub in sec.get('subsections', []):
                sub_item = {
                    'title': sub.get('title', ''),
                    'content': sub.get('content', ''),
                    'subsubsections': []
                }
                for subsub in sub.get('subsubsections', []):
                    sub_item['subsubsections'].append({
                        'title': subsub.get('title', ''),
                        'content': subsub.get('content', '')
                    })
                sec_item['subsections'].append(sub_item)
            chapter_item['sections'].append(sec_item)
    
    gossip_data.append(chapter_item)

# Clean up: remove empty items
def clean_empty(items, key='sections'):
    result = []
    for item in items:
        has_content = bool(item.get('content', '').strip())
        has_children = bool(item.get(key, []))
        if has_content or has_children:
            result.append(item)
    return result

for ch in gossip_data:
    ch['sections'] = clean_empty(ch['sections'], 'subsections')
    for sec in ch['sections']:
        sec['subsections'] = clean_empty(sec['subsections'], 'subsubsections')

# Save
output_path = r'd:\BGWH_Code\pyre-code-main\web\public\gossip-data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(gossip_data, f, ensure_ascii=False, indent=2)

print(f"Built gossip data: {len(gossip_data)} chapters")
for ch in gossip_data:
    sec_count = len(ch['sections'])
    sub_count = sum(len(s.get('subsections', [])) for s in ch['sections'])
    print(f"  {ch['title'][:35]:35s} | {sec_count:3d} sections | {sub_count:3d} subsections")

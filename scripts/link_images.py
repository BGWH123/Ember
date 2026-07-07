import json
import os
import re

# Load data
with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages_data = json.load(f)

with open(r'd:\BGWH_Code\pyre-code-main\web\public\gossip-images\index.json', 'r', encoding='utf-8') as f:
    image_index = json.load(f)

with open(r'd:\BGWH_Code\pyre-code-main\web\public\gossip-data.json', 'r', encoding='utf-8') as f:
    gossip_data = json.load(f)

# Build page-to-chapter mapping
MAIN_PATTERN = re.compile(r'^大模型[（(]LLMs[)）]\s*(.+)[面篇]$')
page_chapters = {}
current_chapter = None
for p in pages_data:
    page_num = p['page']
    text = p['text']
    for line in text.split('\n'):
        line = line.strip()
        if MAIN_PATTERN.match(line):
            current_chapter = line
    page_chapters[page_num] = current_chapter

# Group images by chapter
chapter_images = {}
for img in image_index:
    ch = img['chapter']
    if ch not in chapter_images:
        chapter_images[ch] = []
    chapter_images[ch].append({
        'filename': img['filename'],
        'page': img['page'],
    })

# Update gossip data with images
for ch in gossip_data:
    ch_title = ch['title']
    if ch_title in chapter_images:
        ch['images'] = chapter_images[ch_title]
    else:
        ch['images'] = []

# Save updated data
with open(r'd:\BGWH_Code\pyre-code-main\web\public\gossip-data.json', 'w', encoding='utf-8') as f:
    json.dump(gossip_data, f, ensure_ascii=False, indent=2)

# Stats
for ch in gossip_data:
    print(f"{ch['title'][:35]:35s} | {len(ch['images'])} images")

print(f"\nTotal images linked: {sum(len(ch['images']) for ch in gossip_data)}")

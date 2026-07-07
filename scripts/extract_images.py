import fitz  # PyMuPDF
import os
import json

pdf_path = r'D:\BaiduNetdiskDownload\大模型面试题与答案（全集）.pdf'
output_dir = r'd:\BGWH_Code\pyre-code-main\web\public\gossip-images'
os.makedirs(output_dir, exist_ok=True)

# Load page-to-chapter mapping from earlier extraction
with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\pages_raw.json', 'r', encoding='utf-8') as f:
    pages_data = json.load(f)

# Build page-to-chapter mapping
import re
MAIN_PATTERN = re.compile(r'^大模型[（(]LLMs[)）]\s*(.+)[面篇]$')

page_chapters = {}
current_chapter = None
for p in pages_data:
    page_num = p['page']
    text = p['text']
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if MAIN_PATTERN.match(line):
            current_chapter = line
    page_chapters[page_num] = current_chapter

# Extract images
doc = fitz.open(pdf_path)
image_records = []
img_counter = 0

for page_idx in range(len(doc)):
    page = doc[page_idx]
    page_num = page_idx + 1
    chapter = page_chapters.get(page_num, '未知章节')

    # Get images
    image_list = page.get_images(full=True)
    for img_idx, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Skip small images (likely icons)
            if len(image_bytes) < 2048:
                continue

            img_counter += 1
            filename = f"img_{img_counter:03d}_p{page_num}.{image_ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(image_bytes)

            image_records.append({
                'id': img_counter,
                'filename': filename,
                'page': page_num,
                'chapter': chapter,
                'size': len(image_bytes),
                'width': base_image.get('width', 0),
                'height': base_image.get('height', 0),
            })
        except Exception as e:
            print(f"Error extracting image on page {page_num}: {e}")

doc.close()

# Save image index
with open(r'd:\BGWH_Code\pyre-code-main\web\public\gossip-images\index.json', 'w', encoding='utf-8') as f:
    json.dump(image_records, f, ensure_ascii=False, indent=2)

print(f"Extracted {img_counter} images to {output_dir}")
print(f"Saved index with {len(image_records)} records")

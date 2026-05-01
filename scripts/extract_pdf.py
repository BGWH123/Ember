import pdfplumber
import json
import os

pdf_path = r'D:\BaiduNetdiskDownload\大模型面试题与答案（全集）.pdf'
output_dir = r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract'
os.makedirs(output_dir, exist_ok=True)

pages_data = []
with pdfplumber.open(pdf_path) as pdf:
    total = len(pdf.pages)
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        pages_data.append({'page': i+1, 'text': text})
        if (i+1) % 50 == 0:
            print(f'Extracted {i+1}/{total} pages')

# Save extracted text
with open(os.path.join(output_dir, 'pages_raw.json'), 'w', encoding='utf-8') as f:
    json.dump(pages_data, f, ensure_ascii=False, indent=2)

print(f'\nDone! Total pages: {len(pages_data)}')
print(f'Saved to: {output_dir}')

# Show structure preview
for p in pages_data[:10]:
    lines = [l for l in p['text'].split('\n') if l.strip()][:2]
    print(f"P{p['page']}: {' | '.join(lines)}")

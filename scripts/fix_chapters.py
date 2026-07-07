import json

with open(r'd:\BGWH_Code\pyre-code-main\scripts\pdf_extract\chapters.json', 'r', encoding='utf-8') as f:
    chs = json.load(f)

# Show chapters with 0 sections but with content
for ch in chs:
    sec_count = len(ch.get('sections', []))
    content_len = len(ch.get('content', ''))
    if sec_count == 0 and content_len > 100:
        print(f"=== {ch['title']} (content={content_len}) ===")
        print(ch['content'][:800])
        print("\n" + "="*60 + "\n")

"""WordPiece tokenizer task."""

TASK = {
    "title": "WordPiece Subword Tokenizer",
    "title_zh": "WordPiece 子词分词器",
    "difficulty": "Medium",
    "category": "分词",
    "function_name": "wordpiece_encode",
    "starter_code": '''def wordpiece_encode(text, vocab, unk_token='[UNK]', max_chars=100):
    pass
''',
    "hint": "For each start position, scan candidate ends backward. Buffer a word's pieces and append them only if the whole word can be segmented.",
    "hint_zh": "对每个起点从后向前扫描候选终点。先缓存单词的子词，只有整个单词都能切分时才加入结果。",
    "description_en": "Implement greedy longest-match-first WordPiece encoding. Continuation pieces use the `##` prefix, and any failed word is replaced as a whole by the unknown token.",
    "description_zh": "实现最长匹配优先的贪心 WordPiece 编码。续接子词使用 `##` 前缀；任一位置匹配失败时，整个单词替换为未知标记。",
    "theory_en": "WordPiece segments each whitespace-delimited word independently. Greedy longest matching prefers fewer, longer vocabulary pieces while the continuation marker distinguishes word-internal pieces.",
    "theory_zh": "WordPiece 独立切分每个由空白分隔的单词。贪心最长匹配倾向于更少、更长的词表片段，续接标记用于区分词内片段。",
    "diagram_en": "```mermaid\nflowchart LR\nW[word] --> L[longest prefix]\nL --> C[longest ## continuation]\nC --> O[pieces]\nL -->|failure| U[UNK]\nC -->|failure| U\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nW[单词] --> L[最长前缀]\nL --> C[最长 ## 续接]\nC --> O[子词列表]\nL -->|失败| U[未知标记]\nC -->|失败| U\n```",
    "tests": [
        {"name": "greedy segmentation", "code": """import torch
vocab = {'play', '##ing', '##er', 'player'}
out = {fn}('playing player', vocab)
assert isinstance(out, list) and len(out) == 3
assert out == ['play', '##ing', 'player']
assert torch.tensor([len(out)]).shape == (1,)
"""},
        {"name": "longest match wins", "code": """import torch
vocab = {'a', 'ab', 'abc', '##b', '##bc'}
out = {fn}('abc', vocab)
assert out == ['abc']
assert torch.tensor(len(out)).item() == 1
"""},
        {"name": "failed word becomes one unknown", "code": """import torch
vocab = {'foo', 'bar', '##x'}
out = {fn}('fooz bar', vocab, unk_token='<unk>')
assert out == ['<unk>', 'bar']
assert torch.tensor([len(x) for x in out]).shape == (2,)
"""},
        {"name": "length limit and whitespace", "code": """import torch
vocab = {'ok'}
out = {fn}('   toolong   ok  ', vocab, max_chars=3)
assert out == ['[UNK]', 'ok']
assert torch.tensor(len(out)).shape == ()
"""},
    ],
    "solution": '''def wordpiece_encode(text, vocab, unk_token='[UNK]', max_chars=100):
    output = []
    for word in text.strip().split():
        if len(word) > max_chars:
            output.append(unk_token)
            continue
        pieces = []
        start = 0
        failed = False
        while start < len(word):
            match = None
            match_end = start
            for end in range(len(word), start, -1):
                candidate = word[start:end]
                if start > 0:
                    candidate = '##' + candidate
                if candidate in vocab:
                    match = candidate
                    match_end = end
                    break
            if match is None:
                failed = True
                break
            pieces.append(match)
            start = match_end
        if failed:
            output.append(unk_token)
        else:
            output.extend(pieces)
    return output
''',
    "demo": '''vocab = {'un', '##aff', '##able', 'hello'}
print(wordpiece_encode('hello unaffable', vocab))
''',
    "explanation": "A temporary per-word list is essential: if a later character cannot be matched, earlier pieces from that word must be discarded before emitting one unknown token.",
}

"""WordPiece Subword Tokenizer task."""

TASK = {
    'title': 'WordPiece Subword Tokenizer',
    'title_zh': 'WordPiece 子词分词器',
    'difficulty': 'Medium',
    'category': '分词',
    'function_name': 'wordpiece_encode',
    'starter_code': """def wordpiece_encode(text, vocab, unk_token='[UNK]', max_chars=100):
    pass""",
    'hint': """For each start position, scan candidate ends backward. Buffer a word's pieces and append them only if the whole word can be segmented.""",
    'hint_zh': '对每个起点从后向前扫描候选终点。先缓存单词的子词，只有整个单词都能切分时才加入结果。',
    'description_en': """Implement greedy longest-match-first WordPiece encoding. Continuation pieces use the `##` prefix, and any failed word is replaced as a whole by the unknown token.""",
    'description_zh': '实现最长匹配优先的贪心 WordPiece 编码。续接子词使用 `##` 前缀；任一位置匹配失败时，整个单词替换为未知标记。',
    'theory_en': """WordPiece segments each whitespace-delimited word independently. Greedy longest matching prefers fewer, longer vocabulary pieces while the continuation marker distinguishes word-internal pieces.""",
    'theory_zh': 'WordPiece 独立切分每个由空白分隔的单词。贪心最长匹配倾向于更少、更长的词表片段，续接标记用于区分词内片段。',
    'diagram_en': """```mermaid
flowchart LR
W[word] --> L[longest prefix]
L --> C[longest ## continuation]
C --> O[pieces]
L -->|failure| U[UNK]
C -->|failure| U
```""",
    'diagram_zh': """```mermaid
flowchart LR
W[单词] --> L[最长前缀]
L --> C[最长 ## 续接]
C --> O[子词列表]
L -->|失败| U[未知标记]
C -->|失败| U
```""",
    "tests": [
        {
            "name": 'greedy segmentation',
            "code": """import torch
vocab = {'play', '##ing', '##er', 'player'}
out = {fn}('playing player', vocab)
assert isinstance(out, list) and len(out) == 3
assert out == ['play', '##ing', 'player']
assert torch.tensor([len(out)]).shape == (1,)""",
        },
        {
            "name": 'longest match wins',
            "code": """import torch
vocab = {'a', 'ab', 'abc', '##b', '##bc'}
out = {fn}('abc', vocab)
assert out == ['abc']
assert torch.tensor(len(out)).item() == 1""",
        },
        {
            "name": 'failed word becomes one unknown',
            "code": """import torch
vocab = {'foo', 'bar', '##x'}
out = {fn}('fooz bar', vocab, unk_token='<unk>')
assert out == ['<unk>', 'bar']
assert torch.tensor([len(x) for x in out]).shape == (2,)""",
        },
        {
            "name": 'length limit and whitespace',
            "code": """import torch
vocab = {'ok'}
out = {fn}('   toolong   ok  ', vocab, max_chars=3)
assert out == ['[UNK]', 'ok']
assert torch.tensor(len(out)).shape == ()""",
        },
    ],
    'solution': '''def wordpiece_encode(text, vocab, unk_token='[UNK]', max_chars=100):
    """实现WordPiece 子词分词器（WordPiece Subword Tokenizer）。"""
    # Step 1: 按空白划分独立单词，并对超长词直接发出整词 UNK
    output = []
    # Step 2: 从每个字符起点向后做最长匹配，词内续片添加 ## 前缀
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
    # Step 3: 仅在整词完全可切分时提交暂存片段，否则回滚为单个 UNK
    return output''',
    'demo': """torch.manual_seed(42)
vocab = {"un", "##aff", "##able", "hello"}
tokens = wordpiece_encode("hello unaffable", vocab)
print("Tokens:", tokens)
print("Token count:", len(tokens))""",
    'explanation': """### 核心思路
WordPiece 对每个空白分隔词独立执行 longest-match-first：在字符位置 $s$ 选择词表中覆盖该位置的最长候选，$s=0$ 使用裸片段，$s>0$ 必须使用 `##` 续接标记。它是确定性贪心算法；若任一位置无候选，整个词原子化为一个 `unk_token`。

### 代码步骤拆解
- **Step 1（单词级边界处理）**：`text.strip().split()` 忽略多余空白；长度超过 `max_chars` 的词不尝试二次方扫描，直接输出 UNK。
- **Step 2（最长前缀搜索）**：对当前 `start` 令 `end` 从词尾递减，第一个命中 `vocab` 的候选即最长匹配；续片查询时在字符内容前加 `##`。
- **Step 3（事务式提交）**：匹配片段先存入局部 `pieces`，只有 `start` 到达词尾才 `extend`；失败则丢弃局部结果并仅 `append(unk_token)`。

### 重点与易错点分析
- **注意点 1（整词回滚）**：不能边匹配边写入全局输出，否则后缀失败时会留下部分 token，再附加 UNK，违反 WordPiece 的整词未知规则。
- **注意点 2（## 不是原文字符）**：`##` 仅是词表中标识非词首的命名约定，不参与 `start/end` 的字符偏移；首片段绝不能自动加前缀。
- **注意点 3（与 Unigram 对比）**：WordPiece 选择局部最长项，不比较概率，也不保证任何全局路径目标；`max_chars` 同时限制最坏运行时间并防止异常长词拖慢分词。

### 复杂度分析
- **时间复杂度**：单词长度为 $L$ 时最多检查 $O(L^2)$ 个候选；若计 Python 子串构造，最坏字符复制量为 $O(L^3)$。
- **空间复杂度**：单词暂存片段及最终 token 列表的总字符规模为 $O(L)$。""",
}

"""Byte-Pair Encoding (BPE) task."""

TASK = {
    'title': 'Byte-Pair Encoding (BPE)',
    'title_zh': '字节对编码（BPE）',
    'difficulty': 'Hard',
    'category': '分词',
    'function_name': 'SimpleBPE',
    'starter_code': """class SimpleBPE:
    pass""",
    'hint': """train: split each word → chars + `</w>`, count adjacent pairs, merge most frequent, repeat
encode: apply learned merges in order to split text into subword tokens""",
    'hint_zh': """train：每个词拆分为字符 + `</w>`，统计相邻对频率，合并最高频对，重复
encode：按学习到的合并顺序将文本拆分为子词 token""",
    'description_en': """Implement Byte-Pair Encoding (BPE) tokenization.

BPE iteratively merges the most frequent adjacent symbol pairs to build a subword vocabulary, widely used in modern language models.

**Signature:** `SimpleBPE()` (class)

**Methods:**
- `train(corpus: list[str], num_merges: int)` — learn merge rules from a word list
- `encode(text: str) -> list[str]` — tokenize text into subword tokens

**Constraints:**
- Append `</w>` to mark word boundaries
- `self.merges` stores learned pairs in order
- `encode` applies merges sequentially""",
    'description_zh': """实现字节对编码（BPE）分词。

BPE 通过迭代合并最频繁的相邻符号对来构建子词词表，广泛用于现代语言模型。

**签名:** `SimpleBPE()`（类）

**方法:**
- `train(corpus: list[str], num_merges: int)` — 从词列表中学习合并规则
- `encode(text: str) -> list[str]` — 将文本分词为子词 token

**约束:**
- 使用 `</w>` 标记词边界
- `self.merges` 按顺序存储学习到的合并对
- `encode` 按顺序应用合并规则""",
    "tests": [
        {
            "name": 'Correct number of merges',
            "code": """bpe = {fn}()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=5)
assert len(bpe.merges) == 5, f'Expected 5 merges, got {len(bpe.merges)}'""",
        },
        {
            "name": 'Most frequent pair merged first',
            "code": """bpe = {fn}()
bpe.train(['aaa', 'aaa', 'aaa', 'bbb'], num_merges=1)
assert bpe.merges[0] == ('a', 'a'), f'First merge: {bpe.merges[0]}'""",
        },
        {
            "name": 'Encode returns list of strings',
            "code": """bpe = {fn}()
bpe.train(['low', 'lower', 'lowest'] * 3, num_merges=10)
tokens = bpe.encode('low')
assert isinstance(tokens, list), 'encode must return a list'
assert all(isinstance(t, str) for t in tokens), 'tokens must be strings'
reconstructed = ''.join(t.replace('</w>', '') for t in tokens)
assert reconstructed == 'low', f'Reconstruction: {reconstructed}'""",
        },
        {
            "name": 'More merges -> fewer tokens',
            "code": """bpe1 = {fn}()
bpe1.train(['hello'] * 10, num_merges=2)
bpe2 = {fn}()
bpe2.train(['hello'] * 10, num_merges=10)
assert len(bpe2.encode('hello')) <= len(bpe1.encode('hello')), 'More merges should reduce tokens'""",
        },
        {
            "name": 'Encode compresses known corpus',
            "code": """# Corpus where 'aa' is the dominant pair; after 3 merges 'aab' should encode to fewer tokens than chars
bpe = {fn}()
bpe.train(['aaa', 'aaa', 'aaa', 'aab', 'aab'], num_merges=3)
encoded = bpe.encode('aab')
assert isinstance(encoded, list), 'encode should return a list'
assert len(encoded) > 0, 'encode should return non-empty list'
# 'aa' must be a learned merge (most frequent pair), so 'aab' -> ['aa', 'b', '</w>'] or similar
assert len(encoded) < len('aab') + 1, f'BPE should compress: "aab" encoded to {encoded} ({len(encoded)} tokens) but expected fewer than {len("aab") + 1}'""",
        },
        {
            "name": 'Encode output reconstructs original word',
            "code": """bpe = {fn}()
bpe.train(['lowest', 'newer', 'wider'] * 5, num_merges=6)
for word in ['low', 'new', 'wide']:
    tokens = bpe.encode(word)
    reconstructed = ''.join(t.replace('</w>', '') for t in tokens)
    assert reconstructed == word, f'Reconstruction failed: {word!r} -> {tokens} -> {reconstructed!r}'""",
        },
    ],
    'solution': '''class SimpleBPE:
    """简单字节对编码器 (Byte-Pair Encoding)。"""
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        # Step 1: 统计带词尾标记的初始词表
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ("</w>",)
            vocab[symbols] = vocab.get(symbols, 0) + 1
        self.merges = []

        # Step 2: 反复选择频率最高的相邻符号对并合并
        for _ in range(num_merges):
            pair_counts = {}
            for symbols, frequency in vocab.items():
                for index in range(len(symbols) - 1):
                    pair = (symbols[index], symbols[index + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + frequency
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)
            next_vocab = {}
            for symbols, frequency in vocab.items():
                merged = []
                index = 0
                while index < len(symbols):
                    if (
                        index + 1 < len(symbols)
                        and (symbols[index], symbols[index + 1]) == best_pair
                    ):
                        merged.append(symbols[index] + symbols[index + 1])
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                next_vocab[tuple(merged)] = frequency
            vocab = next_vocab

    def encode(self, text):
        # Step 3: 按训练顺序将合并规则应用到每个单词
        tokens = []
        for word in text.split():
            symbols = list(word) + ["</w>"]
            for left, right in self.merges:
                index = 0
                while index < len(symbols) - 1:
                    if symbols[index] == left and symbols[index + 1] == right:
                        symbols = (
                            symbols[:index]
                            + [left + right]
                            + symbols[index + 2:]
                        )
                    else:
                        index += 1
            tokens.extend(symbols)
        return tokens''',
    'demo': """torch.manual_seed(42)
bpe = SimpleBPE()
bpe.train(["low", "low", "lower", "newest"], num_merges=6)
tokens = bpe.encode("low lower")
print("Merge count:", len(bpe.merges))
print("Merges:", bpe.merges)
print("Tokens:", tokens)""",
    'explanation': """### 核心思路
BPE 从字符词表开始，按语料加权频率反复合并最常见相邻符号对，并按相同顺序编码新文本。

### 代码步骤拆解
- **Step 1（词表统计）**：把每个单词拆成字符并追加 `</w>`，累计重复词频。
- **Step 2（贪心训练）**：统计全部相邻对，选择最高频 `best_pair` 并更新整个词表。
- **Step 3（规则编码）**：逐词、逐规则扫描 `symbols`，输出最终子词 token。

### 重点与易错点分析
- **注意点 1（词频权重）**：相邻对计数必须乘单词频率，否则重复语料不会影响合并。
- **注意点 2（规则顺序）**：编码时必须保持训练得到的 `merges` 顺序，交换规则会改变切分。
- **注意点 3（词尾标记）**：`</w>` 防止跨单词边界合并，并支持无歧义重建。

### 复杂度分析
- **时间复杂度**：训练约为 $O(MVL)$，其中 $M$ 为合并轮数、$V$ 为词表规模、$L$ 为平均符号数。
- **空间复杂度**：$O(VL+M)$，保存词表符号序列和合并规则。""",
}

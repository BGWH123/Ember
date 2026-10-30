"""Unigram LM Viterbi Subword Segmentation task."""

TASK = {
    'title': 'Unigram LM Viterbi Subword Segmentation',
    'title_zh': 'Unigram 维特比最优子词切分',
    'difficulty': 'Hard',
    'category': '分词',
    'function_name': 'unigram_viterbi',
    'starter_code': """def unigram_viterbi(text, token_log_probs, unk_log_prob=-10.0):
    pass""",
    'hint': """Let each character boundary be a DP state. Store the best predecessor for every reachable end position, then reverse the backtracked pieces.""",
    'hint_zh': '将每个字符边界作为 DP 状态。为每个可达终点保存最佳前驱，最后回溯并反转子词。',
    'description_en': """Use Viterbi dynamic programming to find the maximum-log-probability unigram segmentation. Permit an unknown single-character fallback only when no vocabulary transition reaches that end position.""",
    'description_zh': '使用维特比动态规划寻找对数概率最大的 Unigram 切分。仅当没有词表转移到达当前终点时，才允许未知单字符回退。',
    'theory_en': """Log probabilities turn a product of independent piece probabilities into an additive path score. The DP keeps only the best path to each boundary, giving an exact optimum over all segmentations.""",
    'theory_zh': '对数概率将独立子词概率的乘积变成可加路径分数。DP 只保留到每个边界的最佳路径，从而精确得到所有切分中的最优解。',
    'diagram_en': """```mermaid
flowchart LR
Z[boundary 0] -->|piece score| I[boundary i]
I -->|piece score| N[boundary N]
N --> B[backtrack best path]
```""",
    'diagram_zh': """```mermaid
flowchart LR
Z[边界 0] -->|子词分数| I[边界 i]
I -->|子词分数| N[边界 N]
N --> B[回溯最佳路径]
```""",
    "tests": [
        {
            "name": 'finds global optimum',
            "code": """import torch
pieces, score = {fn}('abcd', {'a': -1.0, 'ab': -0.2, 'bc': -0.1, 'cd': -0.2, 'd': -1.0})
assert pieces == ['ab', 'cd']
assert abs(score - (-0.4)) < 1e-12
assert torch.tensor([score]).shape == (1,)""",
        },
        {
            "name": 'unknown fallback',
            "code": """import torch
pieces, score = {fn}('axb', {'a': -0.1, 'b': -0.2}, unk_log_prob=-3.0)
assert pieces == ['a', 'x', 'b']
assert abs(score + 3.3) < 1e-12
assert torch.isfinite(torch.tensor(score))""",
        },
        {
            "name": 'empty text',
            "code": """import torch
pieces, score = {fn}('', {'x': -1.0})
assert pieces == []
assert score == 0.0
assert torch.tensor(pieces).shape == (0,)""",
        },
        {
            "name": 'strict tie keeps first start',
            "code": """import torch
pieces, score = {fn}('aa', {'a': -0.5, 'aa': -1.0})
assert pieces == ['aa']
assert score == -1.0
assert torch.tensor(score).shape == ()""",
        },
    ],
    'solution': '''def unigram_viterbi(text, token_log_probs, unk_log_prob=-10.0):
    """实现Unigram 维特比最优子词切分（Unigram LM Viterbi Subword Segmentation）。"""
    # Step 1: 以字符边界建立 Viterbi 分数与前驱表，枚举词表边更新最优路径
    n = len(text)
    best_score = [float('-inf')] * (n + 1)
    backptr = [-1] * (n + 1)
    best_score[0] = 0.0
    for end in range(1, n + 1):
        for start in range(end):
            piece = text[start:end]
            if piece in token_log_probs:
                score = best_score[start] + token_log_probs[piece]
                if score > best_score[end]:
                    best_score[end] = score
                    backptr[end] = start
        char = text[end - 1:end]
        if char not in token_log_probs and best_score[end] == float('-inf'):
            best_score[end] = best_score[end - 1] + unk_log_prob
            backptr[end] = end - 1
    # Step 2: 从末尾沿 backptr 回溯每个最优子词对应的起始边界
    pieces = []
    end = n
    while end > 0:
        start = backptr[end]
        pieces.append(text[start:end])
        end = start
    pieces.reverse()
    # Step 3: 反转回溯片段，返回从左到右的最优切分及累计对数概率
    return (pieces, float(best_score[n]))''',
    'demo': """torch.manual_seed(42)
token_log_probs = {"un": -0.2, "i": -1.0, "gram": -0.3, "unigram": -1.0}
pieces, score = unigram_viterbi("unigram", token_log_probs)
print("Best pieces:", pieces)
print("Path score:", score)""",
    'explanation': r"""### 核心思路
把文本的 $0\ldots L$ 字符边界视为有向无环图节点，词表片段 `text[start:end]` 是权重为其 log-prob 的边。Viterbi 递推 $dp[e]=\max_{s<e}\{dp[s]+\log p(x_{s:e})\}$ 求从边界 0 到 $L$ 的最大权路径，对数域把子词概率乘积变为可加分数。

### 代码步骤拆解
- **Step 1（边界动态规划）**：`best_score[0]=0`，其余为 `-inf`；对每个 `end` 枚举 `start`，用已可达前缀和词表分数更新 `best_score[end]`、`backptr[end]`。
- **Step 2（最优路径回溯）**：从 `end=n` 反复读取 `start=backptr[end]`，收集路径边对应的 `text[start:end]`，直至边界 0。
- **Step 3（恢复正序切分）**：逆序回溯结果后返回 `pieces` 与 `best_score[n]` 的 Python 浮点值。

### 重点与易错点分析
- **注意点 1（UNK 回退约束）**：未知单字符边只在没有任何词表边到达当前 `end` 时加入；无条件加入会让 UNK 与合法词表路径竞争，改变题目定义。
- **注意点 2（平局规则）**：更新使用严格 `>`，因此同分时保留先枚举的较小 `start`，即更长的当前片段；改成 `>=` 会改变 `aa` 的平局切分。
- **注意点 3（与 WordPiece 对比）**：Unigram Viterbi 比较整条路径概率并得到全局最优，不能用最长前缀贪心替代；空文本对应零条边，最优分数自然为 0。

### 复杂度分析
- **时间复杂度**：枚举全部 $O(L^2)$ 个子串边；Python 字符串切片若计复制成本，最坏可达 $O(L^3)$，哈希查表本身均摊 $O(1)$。
- **空间复杂度**：`best_score`、`backptr` 与回溯结果均为 $O(L)$。""",
}

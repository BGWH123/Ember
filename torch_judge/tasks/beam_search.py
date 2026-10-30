"""Beam Search Decoding task."""

TASK = {
    'title': 'Beam Search Decoding',
    'title_zh': '束搜索解码',
    'difficulty': 'Medium',
    'category': '采样与解码',
    'function_name': 'beam_search',
    'starter_code': """def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    pass""",
    'hint': """1. beams = [(score=0.0, seq=[start_token])]
2. each step: expand each beam with all tokens → keep top beam_width by cumulative score
3. stop when all beams end with eos_token or max_len reached
4. return seq from highest-scoring beam""",
    'hint_zh': """1. beams = [(score=0.0, seq=[start_token])]
2. 每步：用所有 token 扩展每个候选 → 按累积分保留前 beam_width 个
3. 所有候选以 eos_token 结尾或达到 max_len 时停止
4. 返回得分最高的序列""",
    'description_en': """Implement beam search decoding for sequence generation.

Beam search maintains multiple candidate sequences (beams) at each step, expanding and pruning to find the highest-scoring sequence.

**Signature:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`

**Parameters:**
- `log_prob_fn` — callable that takes a token sequence tensor and returns log-probabilities over vocabulary
- `start_token` — integer start token
- `beam_width` — number of beams to keep
- `eos_token` — end-of-sequence token

**Returns:** list of token IDs for the best sequence

**Constraints:**
- Stop when all beams end with eos or max_len is reached
- Return the highest-scoring complete sequence""",
    'description_zh': """实现序列生成的束搜索解码。

束搜索在每一步维护多个候选序列（束），通过扩展和剪枝找到得分最高的序列。

**签名:** `beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]`

**参数:**
- `log_prob_fn` — 接受 token 序列张量并返回词表上对数概率的可调用对象
- `start_token` — 起始 token 整数
- `beam_width` — 保留的束数量
- `eos_token` — 序列结束 token

**返回:** 最佳序列的 token ID 列表

**约束:**
- 当所有束以 eos 结尾或达到 max_len 时停止
- 返回得分最高的完整序列""",
    "tests": [
        {
            "name": 'Returns list starting with start_token',
            "code": """import torch
def dummy(tokens): return torch.zeros(10)
seq = {fn}(dummy, start_token=0, max_len=5, beam_width=3, eos_token=9)
assert isinstance(seq, list), 'Must return a list'
assert seq[0] == 0, f'First token: {seq[0]}'""",
        },
        {
            "name": 'Greedy path (beam=1)',
            "code": """import torch
def greedy_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp
seq = {fn}(greedy_fn, start_token=0, max_len=5, beam_width=1, eos_token=4)
assert seq == [0, 1, 2, 3, 4], f'Greedy: {seq}'""",
        },
        {
            "name": 'Beam finds better path than greedy',
            "code": """import torch
def tricky(tokens):
    lp = torch.full((6,), -100.0)
    if len(tokens) == 1:
        lp[1] = -1.0; lp[2] = -0.5
    elif tokens[-1] == 1:
        lp[5] = 0.0
    elif tokens[-1] == 2:
        lp[5] = -10.0
    else:
        lp[5] = 0.0
    return lp
seq = {fn}(tricky, start_token=0, max_len=5, beam_width=2, eos_token=5)
assert seq == [0, 1, 5], f'Beam should find [0,1,5], got {seq}'""",
        },
        {
            "name": 'Stops at eos',
            "code": """import torch
def eos_fn(tokens):
    lp = torch.zeros(4); lp[3] = 10.0; return lp
seq = {fn}(eos_fn, start_token=0, max_len=100, beam_width=2, eos_token=3)
assert seq[-1] == 3 and len(seq) == 2, f'Should be [0,3], got {seq}'""",
        },
    ],
    'solution': '''def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    """束搜索解码 (Beam Search Decoding)。"""
    beams = [(0.0, [start_token])]
    completed = []

    # Step 1: 逐轮扩展每条未完成候选序列
    for _ in range(max_len):
        candidates = []
        for score, sequence in beams:
            if sequence[-1] == eos_token:
                completed.append((score, sequence))
                continue
            log_probs = log_prob_fn(torch.tensor(sequence))
            top_scores, top_tokens = log_probs.topk(beam_width)
            for index in range(beam_width):
                candidates.append((
                    score + top_scores[index].item(),
                    sequence + [top_tokens[index].item()],
                ))

        # Step 2: 按累计对数概率保留最优 beam
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_width]

    # Step 3: 在完成与未完成候选中选取最高分序列
    all_sequences = completed + beams
    all_sequences.sort(key=lambda item: item[0], reverse=True)
    return all_sequences[0][1]''',
    'demo': """torch.manual_seed(42)
def next_log_probs(tokens):
    scores = torch.full((5,), -10.0)
    scores[min(len(tokens), 4)] = 0.0
    return scores
sequence = beam_search(next_log_probs, 0, 5, 2, 4)
print("Output length:", len(sequence))
print("Sequence:", sequence)""",
    'explanation': r"""### 核心思路
每轮把每个候选扩展为概率最高的若干后继，用累计对数概率排序并只保留 `beam_width` 条路径。

### 代码步骤拆解
- **Step 1（候选扩展）**：调用 `log_prob_fn`，将 top-k token 及分数追加到当前序列。
- **Step 2（束裁剪）**：对 `candidates` 按累计分数降序排序，仅保留固定数量。
- **Step 3（终止选择）**：合并已遇到 `eos_token` 的序列与剩余 beam，返回最高分者。

### 重点与易错点分析
- **注意点 1（分数域）**：对数概率应相加，不能对概率直接相加。
- **注意点 2（结束序列）**：已结束候选不能继续扩展，否则会在 EOS 后生成额外 token。
- **注意点 3（长度偏置）**：本实现没有长度归一化，累计分数可能偏好较短序列，这是常见面试扩展点。

### 复杂度分析
- **时间复杂度**：$O(TKV+TK^2\log K)$，主要是每轮为 $K$ 条 beam 计算词表分数。
- **空间复杂度**：$O(TK^2)$，保存当前扩展候选及其序列副本。""",
}

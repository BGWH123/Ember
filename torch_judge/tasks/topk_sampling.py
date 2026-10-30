"""Top-k / Top-p Sampling task."""

TASK = {
    'title': 'Top-k / Top-p Sampling',
    'title_zh': 'Top-k / Top-p 采样',
    'difficulty': 'Medium',
    'category': '采样与解码',
    'function_name': 'sample_top_k_top_p',
    'starter_code': """def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    pass""",
    'hint': """1. logits /= temperature
2. top-k: set logits below k-th largest to -inf
3. top-p: sort desc → cumsum of softmax probs → mask where cumsum > p → set to -inf
4. sample from softmax(logits)""",
    'hint_zh': """1. logits /= temperature
2. top-k：将低于第 k 大值的 logits 设为 -inf
3. top-p：降序排列 → softmax 概率 cumsum → 遮蔽 cumsum > p 的部分 → 设为 -inf
4. 从 softmax(logits) 中采样""",
    'description_en': """Implement top-k / top-p (nucleus) sampling for language model decoding.

These sampling strategies filter the vocabulary to high-probability tokens before sampling, balancing diversity and quality in text generation.

**Signature:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`

**Parameters:**
- `logits` — raw logits over vocabulary (V,)
- `top_k` — keep only top-k tokens (0 = disabled)
- `top_p` — keep tokens with cumulative prob <= p (1.0 = disabled)
- `temperature` — temperature scaling

**Returns:** sampled token index (int)

**Constraints:**
- Apply temperature first, then top-k, then top-p
- `top_k=1` must always return argmax""",
    'description_zh': """实现语言模型解码的 top-k / top-p（核）采样。

这些采样策略在采样前将词表过滤为高概率 token，在文本生成中平衡多样性和质量。

**签名:** `sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int`

**参数:**
- `logits` — 词表上的原始 logits (V,)
- `top_k` — 仅保留 top-k 个 token（0 = 禁用）
- `top_p` — 保留累积概率 <= p 的 token（1.0 = 禁用）
- `temperature` — 温度缩放

**返回:** 采样的 token 索引（整数）

**约束:**
- 先应用温度，再 top-k，再 top-p
- `top_k=1` 必须始终返回 argmax""",
    "tests": [
        {
            "name": 'top_k=1 always returns argmax',
            "code": """import torch
torch.manual_seed(0)
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
for _ in range(10):
    assert {fn}(logits.clone(), top_k=1) == 1, 'top_k=1 should return argmax'""",
        },
        {
            "name": 'Low temperature concentrates',
            "code": """import torch
torch.manual_seed(42)
logits = torch.tensor([1.0, 3.0, 2.0])
counts = [0, 0, 0]
for _ in range(100):
    counts[{fn}(logits.clone(), temperature=0.01)] += 1
assert counts[1] > 90, f'Low temp should pick argmax, got {counts}'""",
        },
        {
            "name": 'All tokens reachable (no filtering)',
            "code": """import torch
logits = torch.zeros(5)
seen = set()
for i in range(200):
    torch.manual_seed(i)
    seen.add({fn}(logits.clone()))
assert len(seen) == 5, f'Only saw {seen}'""",
        },
        {
            "name": 'Returns valid index',
            "code": """import torch
torch.manual_seed(0)
V = 100
logits = torch.randn(V)
for _ in range(20):
    t = {fn}(logits.clone(), top_k=10, top_p=0.9)
    assert 0 <= t < V, f'Token {t} out of range'""",
        },
        {
            "name": 'top_p excludes low-probability tokens',
            "code": """import torch
# Token 0 has prob ~1.0 after softmax; top_p=0.5 should only allow token 0
logits = torch.zeros(10)
logits[0] = 10.0
torch.manual_seed(0)
results = set()
for _ in range(50):
    results.add({fn}(logits.clone(), top_k=0, top_p=0.5, temperature=1.0))
expected = {0}
assert results == expected, f'With top_p=0.5 and dominant token 0, only token 0 should be sampled, got ' + str(results)""",
        },
    ],
    'solution': '''def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    """实现Top-k / Top-p 采样（Top-k / Top-p Sampling）。"""
    # Step 1: 用温度缩放词表 logits，并以第 k 大值截断 top-k 候选集
    logits = logits / max(temperature, 1e-08)
    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')
    # Step 2: 按概率降序构造累计质量不小于 top_p 的最小 nucleus
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum - probs > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    # Step 3: 重新归一化过滤后的分布并随机抽取一个原词表索引
    return torch.multinomial(probs, 1).item()''',
    'demo': """torch.manual_seed(42)
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))""",
    'explanation': r"""### 核心思路
采样分布先由 $p_i\propto\exp(z_i/T)$ 定义；top-k 只保留最高的 $k$ 个 logit，top-p 则保留降序概率前缀中累计质量首次达到 $p$ 所需的最小 token 集。被过滤位置设为 $-\infty$，最终 softmax 会精确赋予其零概率。

### 代码步骤拆解
- **Step 1（温度与 top-k 截断）**：`logits/temperature` 控制分布尖锐度，并以 `logits.topk(top_k).values[-1]` 为阈值屏蔽较小项。
- **Step 2（nucleus 前缀筛选）**：对剩余 logits 降序 softmax、累加概率；`cumsum-probs>top_p` 只删除当前 token 之前的质量已超过阈值的尾部，再散射回原索引。
- **Step 3（重归一化随机采样）**：过滤后的 logits 再做 softmax，`torch.multinomial` 返回一个合法词表下标。

### 重点与易错点分析
- **注意点 1（越阈 token 必须保留）**：直接使用 `cumsum>top_p` 会删掉使累计质量首次越过阈值的 token，甚至在很小 `top_p` 下清空集合；减去当前 `probs` 可避免此错。
- **注意点 2（操作顺序）**：温度、top-k、top-p 的顺序会改变 nucleus 概率；本题要求先缩放，再在 top-k 子集上计算 top-p。
- **注意点 3（确定性特例）**：`top_k=1` 后仅 argmax 有有限 logit，因此采样必然返回 argmax；一般随机结果则由 PyTorch RNG 状态决定。

### 复杂度分析
- **时间复杂度**：词表大小为 $V$ 时，top-p 的全排序主导为 $O(V\log V)$；只启用 top-k 时 `topk` 通常为 $O(V\log k)$。
- **空间复杂度**：排序值、索引、概率和累计概率均为长度 $V$，故额外空间为 $O(V)$。""",
}

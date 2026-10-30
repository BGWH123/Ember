"""Min-P & Repetition Penalty Sampling Filter task."""

TASK = {
    'title': 'Min-P & Repetition Penalty Sampling Filter',
    'title_zh': 'Min-P 采样与重复惩罚过滤',
    'difficulty': 'Medium',
    'category': '采样与解码',
    'function_name': 'min_p_sampling_probs',
    'starter_code': """def min_p_sampling_probs(logits, temperature=1.0, min_p=0.1, seen_tokens=None, repetition_penalty=1.0):
    pass""",
    'hint': """Scatter seen token IDs into a boolean `(B,V)` mask, apply the sign-aware penalty with `torch.where`, softmax temperature-scaled logits, then threshold relative to each row's maximum probability.""",
    'hint_zh': '把已见 token ID scatter 到 `(B,V)` 布尔掩码，用 `torch.where` 应用符号相关惩罚；温度缩放后 softmax，再按每行最大概率的相对阈值过滤。',
    'description_en': 'Return normalized sampling probabilities after a sign-aware repetition penalty and min-p filtering.',
    'description_zh': '在应用符号相关的重复惩罚与 Min-P 过滤后，返回归一化采样概率。',
    'theory_en': """Min-p adapts its cutoff to model confidence by retaining tokens whose probability is at least a fraction of the most likely token.""",
    'theory_zh': 'Min-P 通过保留概率至少为最高概率一定比例的 token，使截断阈值随模型置信度自适应。',
    'diagram_en': """```text
logits -> repetition penalty -> / temperature -> softmax -> p >= min_p*p_max -> renormalize
```""",
    'diagram_zh': """```text
logits -> 重复惩罚 -> / 温度 -> softmax -> p >= min_p*p_max -> 重归一化
```""",
    "tests": [
        {
            "name": 'probability shape and normalization',
            "code": """import torch
logits = torch.randn(4, 20)
probs = {fn}(logits, temperature=0.8, min_p=0.15)
assert probs.shape == logits.shape
assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
assert (probs >= 0).all()""",
        },
        {
            "name": 'exact penalty and filtering reference',
            "code": """import torch
logits = torch.tensor([[2., 1., -1., 0.5], [-2., 3., 1., -0.5]])
seen = torch.tensor([[0, 2], [0, 3]])
actual = {fn}(logits, temperature=0.7, min_p=0.2, seen_tokens=seen, repetition_penalty=2.0)
adjusted = logits.clone()
adjusted[0, 0] /= 2.0; adjusted[0, 2] *= 2.0
adjusted[1, 0] *= 2.0; adjusted[1, 3] *= 2.0
ref = torch.softmax(adjusted / 0.7, dim=-1)
ref = torch.where(ref >= 0.2 * ref.max(-1, keepdim=True).values, ref, torch.zeros_like(ref))
ref = ref / ref.sum(-1, keepdim=True)
assert torch.allclose(actual, ref, atol=1e-7)""",
        },
        {
            "name": 'duplicate seen tokens and min-p edge',
            "code": """import torch
logits = torch.tensor([[3., 2., 1.]])
seen = torch.tensor([[0, 0, 0]])
probs = {fn}(logits, min_p=1.0, seen_tokens=seen, repetition_penalty=2.0)
adjusted = logits.clone(); adjusted[:, 0] /= 2.0
winner = torch.softmax(adjusted, -1).argmax(-1)
assert torch.equal(probs.argmax(-1), winner)
assert torch.count_nonzero(probs).item() == 1 and probs.max().item() == 1.0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
logits = torch.tensor([[1.5, 1.0, 0.5], [0.3, 0.2, 0.1]], requires_grad=True)
seen = torch.tensor([[0], [2]])
probs = {fn}(logits, min_p=0.0, seen_tokens=seen, repetition_penalty=1.2)
probs[:, 0].sum().backward()
assert logits.grad is not None and logits.grad.shape == logits.shape
assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0""",
        },
    ],
    'solution': '''def min_p_sampling_probs(logits, temperature=1.0, min_p=0.1, seen_tokens=None, repetition_penalty=1.0):
    """计算重复惩罚后的 Min-P 采样分布 (Min-P Sampling)。"""
    # Step 1: 对历史 token 应用符号相关的重复惩罚
    adjusted_logits = logits.clone()
    if seen_tokens is not None and repetition_penalty != 1.0:
        seen_mask = torch.zeros_like(adjusted_logits, dtype=torch.bool)
        seen_mask.scatter_(1, seen_tokens, True)
        penalized = torch.where(
            adjusted_logits > 0,
            adjusted_logits / repetition_penalty,
            adjusted_logits * repetition_penalty,
        )
        adjusted_logits = torch.where(seen_mask, penalized, adjusted_logits)

    # Step 2: 温度缩放并转换为概率
    probs = torch.softmax(adjusted_logits / temperature, dim=-1)

    # Step 3: 按相对最大概率过滤并重归一化
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    filtered = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    return filtered / filtered.sum(dim=-1, keepdim=True)''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(2, 20)
seen_tokens = torch.tensor([[1, 3, 5], [2, 4, 6]])
probs = min_p_sampling_probs(logits, min_p=0.1, seen_tokens=seen_tokens, repetition_penalty=1.2)
print("Output shape:", probs.shape)
print("Kept tokens:", (probs > 0).sum(dim=-1))
print("Probability sums:", probs.sum(dim=-1))""",
    'explanation': r"""### 核心思路
先对已出现 token 做符号相关惩罚，再以温度得到 softmax 概率；Min-P 保留 $p_i\ge min\_p\cdot p_{max}$ 的 token，并把剩余质量重新归一化。

### 代码步骤拆解
- **Step 1（重复惩罚）**：scatter 构造词表 mask，正 logit 除系数、负 logit 乘系数。
- **Step 2（温度概率）**：计算 `softmax(adjusted_logits / temperature)`。
- **Step 3（相对截断）**：以每行最大概率生成阈值，过滤后归一化。

### 重点与易错点分析
- **注意点 1（重复 ID）**：布尔 mask 让同一历史 token 即使重复出现也只惩罚一次。
- **注意点 2（符号规则）**：负 logit 乘大于 1 的系数会更负；统一做除法会反而提高其概率。
- **注意点 3（非空集合）**：最大概率总满足 `min_p <= 1` 时的阈值，因此至少保留一个 token。

### 复杂度分析
- **时间复杂度**：$O(BV+BT)$，softmax 与词表过滤占主导。
- **空间复杂度**：$O(BV)$，保存 mask、调整后 logit 与概率。""",
}

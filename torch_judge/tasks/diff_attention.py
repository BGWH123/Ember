"""Differential Attention task."""

TASK = {
    'title': 'Differential Attention',
    'title_zh': '差分注意力',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'diff_attention',
    'starter_code': """def diff_attention(Q, K, V, lambda_val):
    pass""",
    'hint': """1. `Q1,Q2 = Q[...,:D_h], Q[...,D_h:]`; same for K
2. `scale = D_h**-0.5`
3. `A1 = softmax(Q1@K1.T * scale)`, `A2 = softmax(Q2@K2.T * scale)`
4. `return (A1 - lambda_val*A2) @ V`""",
    'hint_zh': """1. `Q1,Q2 = Q[...,:D_h], Q[...,D_h:]`；K 同理
2. `scale = D_h**-0.5`
3. `A1 = softmax(Q1@K1.T * scale)`，`A2 = softmax(Q2@K2.T * scale)`
4. `return (A1 - lambda_val*A2) @ V`""",
    'description_en': """Implement Differential Attention from the ICLR 2025 paper "Differential Transformer".

The key idea: split Q and K each into two halves, compute two separate softmax attention maps, then subtract them (scaled by a learnable lambda) to cancel noise and improve focus on relevant context.

**Signature:** `diff_attention(Q, K, V, lambda_val) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, 2*D_h)
- `K` — key tensor (B, S, 2*D_h)
- `V` — value tensor (B, S, D_v)
- `lambda_val` — scalar float or 0-dim tensor controlling noise cancellation

**Returns:** output tensor (B, S, D_v)

**Formula:**
```
out = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V
```

**Constraints:**
- Split Q into Q1, Q2 along last dim; same for K
- Use `torch.softmax(..., dim=-1)` (not F.softmax)
- Scale by 1/√D_h before softmax""",
    'description_zh': """实现 ICLR 2025 论文《Differential Transformer》中的差分注意力机制。

核心思想：将 Q 和 K 各自分成两半，分别计算两个 softmax 注意力图，然后相减（乘以可学习的 lambda）以消除噪声，提升对相关上下文的聚焦能力。

**签名:** `diff_attention(Q, K, V, lambda_val) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, 2*D_h)
- `K` — 键张量 (B, S, 2*D_h)
- `V` — 值张量 (B, S, D_v)
- `lambda_val` — 标量浮点数或 0 维张量，控制噪声消除强度

**返回:** 输出张量 (B, S, D_v)

**公式:**
```
out = (softmax(Q1 @ K1.T / √D_h) - lambda_val * softmax(Q2 @ K2.T / √D_h)) @ V
```

**约束:**
- 沿最后一维将 Q 拆分为 Q1、Q2；K 同理
- 使用 `torch.softmax(..., dim=-1)`（不能用 F.softmax）
- softmax 前除以 √D_h""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, S, D_h, D_v = 2, 6, 16, 32
Q = torch.randn(B, S, 2*D_h)
K = torch.randn(B, S, 2*D_h)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 0.5)
assert out.shape == (B, S, D_v), f'Expected ({B}, {S}, {D_v}), got {out.shape}'""",
        },
        {
            "name": 'lambda=0 reduces to standard attention',
            "code": """import torch
torch.manual_seed(1)
B, S, D_h, D_v = 1, 4, 8, 16
Q = torch.randn(B, S, 2*D_h)
K = torch.randn(B, S, 2*D_h)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 0.0)
# Manually compute standard attention with Q1, K1
Q1 = Q[..., :D_h]
K1 = K[..., :D_h]
scale = D_h ** -0.5
A = torch.softmax(Q1 @ K1.transpose(-2, -1) * scale, dim=-1)
expected = A @ V
assert torch.allclose(out, expected, atol=1e-5), 'lambda=0 should match standard attention on Q1/K1'""",
        },
        {
            "name": 'lambda=1 with identical halves gives zero',
            "code": """import torch
torch.manual_seed(2)
B, S, D_h, D_v = 1, 5, 8, 12
# Make Q and K such that Q1==Q2 and K1==K2
half_Q = torch.randn(B, S, D_h)
half_K = torch.randn(B, S, D_h)
Q = torch.cat([half_Q, half_Q], dim=-1)
K = torch.cat([half_K, half_K], dim=-1)
V = torch.randn(B, S, D_v)
out = {fn}(Q, K, V, 1.0)
assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), 'Identical halves with lambda=1 should give zero output'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(3)
B, S, D_h, D_v = 1, 4, 8, 8
Q = torch.randn(B, S, 2*D_h, requires_grad=True)
K = torch.randn(B, S, 2*D_h, requires_grad=True)
V = torch.randn(B, S, D_v, requires_grad=True)
lam = torch.tensor(0.3, requires_grad=True)
out = {fn}(Q, K, V, lam)
out.sum().backward()
assert Q.grad is not None, 'Missing gradient for Q'
assert K.grad is not None, 'Missing gradient for K'
assert V.grad is not None, 'Missing gradient for V'
assert lam.grad is not None, 'Missing gradient for lambda_val'""",
        },
    ],
    'solution': '''def diff_attention(Q, K, V, lambda_val):
    """计算差分注意力 (Differential Attention)。"""
    feature_size = Q.shape[-1]
    d_h = feature_size // 2

    # Step 1: 沿特征维将查询和键各拆成两组
    query_1, query_2 = Q[..., :d_h], Q[..., d_h:]
    key_1, key_2 = K[..., :d_h], K[..., d_h:]

    # Step 2: 分别计算两张缩放 softmax 注意力图
    scale = d_h ** -0.5
    weights_1 = torch.softmax(query_1 @ key_1.transpose(-2, -1) * scale, dim=-1)
    weights_2 = torch.softmax(query_2 @ key_2.transpose(-2, -1) * scale, dim=-1)

    # Step 3: 差分消噪后聚合值向量
    diff_weights = weights_1 - lambda_val * weights_2
    out = diff_weights @ V
    return out''',
    'demo': """torch.manual_seed(42)
B, S, d_h, d_v = 2, 4, 8, 6
Q = torch.randn(B, S, 2 * d_h)
K = torch.randn(B, S, 2 * d_h)
V = torch.randn(B, S, d_v)
out = diff_attention(Q, K, V, 0.5)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0])""",
    'explanation': r"""### 核心思路
差分注意力计算两组 $A_i=\operatorname{softmax}(Q_iK_i^\top/\sqrt{d_h})$，再用 $(A_1-\lambda A_2)V$ 抵消共同噪声并保留更有区分度的上下文。

### 代码步骤拆解
- **Step 1（特征拆分）**：将 `Q`、`K` 的最后一维等分，得到四个 `(B, S, d_h)` 张量。
- **Step 2（双注意力图）**：两组分数都形成 `(B, S, S)`，分别沿键位置执行 softmax。
- **Step 3（差分聚合）**：`weights_1 - lambda_val * weights_2` 后乘 `(B, S, d_v)` 的 `V`。

### 重点与易错点分析
- **注意点 1（维度要求）**：查询和键的最后一维必须为偶数，且两者拆分后的 `d_h` 一致。
- **注意点 2（缩放顺序）**：缩放应发生在 softmax 前；后缩放不能抑制 logits 方差导致的饱和。
- **注意点 3（非概率权重）**：差分后的行和为 $1-\lambda$，且元素可为负，不能再把它当作普通概率分布。

### 复杂度分析
- **时间复杂度**：$O(BS^2(d_h+d_v))$，两张注意力图和价值聚合是主要瓶颈。
- **空间复杂度**：$O(BS^2)$，需同时保存两张注意力矩阵。""",
}

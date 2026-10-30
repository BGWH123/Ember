"""ALiBi Attention task."""

TASK = {
    'title': 'ALiBi Attention',
    'title_zh': 'ALiBi 注意力',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'alibi_attention',
    'starter_code': """def alibi_attention(Q, K, V, num_heads):
    pass""",
    'hint': """1. slopes: `m_h = 2**(-8*h/H)` for h=1..H
2. distance matrix: `|i-j|`, shape `(S,S)`
3. bias: `-slopes[:, None, None] * distances`, shape `(H,S,S)`
4. standard multi-head attention with bias added to scores""",
    'hint_zh': """1. 斜率：`m_h = 2**(-8*h/H)`，h=1..H
2. 距离矩阵：`|i-j|`，形状 `(S,S)`
3. 偏置：`-slopes[:, None, None] * distances`，形状 `(H,S,S)`
4. 标准多头注意力，将偏置加到 scores 上""",
    'description_en': """Implement Attention with Linear Biases (ALiBi).

ALiBi replaces positional embeddings with a fixed linear bias added to attention scores. Each head gets a different slope `m_h`, penalizing attention to distant tokens.

**Signature:** `alibi_attention(Q, K, V, num_heads) -> Tensor`

**Parameters:**
- `Q, K, V` — tensors of shape `(B, S, D)`
- `num_heads` — number of attention heads

**Returns:** output tensor `(B, S, D)`

**Slope schedule:** `m_h = 1 / 2^(8h/H)` for h = 1..H

**Bias:** `bias[h, i, j] = -m_h * |i - j|` (added to attention scores before softmax)""",
    'description_zh': """实现带线性偏置的注意力（ALiBi）。

ALiBi 用固定线性偏置替代位置嵌入，直接加到注意力分数上。每个头有不同的斜率 `m_h`，对远距离 token 的注意力施加惩罚。

**签名:** `alibi_attention(Q, K, V, num_heads) -> Tensor`

**参数:**
- `Q, K, V` — 形状 `(B, S, D)` 的张量
- `num_heads` — 注意力头数

**返回:** 输出张量 `(B, S, D)`

**斜率:** `m_h = 1 / 2^(8h/H)`，h = 1..H

**偏置:** `bias[h, i, j] = -m_h * |i - j|`（在 softmax 前加到注意力分数上）""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, S, D, H = 2, 8, 16, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, H)
assert out.shape == (B, S, D), f'Expected ({B},{S},{D}), got {out.shape}'""",
        },
        {
            "name": 'Slopes follow geometric schedule',
            "code": """import torch
# With zero Q,K (uniform attention), the bias should break symmetry
# Verify by checking that output differs from standard attention
torch.manual_seed(42)
B, S, D, H = 1, 6, 8, 2
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_alibi = {fn}(Q, K, V, H)
# Standard attention (no bias) — compute manually
d_h = D // H
Qh = Q.view(B, S, H, d_h).transpose(1, 2)
Kh = K.view(B, S, H, d_h).transpose(1, 2)
Vh = V.view(B, S, H, d_h).transpose(1, 2)
scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5)
attn = torch.softmax(scores, dim=-1)
out_std = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
assert not torch.allclose(out_alibi, out_std, atol=1e-3), 'ALiBi output should differ from standard attention'""",
        },
        {
            "name": 'Gradient flows',
            "code": """import torch
B, S, D, H = 1, 4, 8, 2
Q = torch.randn(B, S, D, requires_grad=True)
K = torch.randn(B, S, D, requires_grad=True)
V = torch.randn(B, S, D, requires_grad=True)
out = {fn}(Q, K, V, H)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None""",
        },
        {
            "name": 'ALiBi slopes and bias numerical correctness',
            "code": """import torch
torch.manual_seed(11)
B, S, D, H = 1, 5, 8, 4
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V, H)
# Reference: slopes m_h = 1/2^(8h/H) for h=1..H
d_h = D // H
h_idx = torch.arange(1, H + 1, dtype=torch.float32)
slopes = 1.0 / (2.0 ** (8.0 * h_idx / H))          # (H,)
pos = torch.arange(S).float()
dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()   # (S, S)
bias = -slopes.view(H, 1, 1) * dist.unsqueeze(0)    # (1, H, S, S)
Qh = Q.view(B, S, H, d_h).transpose(1, 2)
Kh = K.view(B, S, H, d_h).transpose(1, 2)
Vh = V.view(B, S, H, d_h).transpose(1, 2)
scores = (Qh @ Kh.transpose(-2, -1)) / (d_h ** 0.5) + bias
attn = torch.softmax(scores, dim=-1)
expected = (attn @ Vh).transpose(1, 2).reshape(B, S, D)
assert torch.allclose(out, expected, atol=1e-5), f'ALiBi numerical mismatch: max diff {(out - expected).abs().max()}'""",
        },
    ],
    'solution': '''def alibi_attention(Q, K, V, num_heads):
    """线性偏置多头注意力 (ALiBi Attention)。"""
    B, S, D = Q.shape
    d_h = D // num_heads

    # Step 1: 构造各头斜率与位置距离偏置
    head_indices = torch.arange(1, num_heads + 1, device=Q.device, dtype=torch.float32)
    slopes = 1.0 / (2.0 ** (8.0 * head_indices / num_heads))
    positions = torch.arange(S, device=Q.device, dtype=torch.float32)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    bias = -slopes.view(num_heads, 1, 1) * distances.unsqueeze(0)

    # Step 2: 拆分注意力头并加入 ALiBi 偏置
    q_heads = Q.view(B, S, num_heads, d_h).transpose(1, 2)
    k_heads = K.view(B, S, num_heads, d_h).transpose(1, 2)
    v_heads = V.view(B, S, num_heads, d_h).transpose(1, 2)
    scores = q_heads @ k_heads.transpose(-2, -1) / (d_h ** 0.5)
    attn_weights = torch.softmax(scores + bias.unsqueeze(0), dim=-1)

    # Step 3: 聚合各头并恢复特征维
    context = attn_weights @ v_heads
    return context.transpose(1, 2).reshape(B, S, D)''',
    'demo': """torch.manual_seed(42)
B, S, D, H = 2, 6, 16, 4
Q, K, V = [torch.randn(B, S, D) for _ in range(3)]
out = alibi_attention(Q, K, V, H)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :3])""",
    'explanation': """### 核心思路
ALiBi 在缩放点积得分上加入按距离线性递减的头特定偏置 $-m_h|i-j|$，无需显式位置嵌入。

### 代码步骤拆解
- **Step 1（位置偏置）**：由 `slopes` 和 `(S, S)` 的 `distances` 构造 `(H, S, S)` 偏置。
- **Step 2（多头注意力）**：将 `Q/K/V` 变为 `(B, H, S, d_h)`，加偏置后做 softmax。
- **Step 3（合并头）**：`context` 转置并重排回 `(B, S, D)`。

### 重点与易错点分析
- **注意点 1（缩放维度）**：缩放应使用每头维度 `d_h`，不是总维度 `D`。
- **注意点 2（广播轴）**：`bias.unsqueeze(0)` 只广播批次维，不能混淆头维。
- **注意点 3（整除约束）**：`D` 必须能被 `num_heads` 整除，否则 `view` 无法成立。

### 复杂度分析
- **时间复杂度**：$O(BS^2D)$，主要成本是注意力矩阵乘法。
- **空间复杂度**：$O(BHS^2)$，主要保存得分和权重。""",
}

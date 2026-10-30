"""Causal Self-Attention task."""

TASK = {
    'title': 'Causal Self-Attention',
    'title_zh': '因果自注意力',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'causal_attention',
    'starter_code': """def causal_attention(Q, K, V):
    pass""",
    'hint': """Standard attention + causal mask. `torch.triu(ones, diagonal=1)` → upper triangle → fill with `-inf` before softmax.""",
    'hint_zh': '标准注意力 + 因果遮蔽。`torch.triu(ones, diagonal=1)` 得到上三角 → softmax 前填 `-inf`。',
    'description_en': """Implement causal (masked) self-attention.

Like standard attention but prevents each position from attending to future positions, essential for autoregressive models like GPT.

**Signature:** `causal_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S, D)
- `K` — key tensor (B, S, D)
- `V` — value tensor (B, S, D)

**Returns:** causally masked attention output (B, S, D)

**Constraints:**
- Mask future positions with `-inf` before softmax
- Position 0 should only see itself""",
    'description_zh': """实现因果（掩码）自注意力。

与标准注意力类似，但阻止每个位置关注未来位置，这对 GPT 等自回归模型至关重要。

**签名:** `causal_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S, D)
- `K` — 键张量 (B, S, D)
- `V` — 值张量 (B, S, D)

**返回:** 因果掩码注意力输出 (B, S, D)

**约束:**
- 在 softmax 之前用 `-inf` 掩盖未来位置
- 位置 0 只能看到自身""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
out = {fn}(torch.randn(2, 6, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16))
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": "Future tokens don't affect past",
            "code": """import torch
torch.manual_seed(0)
B, S, D = 1, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V)
K2, V2 = K.clone(), V.clone()
K2[:, 4:] = torch.randn(B, 4, D)
V2[:, 4:] = torch.randn(B, 4, D)
out2 = {fn}(Q, K2, V2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'""",
        },
        {
            "name": 'First position only sees itself',
            "code": """import torch
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V)
assert torch.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch
torch.manual_seed(42)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
# Reference computation
d_k = Q.shape[-1]
scale = d_k ** -0.5
scores = Q @ K.transpose(-2, -1) * scale
mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
attn = torch.softmax(scores, dim=-1)
expected = attn @ V
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff {(out - expected).abs().max()}'""",
        },
    ],
    'solution': '''def causal_attention(Q, K, V):
    """因果掩码自注意力 (Causal Self-Attention)。"""
    d_k = K.shape[-1]

    # Step 1: 计算缩放点积注意力得分
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
    S_q, S_k = scores.shape[-2:]

    # Step 2: 构造并应用禁止访问未来位置的上三角掩码
    mask = torch.triu(
        torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    # Step 3: 归一化并聚合值向量
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.bmm(attn_weights, V)''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 4, 8
Q, K, V = [torch.randn(B, S, D) for _ in range(3)]
out = causal_attention(Q, K, V)
print("Output shape:", out.shape)
print("First position:", out[0, 0, :3])
print("Matches first value:", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))""",
    'explanation': r"""### 核心思路
在缩放点积得分上把未来键位置设为 $-\infty$，使 softmax 后每个查询只能聚合当前位置及历史值。

### 代码步骤拆解
- **Step 1（缩放得分）**：`Q @ K^T` 得到 `(B, S_q, S_k)`，再除以 $\sqrt{d_k}$。
- **Step 2（因果屏蔽）**：`triu(diagonal=1)` 标记主对角线以上未来区域。
- **Step 3（上下文聚合）**：softmax 后未来权重为零，结果与 `V` 批量相乘。

### 重点与易错点分析
- **注意点 1（掩码顺序）**：必须在 softmax 前填充负无穷，不能在 softmax 后简单清零而不重归一化。
- **注意点 2（对角线保留）**：`diagonal=1` 允许 token 关注自身，避免首行全被屏蔽。
- **注意点 3（设备一致）**：掩码应直接创建在 `scores.device`，避免 CPU/GPU 混用。

### 复杂度分析
- **时间复杂度**：$O(BS^2D)$，主要是得分与值聚合矩阵乘法。
- **空间复杂度**：$O(BS^2)$，主要保存注意力得分和权重。""",
}

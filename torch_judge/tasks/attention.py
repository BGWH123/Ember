"""Softmax Attention task."""

TASK = {
    'title': 'Softmax Attention',
    'title_zh': 'Softmax 注意力',
    'difficulty': 'Easy',
    'category': '注意力机制',
    'function_name': 'scaled_dot_product_attention',
    'starter_code': """def scaled_dot_product_attention(Q, K, V):
    pass""",
    'hint': '`scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)` → `torch.bmm(softmax(scores, dim=-1), V)`.',
    'hint_zh': '`scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(d_k)` → `torch.bmm(softmax(scores, dim=-1), V)`。',
    'description_en': """Implement scaled dot-product attention.

This is the core attention mechanism: compute similarity scores between queries and keys, then use them to weight values.

**Signature:** `scaled_dot_product_attention(Q, K, V) -> Tensor`

**Parameters:**
- `Q` — query tensor (B, S_q, D)
- `K` — key tensor (B, S_k, D)
- `V` — value tensor (B, S_k, D_v)

**Returns:** weighted values tensor (B, S_q, D_v)

**Constraints:**
- Scale scores by `1/sqrt(d_k)`
- Support cross-attention (S_q != S_k)""",
    'description_zh': """实现缩放点积注意力。

这是核心注意力机制：计算查询和键之间的相似度分数，然后用它们对值进行加权。

**签名:** `scaled_dot_product_attention(Q, K, V) -> Tensor`

**参数:**
- `Q` — 查询张量 (B, S_q, D)
- `K` — 键张量 (B, S_k, D)
- `V` — 值张量 (B, S_k, D_v)

**返回:** 加权后的值张量 (B, S_q, D_v)

**约束:**
- 分数需除以 `sqrt(d_k)` 进行缩放
- 支持交叉注意力（S_q != S_k）""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
assert out.shape == (B, S, D), f'Shape mismatch: {out.shape} vs {(B, S, D)}'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch, math
torch.manual_seed(42)
B, S, D = 2, 4, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out = {fn}(Q, K, V)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
weights = torch.softmax(scores, dim=-1)
ref = torch.bmm(weights, V)
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch vs reference'""",
        },
        {
            "name": 'Gradient check',
            "code": """import torch, math
torch.manual_seed(42)
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None, 'Q.grad is None'
assert K.grad is not None, 'K.grad is None'
assert V.grad is not None, 'V.grad is None'""",
        },
        {
            "name": 'Cross-attention (seq_q != seq_k)',
            "code": """import torch
Q = torch.randn(1, 3, 16)
K = torch.randn(1, 5, 16)
V = torch.randn(1, 5, 32)
out = {fn}(Q, K, V)
assert out.shape == (1, 3, 32), f'Cross-attention shape: {out.shape}'""",
        },
    ],
    'solution': '''def scaled_dot_product_attention(Q, K, V):
    """缩放点积注意力 (Scaled Dot-Product Attention)。"""
    d_k = K.shape[-1]

    # Step 1: 计算查询与键的缩放相似度
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)

    # Step 2: 沿键序列维归一化注意力权重
    attn_weights = torch.softmax(scores, dim=-1)

    # Step 3: 对值向量加权求和
    return torch.bmm(attn_weights, V)''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 4, 8
Q, K, V = [torch.randn(B, S, D) for _ in range(3)]
out = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :3])""",
    'explanation': r"""### 核心思路
通过 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ 将查询与键的相似度转化为对值向量的加权组合。

### 代码步骤拆解
- **Step 1（相似度计算）**：`Q` 与转置后的 `K` 批量相乘，得到 `(B, S_q, S_k)`。
- **Step 2（概率归一化）**：沿键维对 `scores` 做 softmax，得到每个查询的权重分布。
- **Step 3（上下文聚合）**：权重与 `V` 相乘，输出 `(B, S_q, D_v)`。

### 重点与易错点分析
- **注意点 1（缩放稳定性）**：除以 $\sqrt{d_k}$ 可避免大维度点积使 softmax 饱和。
- **注意点 2（转置维度）**：只交换 `K` 的最后两个维度，必须保留批次维。
- **注意点 3（交叉注意力）**：`S_q` 可不同于 `S_k`，但 `K` 与 `V` 的序列长度必须一致。

### 复杂度分析
- **时间复杂度**：$O(BS_qS_k(D_k+D_v))$，瓶颈是两次批量矩阵乘法。
- **空间复杂度**：$O(BS_qS_k)$，主要保存注意力得分与权重。""",
}

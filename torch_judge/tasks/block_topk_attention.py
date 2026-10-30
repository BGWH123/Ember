"""Block Top-K Sparse Attention task."""

TASK = {
    'title': 'Block Top-K Sparse Attention',
    'title_zh': '分块 Top-K 稀疏注意力',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'block_topk_attention',
    'starter_code': """def block_topk_attention(q, k, v, block_size, top_k):
    pass""",
    'hint': """Score mean-pooled key blocks, scatter the top-k indices into a boolean block mask, then repeat each block decision across its tokens.""",
    'hint_zh': '对均值池化的键块打分，将 top-k 索引散射成布尔块掩码，再把每个块选择重复到块内 token。',
    'description_en': """Implement sparse attention that routes each query to the tokens in its top-scoring mean-pooled key blocks.""",
    'description_zh': '实现分块稀疏注意力：每个查询仅关注均值池化得分最高的若干键块中的 token。',
    'theory_en': """Coarse block scores cheaply choose a subset of keys. Exact token-level attention is then computed only over selected blocks, reducing the effective attention support.""",
    'theory_zh': '粗粒度块分数以较低成本筛选键子集，随后仅在选中块内计算精确 token 级注意力，缩小注意力范围。',
    'diagram_en': """```text
k -> block means -> block scores -> top-k mask -> token scores -> softmax -> @v
```""",
    'diagram_zh': """```text
k -> 分块均值 -> 块分数 -> top-k 掩码 -> token 分数 -> softmax -> @v
```""",
    "tests": [
        {
            "name": 'output shape',
            "code": """import torch
q, k, v = [torch.randn(2, 8, 4) for _ in range(3)]
out = {fn}(q, k, v, block_size=2, top_k=2)
assert out.shape == (2, 8, 4)""",
        },
        {
            "name": 'numerical sparse reference',
            "code": """import torch
torch.manual_seed(3)
q, k, v = [torch.randn(1, 6, 3) for _ in range(3)]
out = {fn}(q, k, v, 2, 1)
kb = k.view(1, 3, 2, 3).mean(2)
idx = (q @ kb.transpose(-1, -2) / (3 ** 0.5)).topk(1, dim=-1).indices
bm = torch.zeros(1, 6, 3, dtype=torch.bool).scatter(-1, idx, True)
mask = bm.repeat_interleave(2, dim=-1)
scores = (q @ k.transpose(-1, -2) / (3 ** 0.5)).masked_fill(~mask, float("-inf"))
ref = torch.softmax(scores, -1) @ v
assert torch.allclose(out, ref, atol=1e-6)""",
        },
        {
            "name": 'all blocks equals dense attention',
            "code": """import torch
torch.manual_seed(4)
q, k, v = [torch.randn(2, 8, 5) for _ in range(3)]
out = {fn}(q, k, v, block_size=2, top_k=4)
ref = torch.softmax(q @ k.transpose(-1, -2) / (5 ** 0.5), -1) @ v
assert torch.allclose(out, ref, atol=1e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q = torch.randn(1, 6, 4, requires_grad=True)
k = torch.randn(1, 6, 4, requires_grad=True)
v = torch.randn(1, 6, 4, requires_grad=True)
{fn}(q, k, v, 3, 1).square().sum().backward()
assert q.grad is not None and k.grad is not None and v.grad is not None""",
        },
    ],
    'solution': '''def block_topk_attention(q, k, v, block_size, top_k):
    """分块 Top-K 稀疏注意力 (Block Top-K Attention)。"""
    B, S, D = q.shape
    num_blocks = S // block_size

    # Step 1: 均值池化键块并为每个查询选择 Top-K 块
    k_blocks = k.view(B, num_blocks, block_size, D).mean(dim=2)
    block_scores = q @ k_blocks.transpose(-1, -2) / (D ** 0.5)
    selected_blocks = block_scores.topk(top_k, dim=-1).indices

    # Step 2: 将块级选择扩展为 token 级布尔掩码
    block_mask = torch.zeros(
        B, S, num_blocks, device=q.device, dtype=torch.bool
    )
    block_mask.scatter_(-1, selected_blocks, True)
    token_mask = block_mask.repeat_interleave(block_size, dim=-1)

    # Step 3: 在选中 token 上执行精确注意力
    scores = q @ k.transpose(-1, -2) / (D ** 0.5)
    scores = scores.masked_fill(~token_mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 32, 16
q, k, v = [torch.randn(B, S, D) for _ in range(3)]
out = block_topk_attention(q, k, v, block_size=8, top_k=2)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :3])""",
    'explanation': r"""### 核心思路
先用块均值近似键表示，为每个查询路由到 Top-K 块，再仅让这些块内 token 参与精确 softmax 注意力。

### 代码步骤拆解
- **Step 1（粗粒度路由）**：`k_blocks` 为 `(B, num_blocks, D)`，`selected_blocks` 为每个查询的块编号。
- **Step 2（掩码展开）**：块级 `(B, S, num_blocks)` 决策重复为 `(B, S, S)` token 掩码。
- **Step 3（精确聚合）**：未选 logits 置为负无穷，归一化后乘 `v`。

### 重点与易错点分析
- **注意点 1（整块约束）**：当前实现要求 `S` 可被 `block_size` 整除。
- **注意点 2（离散路由梯度）**：`topk` 索引不可微，但选中路径上的 `q/k/v` 仍有梯度。
- **注意点 3（稀疏实现）**：本教学实现仍构造密集 token 得分，真实加速需使用稀疏内核。

### 复杂度分析
- **时间复杂度**：当前密集实现为 $O(BS^2D)$，块路由额外为 $O(BS^2D/block\_size)$。
- **空间复杂度**：$O(BS^2)$，主要是 token 掩码和得分。""",
}

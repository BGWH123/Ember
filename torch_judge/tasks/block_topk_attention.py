"""Block Top-K sparse attention task."""

TASK = {
    "title": "Block Top-K Sparse Attention",
    "title_zh": "分块 Top-K 稀疏注意力",
    "difficulty": "Hard",
    "category": "注意力机制",
    "function_name": "block_topk_attention",
    "starter_code": '''def block_topk_attention(q, k, v, block_size, top_k):
    pass
''',
    "hint": "Score mean-pooled key blocks, scatter the top-k indices into a boolean block mask, then repeat each block decision across its tokens.",
    "hint_zh": "对均值池化的键块打分，将 top-k 索引散射成布尔块掩码，再把每个块选择重复到块内 token。",
    "description_en": "Implement sparse attention that routes each query to the tokens in its top-scoring mean-pooled key blocks.",
    "description_zh": "实现分块稀疏注意力：每个查询仅关注均值池化得分最高的若干键块中的 token。",
    "theory_en": "Coarse block scores cheaply choose a subset of keys. Exact token-level attention is then computed only over selected blocks, reducing the effective attention support.",
    "theory_zh": "粗粒度块分数以较低成本筛选键子集，随后仅在选中块内计算精确 token 级注意力，缩小注意力范围。",
    "diagram_en": "```text\nk -> block means -> block scores -> top-k mask -> token scores -> softmax -> @v\n```",
    "diagram_zh": "```text\nk -> 分块均值 -> 块分数 -> top-k 掩码 -> token 分数 -> softmax -> @v\n```",
    "tests": [
        {
            "name": "output shape",
            "code": '''import torch
q, k, v = [torch.randn(2, 8, 4) for _ in range(3)]
out = {fn}(q, k, v, block_size=2, top_k=2)
assert out.shape == (2, 8, 4)
''',
        },
        {
            "name": "numerical sparse reference",
            "code": '''import torch
torch.manual_seed(3)
q, k, v = [torch.randn(1, 6, 3) for _ in range(3)]
out = {fn}(q, k, v, 2, 1)
kb = k.view(1, 3, 2, 3).mean(2)
idx = (q @ kb.transpose(-1, -2) / (3 ** 0.5)).topk(1, dim=-1).indices
bm = torch.zeros(1, 6, 3, dtype=torch.bool).scatter(-1, idx, True)
mask = bm.repeat_interleave(2, dim=-1)
scores = (q @ k.transpose(-1, -2) / (3 ** 0.5)).masked_fill(~mask, float("-inf"))
ref = torch.softmax(scores, -1) @ v
assert torch.allclose(out, ref, atol=1e-6)
''',
        },
        {
            "name": "all blocks equals dense attention",
            "code": '''import torch
torch.manual_seed(4)
q, k, v = [torch.randn(2, 8, 5) for _ in range(3)]
out = {fn}(q, k, v, block_size=2, top_k=4)
ref = torch.softmax(q @ k.transpose(-1, -2) / (5 ** 0.5), -1) @ v
assert torch.allclose(out, ref, atol=1e-6)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q = torch.randn(1, 6, 4, requires_grad=True)
k = torch.randn(1, 6, 4, requires_grad=True)
v = torch.randn(1, 6, 4, requires_grad=True)
{fn}(q, k, v, 3, 1).square().sum().backward()
assert q.grad is not None and k.grad is not None and v.grad is not None
''',
        },
    ],
    "solution": '''import torch

def block_topk_attention(q, k, v, block_size, top_k):
    B, S, D = q.shape
    num_blocks = S // block_size
    k_blocks = k.view(B, num_blocks, block_size, D).mean(dim=2)
    block_scores = (q @ k_blocks.transpose(-1, -2)) / (D ** 0.5)
    selected = block_scores.topk(top_k, dim=-1).indices
    block_mask = torch.zeros(B, S, num_blocks, device=q.device, dtype=torch.bool)
    block_mask = block_mask.scatter(-1, selected, True)
    token_mask = block_mask.repeat_interleave(block_size, dim=-1)
    token_scores = (q @ k.transpose(-1, -2)) / (D ** 0.5)
    token_scores = token_scores.masked_fill(~token_mask, float("-inf"))
    return torch.softmax(token_scores, dim=-1) @ v
''',
    "demo": '''q, k, v = [torch.randn(2, 1024, 64) for _ in range(3)]
print(block_topk_attention(q, k, v, block_size=64, top_k=4).shape)
''',
    "explanation": "The repeated block mask has one decision for every query-token pair. Unselected logits become negative infinity, so their softmax probability is exactly zero.",
}

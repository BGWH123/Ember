"""QK-Norm scaled dot-product attention task."""

TASK = {
    "title": "QK-Norm Scaled Dot-Product Attention",
    "title_zh": "QK-Norm 注意力机制",
    "difficulty": "Medium",
    "category": "注意力机制",
    "function_name": "qk_norm_attention",
    "starter_code": '''def qk_norm_attention(q, k, v, gamma_q, gamma_k, eps=1e-6):
    pass
''',
    "hint": "RMS-normalize q and k over D, apply their scale vectors, then use ordinary scaled dot-product attention.",
    "hint_zh": "沿 D 维对 q 和 k 做 RMS 归一化并乘各自缩放向量，然后执行标准缩放点积注意力。",
    "description_en": "Implement multi-head scaled dot-product attention after applying independently scaled RMSNorm to queries and keys.",
    "description_zh": "先对查询和键应用带独立缩放参数的 RMSNorm，再实现多头缩放点积注意力。",
    "theory_en": "QK-Norm controls query/key magnitudes before their dot product. This can prevent attention logits from growing excessively while preserving learned per-feature scales.",
    "theory_zh": "QK-Norm 在点积前控制查询和键的幅值，可避免注意力 logits 过大，同时保留逐特征可学习缩放。",
    "diagram_en": "```text\nq -> RMSNorm(gamma_q) --\\\n                         QK^T/sqrt(D) -> softmax -> @v\nk -> RMSNorm(gamma_k) --/\n```",
    "diagram_zh": "```text\nq -> RMSNorm(gamma_q) --\\\n                         QK^T/sqrt(D) -> softmax -> @v\nk -> RMSNorm(gamma_k) --/\n```",
    "tests": [
        {
            "name": "output shape",
            "code": '''import torch
q = torch.randn(2, 3, 5, 4)
k = torch.randn(2, 3, 7, 4)
v = torch.randn(2, 3, 7, 6)
out = {fn}(q, k, v, torch.ones(4), torch.ones(4))
assert out.shape == (2, 3, 5, 6)
''',
        },
        {
            "name": "numerical reference",
            "code": '''import torch
torch.manual_seed(2)
q, k, v = [torch.randn(1, 2, 4, 8) for _ in range(3)]
gq, gk = torch.randn(8), torch.randn(8)
out = {fn}(q, k, v, gq, gk)
qn = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * gq
kn = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * gk
ref = torch.softmax(qn @ kn.transpose(-2, -1) / (8 ** 0.5), -1) @ v
assert torch.allclose(out, ref, atol=1e-6)
''',
        },
        {
            "name": "zero q edge case",
            "code": '''import torch
q = torch.zeros(1, 1, 2, 3)
k = torch.randn(1, 1, 4, 3)
v = torch.randn(1, 1, 4, 5)
out = {fn}(q, k, v, torch.ones(3), torch.ones(3))
ref = v.mean(dim=-2, keepdim=True).expand(1, 1, 2, 5)
assert torch.allclose(out, ref, atol=1e-6) and torch.isfinite(out).all()
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q = torch.randn(1, 2, 3, 4, requires_grad=True)
k = torch.randn(1, 2, 3, 4, requires_grad=True)
v = torch.randn(1, 2, 3, 4, requires_grad=True)
gq = torch.ones(4, requires_grad=True)
gk = torch.ones(4, requires_grad=True)
{fn}(q, k, v, gq, gk).square().sum().backward()
assert all(t.grad is not None for t in (q, k, v, gq, gk))
''',
        },
    ],
    "solution": '''import torch

def qk_norm_attention(q, k, v, gamma_q, gamma_k, eps=1e-6):
    D = q.size(-1)
    q_norm = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps) * gamma_q
    k_norm = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps) * gamma_k
    weights = torch.softmax((q_norm @ k_norm.transpose(-2, -1)) / (D ** 0.5), dim=-1)
    return weights @ v
''',
    "demo": '''q = torch.randn(2, 4, 16, 32)
k = torch.randn(2, 4, 16, 32)
v = torch.randn(2, 4, 16, 32)
print(qk_norm_attention(q, k, v, torch.ones(32), torch.ones(32)).shape)
''',
    "explanation": "RMSNorm is applied independently at every batch, head, and sequence position. Broadcasting applies each `(D,)` scale vector before attention.",
}

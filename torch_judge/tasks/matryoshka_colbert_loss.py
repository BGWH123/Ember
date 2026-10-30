"""ColBERT late-interaction MaxSim scoring task."""

TASK = {
    "title": "ColBERT Late-Interaction MaxSim & Matryoshka Scoring",
    "title_zh": "ColBERT 晚期交互 MaxSim 评分",
    "difficulty": "Medium",
    "category": "检索与智能体",
    "function_name": "colbert_maxsim",
    "starter_code": '''def colbert_maxsim(query_toks, doc_toks, query_mask=None, doc_mask=None, eps=1e-8):
    pass
''',
    "hint": "Normalize token vectors, mask invalid document columns before taking each query token's maximum, then mask and sum query-token maxima.",
    "hint_zh": "归一化 token 向量，在每个查询 token 取最大值前屏蔽无效文档列，再屏蔽并求和查询 token 的最大相似度。",
    "description_en": "Implement batched ColBERT late-interaction scoring with optional query and document padding masks.",
    "description_zh": "实现带可选查询和文档填充掩码的批量 ColBERT 晚期交互评分。",
    "theory_en": "MaxSim lets each query token independently select its closest document token. Summing these maxima preserves token-level matching while producing one retrieval score per pair.",
    "theory_zh": "MaxSim 允许每个查询 token 独立选择最相近的文档 token；对最大值求和既保留 token 级匹配，又为每个配对产生一个检索分数。",
    "diagram_en": "```text\nnormalized Q @ normalized D^T -> mask D -> max over D -> mask Q -> sum\n```",
    "diagram_zh": "```text\n归一化 Q @ 归一化 D^T -> 屏蔽 D -> 对 D 取最大 -> 屏蔽 Q -> 求和\n```",
    "tests": [
        {"name": "batch shape", "code": '''import torch
q = torch.randn(3, 4, 6)
d = torch.randn(3, 7, 6)
out = {fn}(q, d)
assert out.shape == (3,)
'''},
        {"name": "exact masked reference", "code": '''import torch
q = torch.tensor([[[1., 0.], [0., 1.]]])
d = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
qm = torch.tensor([[1, 1]])
dm = torch.tensor([[1, 0, 1]])
out = {fn}(q, d, qm, dm)
qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
sim = (qn @ dn.transpose(-1, -2)).masked_fill(dm[:, None, :] == 0, float("-inf"))
ref = (sim.max(-1).values * qm).sum(-1)
assert torch.equal(out, ref)
'''},
        {"name": "zero query mask edge case", "code": '''import torch
q = torch.randn(1, 2, 3)
d = torch.randn(1, 1, 3)
out = {fn}(q, d, query_mask=torch.zeros(1, 2))
assert out.shape == (1,) and torch.equal(out, torch.zeros(1))
'''},
        {"name": "gradient flow", "code": '''import torch
q = torch.randn(2, 3, 4, requires_grad=True)
d = torch.randn(2, 5, 4, requires_grad=True)
out = {fn}(q, d, doc_mask=torch.ones(2, 5, dtype=torch.bool))
out.sum().backward()
assert q.grad is not None and d.grad is not None and torch.isfinite(q.grad).all() and torch.isfinite(d.grad).all()
'''},
    ],
    "solution": '''import torch

def colbert_maxsim(query_toks, doc_toks, query_mask=None, doc_mask=None, eps=1e-8):
    q = query_toks / (query_toks.norm(dim=-1, keepdim=True) + eps)
    d = doc_toks / (doc_toks.norm(dim=-1, keepdim=True) + eps)
    sim = q @ d.transpose(-1, -2)
    if doc_mask is not None:
        sim = sim.masked_fill(doc_mask.unsqueeze(-2) == 0, float("-inf"))
    maxima = sim.max(dim=-1).values
    if query_mask is not None:
        maxima = maxima.masked_fill(query_mask == 0, 0.0)
    return maxima.sum(dim=-1)
''',
    "demo": '''queries = torch.randn(2, 16, 128)
documents = torch.randn(2, 64, 128)
print(colbert_maxsim(queries, documents))
''',
    "explanation": "Document padding must be removed before MaxSim so padded tokens cannot win. Query padding is removed after MaxSim so it contributes exactly zero.",
}

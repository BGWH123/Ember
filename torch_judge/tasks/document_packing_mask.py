"""Packed-document causal and loss mask task."""

TASK = {
    "title": "Document Packing & Intra-Document Causal Mask",
    "title_zh": "预训练文档拼接与文档内因果掩码",
    "difficulty": "Medium",
    "category": "高效训练",
    "function_name": "pack_documents_mask",
    "starter_code": '''def pack_documents_mask(doc_ids):
    pass
''',
    "hint": "Broadcast query IDs as `(B,L,1)` and key IDs as `(B,1,L)`, combine equality with a lower-triangular position comparison and valid-query test.",
    "hint_zh": "将查询文档 ID 广播为 `(B,L,1)`、键文档 ID 广播为 `(B,1,L)`，再与下三角位置比较和有效查询判断组合。",
    "description_en": "Build causal attention and next-token loss masks for multiple documents packed into each sequence, excluding padding and cross-document boundaries.",
    "description_zh": "为每条序列中拼接的多个文档构造因果注意力掩码和下一 token 损失掩码，排除 padding 与跨文档边界。",
    "theory_en": "Packing removes padding waste, but document identity must constrain attention and training targets so unrelated documents do not leak into one another.",
    "theory_zh": "文档拼接可减少 padding 浪费，但必须用文档身份约束注意力和训练目标，以避免无关文档间的信息泄漏。",
    "diagram_en": "```text\npacked IDs: [0 0 0 | 1 1 | pad]\nattention: causal blocks on the diagonal\nloss:      1 1 0   1 0\n```",
    "diagram_zh": "```text\n拼接 ID: [0 0 0 | 1 1 | padding]\n注意力:   对角线上的因果块\n损失:     1 1 0   1 0\n```",
    "tests": [
        {
            "name": "mask shapes and dtypes",
            "code": '''import torch
ids = torch.tensor([[0, 0, 1, 1, -1], [3, 3, 3, 4, 4]])
attn, loss = {fn}(ids)
assert attn.shape == (2, 5, 5) and loss.shape == (2, 4)
assert attn.dtype == torch.bool and loss.dtype == torch.bool
''',
        },
        {
            "name": "exact packed masks",
            "code": '''import torch
ids = torch.tensor([[0, 0, 1, 1, -1]])
attn, loss = {fn}(ids)
ref_attn = torch.tensor([[
    [1,0,0,0,0],
    [1,1,0,0,0],
    [0,0,1,0,0],
    [0,0,1,1,0],
    [0,0,0,0,0],
]], dtype=torch.bool)
ref_loss = torch.tensor([[1,0,1,0]], dtype=torch.bool)
assert torch.equal(attn, ref_attn)
assert torch.equal(loss, ref_loss)
''',
        },
        {
            "name": "all padding and length-one edges",
            "code": '''import torch
ids = torch.full((3, 1), -1)
attn, loss = {fn}(ids)
assert attn.shape == (3, 1, 1) and not attn.any()
assert loss.shape == (3, 0) and loss.numel() == 0
''',
        },
        {
            "name": "attention mask preserves gradient",
            "code": '''import torch
ids = torch.tensor([[0, 0, 1, 1]])
attn, loss = {fn}(ids)
scores = torch.randn(1, 4, 4, requires_grad=True)
scores.masked_fill(~attn, 0.0).sum().backward()
assert loss.tolist() == [[True, False, True]]
assert scores.grad is not None and torch.equal(scores.grad != 0, attn)
''',
        },
    ],
    "solution": '''import torch

def pack_documents_mask(doc_ids):
    B, L = doc_ids.shape
    positions = torch.arange(L, device=doc_ids.device)
    causal = positions.view(1, L, 1) >= positions.view(1, 1, L)
    query_ids = doc_ids.unsqueeze(-1)
    key_ids = doc_ids.unsqueeze(1)
    attn_mask = causal & (query_ids == key_ids) & (query_ids >= 0)
    loss_mask = (doc_ids[:, :-1] == doc_ids[:, 1:]) & (doc_ids[:, :-1] >= 0)
    return attn_mask, loss_mask
''',
    "demo": '''doc_ids = torch.tensor([[0, 0, 0, 1, 1, -1]])
attn_mask, loss_mask = pack_documents_mask(doc_ids)
print(attn_mask[0].int())
print(loss_mask)
''',
    "explanation": "Document equality creates independent square blocks, causality keeps only each block's lower triangle, and valid query IDs remove padding rows. The loss mask checks adjacent IDs.",
}

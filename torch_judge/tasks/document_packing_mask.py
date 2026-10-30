"""Document Packing & Intra-Document Causal Mask task."""

TASK = {
    'title': 'Document Packing & Intra-Document Causal Mask',
    'title_zh': '预训练文档拼接与文档内因果掩码',
    'difficulty': 'Medium',
    'category': '高效训练',
    'function_name': 'pack_documents_mask',
    'starter_code': """def pack_documents_mask(doc_ids):
    pass""",
    'hint': """Broadcast query IDs as `(B,L,1)` and key IDs as `(B,1,L)`, combine equality with a lower-triangular position comparison and valid-query test.""",
    'hint_zh': '将查询文档 ID 广播为 `(B,L,1)`、键文档 ID 广播为 `(B,1,L)`，再与下三角位置比较和有效查询判断组合。',
    'description_en': """Build causal attention and next-token loss masks for multiple documents packed into each sequence, excluding padding and cross-document boundaries.""",
    'description_zh': '为每条序列中拼接的多个文档构造因果注意力掩码和下一 token 损失掩码，排除 padding 与跨文档边界。',
    'theory_en': """Packing removes padding waste, but document identity must constrain attention and training targets so unrelated documents do not leak into one another.""",
    'theory_zh': '文档拼接可减少 padding 浪费，但必须用文档身份约束注意力和训练目标，以避免无关文档间的信息泄漏。',
    'diagram_en': """```text
packed IDs: [0 0 0 | 1 1 | pad]
attention: causal blocks on the diagonal
loss:      1 1 0   1 0
```""",
    'diagram_zh': """```text
拼接 ID: [0 0 0 | 1 1 | padding]
注意力:   对角线上的因果块
损失:     1 1 0   1 0
```""",
    "tests": [
        {
            "name": 'mask shapes and dtypes',
            "code": """import torch
ids = torch.tensor([[0, 0, 1, 1, -1], [3, 3, 3, 4, 4]])
attn, loss = {fn}(ids)
assert attn.shape == (2, 5, 5) and loss.shape == (2, 4)
assert attn.dtype == torch.bool and loss.dtype == torch.bool""",
        },
        {
            "name": 'exact packed masks',
            "code": """import torch
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
assert torch.equal(loss, ref_loss)""",
        },
        {
            "name": 'all padding and length-one edges',
            "code": """import torch
ids = torch.full((3, 1), -1)
attn, loss = {fn}(ids)
assert attn.shape == (3, 1, 1) and not attn.any()
assert loss.shape == (3, 0) and loss.numel() == 0""",
        },
        {
            "name": 'attention mask preserves gradient',
            "code": """import torch
ids = torch.tensor([[0, 0, 1, 1]])
attn, loss = {fn}(ids)
scores = torch.randn(1, 4, 4, requires_grad=True)
scores.masked_fill(~attn, 0.0).sum().backward()
assert loss.tolist() == [[True, False, True]]
assert scores.grad is not None and torch.equal(scores.grad != 0, attn)""",
        },
    ],
    'solution': '''def pack_documents_mask(doc_ids):
    """构造文档拼接的因果注意力与损失掩码 (Document Packing Mask)。"""
    B, S = doc_ids.shape

    # Step 1: 构造所有查询—键位置的下三角因果关系
    positions = torch.arange(S, device=doc_ids.device)
    causal_mask = positions.view(1, S, 1) >= positions.view(1, 1, S)

    # Step 2: 仅保留同文档且非 padding 的注意力连接
    query_ids = doc_ids.unsqueeze(-1)
    key_ids = doc_ids.unsqueeze(1)
    attn_mask = causal_mask & (query_ids == key_ids) & (query_ids >= 0)

    # Step 3: 相邻 token 同属有效文档时才计算下一 token 损失
    loss_mask = (doc_ids[:, :-1] == doc_ids[:, 1:]) & (doc_ids[:, :-1] >= 0)
    return attn_mask, loss_mask''',
    'demo': """torch.manual_seed(42)
doc_ids = torch.tensor([[0, 0, 0, 1, 1, -1]])
attn_mask, loss_mask = pack_documents_mask(doc_ids)
print("Attention mask shape:", attn_mask.shape)
print("Attention mask:", attn_mask[0].int())
print("Loss mask:", loss_mask)""",
    'explanation': """### 核心思路
文档拼接掩码同时要求查询位置不早于键位置、两者文档 ID 相同且查询有效；下一 token 损失仅在相邻位置属于同一有效文档时启用。

### 代码步骤拆解
- **Step 1（因果关系）**：广播位置索引生成 `(1, S, S)` 的下三角 `causal_mask`。
- **Step 2（文档隔离）**：`query_ids` 与 `key_ids` 广播比较，得到 `(B, S, S)` 的块对角因果掩码。
- **Step 3（损失边界）**：比较 `doc_ids[:, :-1]` 与后一位置，生成 `(B, S-1)` 的 `loss_mask`。

### 重点与易错点分析
- **注意点 1（padding 行）**：仅检查 ID 相等会让所有 `-1` padding 互相可见，因此必须额外要求 `query_ids >= 0`。
- **注意点 2（边界目标）**：文档末 token 不应预测下一文档首 token，故损失掩码必须检查相邻 ID。
- **注意点 3（短序列）**：当 $S=1$ 时切片自然产生 `(B, 0)` 空掩码，不需要特殊分支。

### 复杂度分析
- **时间复杂度**：$O(BS^2)$，构造完整注意力掩码是主要开销。
- **空间复杂度**：$O(BS^2)$，布尔注意力矩阵占据主要空间。""",
}

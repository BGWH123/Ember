"""ColBERT Late-Interaction MaxSim & Matryoshka Scoring task."""

TASK = {
    'title': 'ColBERT Late-Interaction MaxSim & Matryoshka Scoring',
    'title_zh': 'ColBERT 晚期交互 MaxSim 评分',
    'difficulty': 'Medium',
    'category': '检索与智能体',
    'function_name': 'colbert_maxsim',
    'starter_code': """def colbert_maxsim(query_toks, doc_toks, query_mask=None, doc_mask=None, eps=1e-8):
    pass""",
    'hint': """Normalize token vectors, mask invalid document columns before taking each query token's maximum, then mask and sum query-token maxima.""",
    'hint_zh': '归一化 token 向量，在每个查询 token 取最大值前屏蔽无效文档列，再屏蔽并求和查询 token 的最大相似度。',
    'description_en': 'Implement batched ColBERT late-interaction scoring with optional query and document padding masks.',
    'description_zh': '实现带可选查询和文档填充掩码的批量 ColBERT 晚期交互评分。',
    'theory_en': """MaxSim lets each query token independently select its closest document token. Summing these maxima preserves token-level matching while producing one retrieval score per pair.""",
    'theory_zh': 'MaxSim 允许每个查询 token 独立选择最相近的文档 token；对最大值求和既保留 token 级匹配，又为每个配对产生一个检索分数。',
    'diagram_en': """```text
normalized Q @ normalized D^T -> mask D -> max over D -> mask Q -> sum
```""",
    'diagram_zh': """```text
归一化 Q @ 归一化 D^T -> 屏蔽 D -> 对 D 取最大 -> 屏蔽 Q -> 求和
```""",
    "tests": [
        {
            "name": 'batch shape',
            "code": """import torch
q = torch.randn(3, 4, 6)
d = torch.randn(3, 7, 6)
out = {fn}(q, d)
assert out.shape == (3,)""",
        },
        {
            "name": 'exact masked reference',
            "code": """import torch
q = torch.tensor([[[1., 0.], [0., 1.]]])
d = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
qm = torch.tensor([[1, 1]])
dm = torch.tensor([[1, 0, 1]])
out = {fn}(q, d, qm, dm)
qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
sim = (qn @ dn.transpose(-1, -2)).masked_fill(dm[:, None, :] == 0, float("-inf"))
ref = (sim.max(-1).values * qm).sum(-1)
assert torch.equal(out, ref)""",
        },
        {
            "name": 'zero query mask edge case',
            "code": """import torch
q = torch.randn(1, 2, 3)
d = torch.randn(1, 1, 3)
out = {fn}(q, d, query_mask=torch.zeros(1, 2))
assert out.shape == (1,) and torch.equal(out, torch.zeros(1))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q = torch.randn(2, 3, 4, requires_grad=True)
d = torch.randn(2, 5, 4, requires_grad=True)
out = {fn}(q, d, doc_mask=torch.ones(2, 5, dtype=torch.bool))
out.sum().backward()
assert q.grad is not None and d.grad is not None and torch.isfinite(q.grad).all() and torch.isfinite(d.grad).all()""",
        },
    ],
    'solution': '''def colbert_maxsim(query_toks, doc_toks, query_mask=None, doc_mask=None, eps=1e-8):
    """计算 ColBERT 晚期交互评分 (MaxSim Scoring)。"""
    # Step 1: 沿特征维归一化查询与文档 token
    query_norm = query_toks.norm(dim=-1, keepdim=True)
    doc_norm = doc_toks.norm(dim=-1, keepdim=True)
    normalized_query = query_toks / (query_norm + eps)
    normalized_doc = doc_toks / (doc_norm + eps)

    # Step 2: 计算 token 相似度并在取最大值前屏蔽文档填充
    similarities = normalized_query @ normalized_doc.transpose(-1, -2)
    if doc_mask is not None:
        similarities = similarities.masked_fill(
            doc_mask.unsqueeze(-2) == 0, float("-inf")
        )
    maxima = similarities.max(dim=-1).values

    # Step 3: 清零查询填充并汇总每个样本分数
    if query_mask is not None:
        maxima = maxima.masked_fill(query_mask == 0, 0.0)
    scores = maxima.sum(dim=-1)
    return scores''',
    'demo': """torch.manual_seed(42)
queries = torch.randn(2, 4, 16)
documents = torch.randn(2, 8, 16)
query_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
scores = colbert_maxsim(queries, documents, query_mask=query_mask)
print("Score shape:", scores.shape)
print("Scores:", scores)
print("Finite values:", torch.isfinite(scores).all().item())""",
    'explanation': """### 核心思路
ColBERT 先对 token 向量做 L2 归一化，构造 `(B, Q, K)` 余弦相似度；每个查询 token 选择最相似的有效文档 token，再沿查询维求和得到 `(B,)` 分数。

### 代码步骤拆解
- **Step 1（向量归一化）**：分母加 `eps`，保持 `(B, Q, 1)` 与 `(B, K, 1)` 便于广播。
- **Step 2（文档 MaxSim）**：先把无效文档列设为负无穷，再沿文档维取最大值。
- **Step 3（查询汇总）**：无效查询位置清零后对 `maxima` 求和。

### 重点与易错点分析
- **注意点 1（掩码顺序）**：文档 mask 必须在 `max` 前应用，否则填充 token 可能获胜；查询 mask 则在 `max` 后清零。
- **注意点 2（零向量）**：范数分母加 `eps` 防止零 token 产生 NaN。
- **注意点 3（全空文档）**：若某样本全部文档位置无效，最大值会是负无穷；生产代码应额外定义该边界语义。

### 复杂度分析
- **时间复杂度**：$O(BQKD)$，相似度矩阵乘法占主导。
- **空间复杂度**：$O(BQK)$，主要显存是 token 两两相似度。""",
}

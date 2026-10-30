"""BM25 Scoring & Reciprocal Rank Fusion (RRF) task."""

TASK = {
    'title': 'BM25 Scoring & Reciprocal Rank Fusion (RRF)',
    'title_zh': 'BM25 检索评分与倒数排名融合（RRF）',
    'difficulty': 'Medium',
    'category': '检索与智能体',
    'function_name': 'bm25_and_rrf',
    'starter_code': """def bm25_and_rrf(query_mask, doc_tf, doc_df, doc_lens, dense_scores, k1=1.5, b=0.75, rrf_k=60):
    pass""",
    'hint': """Compute BM25 term contributions with broadcasted document-length normalization, then convert each score vector to 1-based ranks using argsort twice.""",
    'hint_zh': '通过广播的文档长度归一化计算 BM25 词项贡献，再对每个分数向量执行两次 argsort 得到从 1 开始的排名。',
    'description_en': """Compute sparse BM25 document scores and fuse their ranks with dense-retrieval ranks using reciprocal rank fusion.""",
    'description_zh': '计算稀疏 BM25 文档分数，并使用倒数排名融合将其排名与稠密检索排名结合。',
    'theory_en': """BM25 combines inverse document frequency with saturating term frequency and length normalization. RRF combines rankings without requiring their score scales to be comparable.""",
    'theory_zh': 'BM25 结合逆文档频率、饱和词频和长度归一化；RRF 无需不同检索器的分数尺度可比即可融合排名。',
    'diagram_en': """```text
query, corpus stats -> BM25 -> rank --+--> reciprocal sums
dense scores ----------------> rank --+
```""",
    'diagram_zh': """```text
查询、语料统计 -> BM25 -> 排名 --+--> 倒数求和
稠密分数 --------------------> 排名 --+
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
q = torch.tensor([1, 0, 1])
tf = torch.tensor([[2, 0, 1], [0, 3, 1]])
bm, rrf = {fn}(q, tf, torch.tensor([1, 1, 2]), torch.tensor([3, 4]), torch.tensor([.2, .8]))
assert bm.shape == (2,) and rrf.shape == (2,)""",
        },
        {
            "name": 'exact reference',
            "code": """import torch
q = torch.tensor([1., 1.])
tf = torch.tensor([[2., 0.], [1., 3.]])
df = torch.tensor([2., 1.])
lens = torch.tensor([2., 4.])
dense = torch.tensor([.9, .1])
bm, rrf = {fn}(q, tf, df, lens, dense, rrf_k=10)
avg = lens.mean()
idf = torch.log(1 + (2 - df + .5) / (df + .5))
norm = tf * 2.5 / (tf + 1.5 * (1 - .75 + .75 * lens[:, None] / avg))
ref_bm = (norm * idf * q).sum(-1)
rb = torch.argsort(torch.argsort(ref_bm, descending=True)) + 1
rd = torch.argsort(torch.argsort(dense, descending=True)) + 1
ref_rrf = 1 / (10 + rb.float()) + 1 / (10 + rd.float())
assert torch.allclose(bm, ref_bm) and torch.allclose(rrf, ref_rrf)""",
        },
        {
            "name": 'single document edge case',
            "code": """import torch
bm, rrf = {fn}(torch.tensor([0, 1]), torch.tensor([[0, 2]]), torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([-3.]), rrf_k=60)
assert bm.shape == (1,) and torch.allclose(rrf, torch.tensor([2 / 61]))""",
        },
        {
            "name": 'bm25 gradient flow',
            "code": """import torch
tf = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
bm, rrf = {fn}(torch.ones(2), tf, torch.ones(2), torch.tensor([3., 4.]), torch.tensor([.1, .2]))
(bm.sum() + rrf.sum() * 0).backward()
assert tf.grad is not None and tf.grad.shape == tf.shape and torch.isfinite(tf.grad).all()""",
        },
    ],
    'solution': '''def bm25_and_rrf(query_mask, doc_tf, doc_df, doc_lens, dense_scores, k1=1.5, b=0.75, rrf_k=60):
    """BM25 检索与倒数排名融合 (BM25 and RRF)。"""
    N = doc_tf.shape[0]

    # Step 1: 计算 IDF 与长度归一化的 BM25 分数
    avg_doc_len = doc_lens.float().mean()
    idf = torch.log(
        1.0 + (N - doc_df.float() + 0.5) / (doc_df.float() + 0.5)
    )
    term_freq = doc_tf.float()
    length_ratio = doc_lens.float().unsqueeze(-1) / avg_doc_len
    normalized_tf = term_freq * (k1 + 1.0) / (
        term_freq + k1 * (1.0 - b + b * length_ratio)
    )
    bm25_scores = (
        normalized_tf * idf.unsqueeze(0) * query_mask.float().unsqueeze(0)
    ).sum(dim=-1)

    # Step 2: 将稀疏和稠密分数转换为从 1 开始的排名
    bm25_ranks = torch.argsort(torch.argsort(bm25_scores, descending=True)) + 1
    dense_ranks = torch.argsort(torch.argsort(dense_scores, descending=True)) + 1

    # Step 3: 对两个排名执行倒数排名融合
    rrf_scores = 1.0 / (rrf_k + bm25_ranks.float())
    rrf_scores = rrf_scores + 1.0 / (rrf_k + dense_ranks.float())
    return bm25_scores, rrf_scores''',
    'demo': """torch.manual_seed(42)
query_mask = torch.tensor([1, 0, 1])
doc_tf = torch.tensor([[2, 0, 1], [0, 2, 3]])
bm25, fused = bm25_and_rrf(
    query_mask, doc_tf, torch.tensor([1, 1, 2]),
    torch.tensor([3, 5]), torch.tensor([0.2, 0.9])
)
print("Output shape:", bm25.shape)
print("BM25 scores:", bm25)
print("Fused scores:", fused)""",
    'explanation': r"""### 核心思路
BM25 用饱和词频、IDF 和文档长度归一化生成稀疏分数；RRF 用 $1/(k+r)$ 融合稀疏与稠密排名。

### 代码步骤拆解
- **Step 1（BM25）**：通过 `(N, T)` 广播计算词项贡献并沿词项维求和。
- **Step 2（排名转换）**：两次 `argsort` 把降序文档顺序映射回每个文档的 1-based rank。
- **Step 3（RRF）**：分别计算倒数排名并相加，得到尺度无关的融合分数。

### 重点与易错点分析
- **注意点 1（广播维度）**：文档长度为 `(N, 1)`，IDF 和查询掩码为 `(1, T)`。
- **注意点 2（分数尺度）**：RRF 融合排名而非原始分数，因此无需校准两种检索器。
- **注意点 3（并列排名）**：`argsort` 对完全相同分数仍给出确定顺序，不实现平均并列名次。

### 复杂度分析
- **时间复杂度**：$O(NT+N\log N)$，BM25 词项计算通常占主导。
- **空间复杂度**：$O(NT)$，主要保存逐文档词项中间量。""",
}

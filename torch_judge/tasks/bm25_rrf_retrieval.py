"""BM25 scoring and reciprocal rank fusion task."""

TASK = {
    "title": "BM25 Scoring & Reciprocal Rank Fusion (RRF)",
    "title_zh": "BM25 检索评分与倒数排名融合（RRF）",
    "difficulty": "Medium",
    "category": "检索与智能体",
    "function_name": "bm25_and_rrf",
    "starter_code": '''def bm25_and_rrf(query_mask, doc_tf, doc_df, doc_lens, dense_scores, k1=1.5, b=0.75, rrf_k=60):
    pass
''',
    "hint": "Compute BM25 term contributions with broadcasted document-length normalization, then convert each score vector to 1-based ranks using argsort twice.",
    "hint_zh": "通过广播的文档长度归一化计算 BM25 词项贡献，再对每个分数向量执行两次 argsort 得到从 1 开始的排名。",
    "description_en": "Compute sparse BM25 document scores and fuse their ranks with dense-retrieval ranks using reciprocal rank fusion.",
    "description_zh": "计算稀疏 BM25 文档分数，并使用倒数排名融合将其排名与稠密检索排名结合。",
    "theory_en": "BM25 combines inverse document frequency with saturating term frequency and length normalization. RRF combines rankings without requiring their score scales to be comparable.",
    "theory_zh": "BM25 结合逆文档频率、饱和词频和长度归一化；RRF 无需不同检索器的分数尺度可比即可融合排名。",
    "diagram_en": "```text\nquery, corpus stats -> BM25 -> rank --+--> reciprocal sums\ndense scores ----------------> rank --+\n```",
    "diagram_zh": "```text\n查询、语料统计 -> BM25 -> 排名 --+--> 倒数求和\n稠密分数 --------------------> 排名 --+\n```",
    "tests": [
        {"name": "output shapes", "code": '''import torch
q = torch.tensor([1, 0, 1])
tf = torch.tensor([[2, 0, 1], [0, 3, 1]])
bm, rrf = {fn}(q, tf, torch.tensor([1, 1, 2]), torch.tensor([3, 4]), torch.tensor([.2, .8]))
assert bm.shape == (2,) and rrf.shape == (2,)
'''},
        {"name": "exact reference", "code": '''import torch
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
assert torch.allclose(bm, ref_bm) and torch.allclose(rrf, ref_rrf)
'''},
        {"name": "single document edge case", "code": '''import torch
bm, rrf = {fn}(torch.tensor([0, 1]), torch.tensor([[0, 2]]), torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([-3.]), rrf_k=60)
assert bm.shape == (1,) and torch.allclose(rrf, torch.tensor([2 / 61]))
'''},
        {"name": "bm25 gradient flow", "code": '''import torch
tf = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
bm, rrf = {fn}(torch.ones(2), tf, torch.ones(2), torch.tensor([3., 4.]), torch.tensor([.1, .2]))
(bm.sum() + rrf.sum() * 0).backward()
assert tf.grad is not None and tf.grad.shape == tf.shape and torch.isfinite(tf.grad).all()
'''},
    ],
    "solution": '''import torch

def bm25_and_rrf(query_mask, doc_tf, doc_df, doc_lens, dense_scores, k1=1.5, b=0.75, rrf_k=60):
    N = doc_tf.shape[0]
    avgdl = doc_lens.float().mean()
    idf = torch.log(1.0 + (N - doc_df.float() + 0.5) / (doc_df.float() + 0.5))
    tf = doc_tf.float()
    length_norm = doc_lens.float().unsqueeze(-1) / avgdl
    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * length_norm))
    bm25_scores = (tf_norm * idf.unsqueeze(0) * query_mask.float().unsqueeze(0)).sum(dim=-1)
    order_bm25 = torch.argsort(bm25_scores, descending=True)
    ranks_bm25 = torch.argsort(order_bm25) + 1
    order_dense = torch.argsort(dense_scores, descending=True)
    ranks_dense = torch.argsort(order_dense) + 1
    rrf_scores = 1.0 / (rrf_k + ranks_bm25.float()) + 1.0 / (rrf_k + ranks_dense.float())
    return bm25_scores, rrf_scores
''',
    "demo": '''bm25, fused = bm25_and_rrf(torch.tensor([1, 0, 1]), torch.tensor([[2, 0, 1], [0, 2, 3]]), torch.tensor([1, 1, 2]), torch.tensor([3, 5]), torch.tensor([0.2, 0.9]))
print(bm25, fused)
''',
    "explanation": "The first argsort yields document order; sorting that order maps each original document back to its zero-based position, after which one is added.",
}

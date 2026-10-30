"""H2O and attention-sink KV-cache eviction task."""

TASK = {
    "title": "H2O (Heavy-Hitter Oracle) & Sink KV Cache Eviction",
    "title_zh": "H2O 重命中与汇聚 Token KV 驱逐",
    "difficulty": "Hard",
    "category": "推理优化",
    "function_name": "h2o_evict_kv",
    "starter_code": '''def h2o_evict_kv(K, V, attn_weights, sink_size, heavy_size, recent_size):
    pass
''',
    "hint": "Sum attention over queries, top-k only the middle slice, offset those local indices by `sink_size`, concatenate sink/heavy/recent indices, sort, and gather with an expanded index.",
    "hint_zh": "先沿查询维累加注意力，仅在中间区间取 top-k，并给局部索引加 `sink_size`；拼接 sink/heavy/recent 索引、排序后用扩展索引收集。",
    "description_en": "Evict a KV cache while preserving attention sinks, high-cumulative-attention heavy hitters, and the most recent positions.",
    "description_zh": "驱逐 KV 缓存，同时保留注意力汇聚位置、累计注意力最高的重命中位置以及最近位置。",
    "theory_en": "H2O exploits the persistence of heavy hitters, while sink and recency reservations protect positional stability and near-term context.",
    "theory_zh": "H2O 利用重命中 token 的持续重要性；保留汇聚和最近位置则保护位置稳定性与短期上下文。",
    "diagram_en": "```text\n[sinks | middle candidates | recent]\n          attention sum -> top heavy\nconcat -> temporal sort -> gather K,V\n```",
    "diagram_zh": "```text\n[汇聚 | 中间候选 | 最近]\n       注意力累加 -> top 重命中\n拼接 -> 时间排序 -> 收集 K,V\n```",
    "tests": [
        {
            "name": "budgeted output shapes",
            "code": '''import torch
K = torch.randn(2, 10, 4)
V = torch.randn(2, 10, 6)
a = torch.rand(2, 3, 10)
Kk, Vk, idx = {fn}(K, V, a, sink_size=2, heavy_size=3, recent_size=2)
assert Kk.shape == (2, 7, 4) and Vk.shape == (2, 7, 6)
assert idx.shape == (2, 7) and idx.dtype == torch.long
''',
        },
        {
            "name": "exact per-batch heavy hitters",
            "code": '''import torch
K = torch.arange(8.).view(1, 8, 1).expand(2, -1, -1).clone()
V = K + 100
a = torch.zeros(2, 2, 8)
a[0, :, 2:6] = torch.tensor([[1., 8., 2., 4.], [1., 1., 2., 4.]])
a[1, :, 2:6] = torch.tensor([[9., 1., 7., 0.], [1., 1., 1., 0.]])
Kk, Vk, idx = {fn}(K, V, a, 2, 2, 2)
ref_idx = torch.tensor([[0, 1, 3, 5, 6, 7], [0, 1, 2, 4, 6, 7]])
assert torch.equal(idx, ref_idx)
assert torch.equal(Kk.squeeze(-1), ref_idx.float())
assert torch.equal(Vk.squeeze(-1), ref_idx.float() + 100)
''',
        },
        {
            "name": "short cache keeps everything",
            "code": '''import torch
K = torch.randn(3, 5, 2)
V = torch.randn(3, 5, 2)
a = torch.rand(3, 1, 5)
Kk, Vk, idx = {fn}(K, V, a, sink_size=2, heavy_size=2, recent_size=2)
ref = torch.arange(5).expand(3, 5)
assert torch.equal(idx, ref)
assert torch.equal(Kk, K) and torch.equal(Vk, V)
''',
        },
        {
            "name": "gradient through gathered cache",
            "code": '''import torch
K = torch.randn(2, 9, 3, requires_grad=True)
V = torch.randn(2, 9, 3, requires_grad=True)
a = torch.rand(2, 4, 9)
Kk, Vk, idx = {fn}(K, V, a, 1, 3, 2)
(Kk.square().sum() + Vk.square().sum()).backward()
assert idx.shape == (2, 6)
assert K.grad is not None and V.grad is not None
assert torch.isfinite(K.grad).all() and torch.isfinite(V.grad).all()
''',
        },
    ],
    "solution": '''import torch

def h2o_evict_kv(K, V, attn_weights, sink_size, heavy_size, recent_size):
    B, L, D = K.shape
    budget = sink_size + heavy_size + recent_size
    if L <= budget:
        kept_indices = torch.arange(L, device=K.device).unsqueeze(0).expand(B, -1)
    else:
        scores = attn_weights.sum(dim=1)
        middle_end = L - recent_size
        middle_scores = scores[:, sink_size:middle_end]
        heavy_local = torch.topk(middle_scores, k=heavy_size, dim=-1).indices
        heavy_indices = heavy_local + sink_size
        sink_indices = torch.arange(sink_size, device=K.device).unsqueeze(0).expand(B, -1)
        recent_indices = torch.arange(middle_end, L, device=K.device).unsqueeze(0).expand(B, -1)
        kept_indices = torch.cat((sink_indices, heavy_indices, recent_indices), dim=-1)
        kept_indices = torch.sort(kept_indices, dim=-1).values
    K_index = kept_indices.unsqueeze(-1).expand(-1, -1, K.shape[-1])
    V_index = kept_indices.unsqueeze(-1).expand(-1, -1, V.shape[-1])
    K_kept = torch.gather(K, dim=1, index=K_index)
    V_kept = torch.gather(V, dim=1, index=V_index)
    return K_kept, V_kept, kept_indices
''',
    "demo": '''K = torch.randn(2, 2048, 64)
V = torch.randn(2, 2048, 64)
attn = torch.softmax(torch.randn(2, 16, 2048), dim=-1)
K_kept, V_kept, positions = h2o_evict_kv(K, V, attn, 4, 124, 128)
print(K_kept.shape, positions.shape)
''',
    "explanation": "Heavy-hitter selection is independent per batch item. Sorting the union restores chronological cache order before K and V are gathered.",
}

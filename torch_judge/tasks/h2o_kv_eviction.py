"""H2O (Heavy-Hitter Oracle) & Sink KV Cache Eviction task."""

TASK = {
    'title': 'H2O (Heavy-Hitter Oracle) & Sink KV Cache Eviction',
    'title_zh': 'H2O 重命中与汇聚 Token KV 驱逐',
    'difficulty': 'Hard',
    'category': '推理优化',
    'function_name': 'h2o_evict_kv',
    'starter_code': """def h2o_evict_kv(K, V, attn_weights, sink_size, heavy_size, recent_size):
    pass""",
    'hint': """Sum attention over queries, top-k only the middle slice, offset those local indices by `sink_size`, concatenate sink/heavy/recent indices, sort, and gather with an expanded index.""",
    'hint_zh': '先沿查询维累加注意力，仅在中间区间取 top-k，并给局部索引加 `sink_size`；拼接 sink/heavy/recent 索引、排序后用扩展索引收集。',
    'description_en': """Evict a KV cache while preserving attention sinks, high-cumulative-attention heavy hitters, and the most recent positions.""",
    'description_zh': '驱逐 KV 缓存，同时保留注意力汇聚位置、累计注意力最高的重命中位置以及最近位置。',
    'theory_en': """H2O exploits the persistence of heavy hitters, while sink and recency reservations protect positional stability and near-term context.""",
    'theory_zh': 'H2O 利用重命中 token 的持续重要性；保留汇聚和最近位置则保护位置稳定性与短期上下文。',
    'diagram_en': """```text
[sinks | middle candidates | recent]
          attention sum -> top heavy
concat -> temporal sort -> gather K,V
```""",
    'diagram_zh': """```text
[汇聚 | 中间候选 | 最近]
       注意力累加 -> top 重命中
拼接 -> 时间排序 -> 收集 K,V
```""",
    "tests": [
        {
            "name": 'budgeted output shapes',
            "code": """import torch
K = torch.randn(2, 10, 4)
V = torch.randn(2, 10, 6)
a = torch.rand(2, 3, 10)
Kk, Vk, idx = {fn}(K, V, a, sink_size=2, heavy_size=3, recent_size=2)
assert Kk.shape == (2, 7, 4) and Vk.shape == (2, 7, 6)
assert idx.shape == (2, 7) and idx.dtype == torch.long""",
        },
        {
            "name": 'exact per-batch heavy hitters',
            "code": """import torch
K = torch.arange(8.).view(1, 8, 1).expand(2, -1, -1).clone()
V = K + 100
a = torch.zeros(2, 2, 8)
a[0, :, 2:6] = torch.tensor([[1., 8., 2., 4.], [1., 1., 2., 4.]])
a[1, :, 2:6] = torch.tensor([[9., 1., 7., 0.], [1., 1., 1., 0.]])
Kk, Vk, idx = {fn}(K, V, a, 2, 2, 2)
ref_idx = torch.tensor([[0, 1, 3, 5, 6, 7], [0, 1, 2, 4, 6, 7]])
assert torch.equal(idx, ref_idx)
assert torch.equal(Kk.squeeze(-1), ref_idx.float())
assert torch.equal(Vk.squeeze(-1), ref_idx.float() + 100)""",
        },
        {
            "name": 'short cache keeps everything',
            "code": """import torch
K = torch.randn(3, 5, 2)
V = torch.randn(3, 5, 2)
a = torch.rand(3, 1, 5)
Kk, Vk, idx = {fn}(K, V, a, sink_size=2, heavy_size=2, recent_size=2)
ref = torch.arange(5).expand(3, 5)
assert torch.equal(idx, ref)
assert torch.equal(Kk, K) and torch.equal(Vk, V)""",
        },
        {
            "name": 'gradient through gathered cache',
            "code": """import torch
K = torch.randn(2, 9, 3, requires_grad=True)
V = torch.randn(2, 9, 3, requires_grad=True)
a = torch.rand(2, 4, 9)
Kk, Vk, idx = {fn}(K, V, a, 1, 3, 2)
(Kk.square().sum() + Vk.square().sum()).backward()
assert idx.shape == (2, 6)
assert K.grad is not None and V.grad is not None
assert torch.isfinite(K.grad).all() and torch.isfinite(V.grad).all()""",
        },
    ],
    'solution': '''def h2o_evict_kv(K, V, attn_weights, sink_size, heavy_size, recent_size):
    """执行 H2O 与注意力汇聚 KV 驱逐 (H2O KV Eviction)。"""
    B, S, D = K.shape
    budget = sink_size + heavy_size + recent_size

    # Step 1: 根据预算选择保留位置
    if S <= budget:
        kept_indices = torch.arange(S, device=K.device).unsqueeze(0).expand(B, -1)
    else:
        scores = attn_weights.sum(dim=1)
        middle_end = S - recent_size
        middle_scores = scores[:, sink_size:middle_end]
        heavy_local = torch.topk(middle_scores, k=heavy_size, dim=-1).indices
        heavy_indices = heavy_local + sink_size
        sink_indices = torch.arange(sink_size, device=K.device).unsqueeze(0).expand(B, -1)
        recent_indices = torch.arange(middle_end, S, device=K.device).unsqueeze(0).expand(B, -1)
        kept_indices = torch.cat((sink_indices, heavy_indices, recent_indices), dim=-1)
        kept_indices = kept_indices.sort(dim=-1).values

    # Step 2: 扩展索引以匹配键和值的特征维
    key_index = kept_indices.unsqueeze(-1).expand(-1, -1, D)
    value_index = kept_indices.unsqueeze(-1).expand(-1, -1, V.shape[-1])

    # Step 3: 按时间顺序收集缓存
    K_kept = torch.gather(K, dim=1, index=key_index)
    V_kept = torch.gather(V, dim=1, index=value_index)
    return K_kept, V_kept, kept_indices''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 16, 8
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
attn_weights = torch.softmax(torch.randn(B, 4, S), dim=-1)
K_kept, V_kept, positions = h2o_evict_kv(K, V, attn_weights, 2, 4, 3)
print("Output shape:", K_kept.shape)
print("Position shape:", positions.shape)
print("Kept positions:", positions[0])""",
    'explanation': """### 核心思路
固定保留开头 sink token 与最近 token，并在中间候选区按累计注意力选择 heavy hitters，合并排序后同步收集 K、V。

### 代码步骤拆解
- **Step 1（位置选择）**：注意力沿查询维求和，`topk` 为每个批次独立选出重命中位置。
- **Step 2（索引扩展）**：`kept_indices` 从 `(B, K)` 扩展到键和值各自的特征维。
- **Step 3（缓存收集）**：`gather` 返回时间顺序一致的 `K_kept` 与 `V_kept`。

### 重点与易错点分析
- **注意点 1（候选区间）**：heavy hitter 只能来自 sink 与 recent 之间，避免位置重复。
- **注意点 2（局部偏移）**：中间切片的 `topk` 索引必须加 `sink_size` 才是全局位置。
- **注意点 3（短缓存边界）**：当序列不超过预算时应原样保留，不能调用超范围 `topk`。

### 复杂度分析
- **时间复杂度**：$O(BQS+BMD)$，累计注意力与收集缓存占主导。
- **空间复杂度**：$O(BS+BKD)$，保存分数、索引与输出缓存。""",
}

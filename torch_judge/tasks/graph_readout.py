"""Graph Readout (Graph-Level Pooling) task."""

TASK = {
    'title': 'Graph Readout (Graph-Level Pooling)',
    'title_zh': '图读出（图级池化）',
    'difficulty': 'Easy',
    'category': '图神经网络',
    'function_name': 'graph_readout',
    'starter_code': """def graph_readout(X, batch, mode='mean'):
    pass""",
    'hint': """1. `num_graphs = batch.max().item() + 1`
2. For sum: `out.index_add_(0, batch, X)`
3. For mean: sum / count per graph
4. For max: loop over graph indices or use `scatter` with `reduce='amax'`""",
    'hint_zh': """1. `num_graphs = batch.max().item() + 1`
2. sum 模式：`out.index_add_(0, batch, X)`
3. mean 模式：sum / 每个图的节点数
4. max 模式：按图索引循环或使用 `scatter` 配合 `reduce='amax'`""",
    'description_en': """Implement graph-level readout (pooling) that aggregates node embeddings into a single vector per graph.

**Signature:** `graph_readout(X, batch, mode='mean') -> Tensor`

**Parameters:**
- `X` — node embedding matrix (total_nodes, D)
- `batch` — integer tensor (total_nodes,) mapping each node to its graph index (0-indexed)
- `mode` — aggregation mode: `'mean'`, `'sum'`, or `'max'`

**Returns:** graph-level embeddings of shape (num_graphs, D) where `num_graphs = batch.max() + 1`

**Algorithm:**
- `'sum'`: accumulate node features per graph (e.g., `index_add_` or `scatter_add_`)
- `'mean'`: sum per graph divided by node count per graph
- `'max'`: per-graph element-wise maximum over node features

**Constraints:**
- Do not use PyG or DGL — use only raw PyTorch operations
- Handle all three modes""",
    'description_zh': """实现图级别的读出（池化）操作，将节点嵌入聚合为每个图的单一向量表示。

**签名:** `graph_readout(X, batch, mode='mean') -> Tensor`

**参数:**
- `X` — 节点嵌入矩阵 (total_nodes, D)
- `batch` — 整数张量 (total_nodes,)，将每个节点映射到所属图的索引（从 0 开始）
- `mode` — 聚合方式：`'mean'`、`'sum'` 或 `'max'`

**返回:** 形状为 (num_graphs, D) 的图级嵌入，其中 `num_graphs = batch.max() + 1`

**算法:**
- `'sum'`：按图累加节点特征（可用 `index_add_` 或 `scatter_add_`）
- `'mean'`：按图求和后除以各图节点数
- `'max'`：按图对节点特征逐元素取最大值

**约束:**
- 仅使用原生 PyTorch 操作，不得使用 PyG 或 DGL
- 需支持全部三种聚合模式""",
    "tests": [
        {
            "name": 'Single graph (all batch=0), mean mode',
            "code": """import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
batch = torch.tensor([0, 0, 0])
out = {fn}(X, batch, mode='mean')
assert out.shape == (1, 2), f'Expected (1, 2), got {out.shape}'
expected = X.mean(dim=0, keepdim=True)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'""",
        },
        {
            "name": 'Multi-graph batch, sum mode',
            "code": """import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
batch = torch.tensor([0, 0, 1, 1])
out = {fn}(X, batch, mode='sum')
assert out.shape == (2, 2), f'Expected (2, 2), got {out.shape}'
expected = torch.tensor([[4.0, 6.0], [12.0, 14.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'""",
        },
        {
            "name": 'Sum mode accumulates correctly',
            "code": """import torch
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
batch = torch.tensor([0, 1, 0])
out = {fn}(X, batch, mode='sum')
assert out.shape == (2, 2), f'Expected (2, 2), got {out.shape}'
expected = torch.tensor([[3.0, 3.0], [0.0, 1.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'""",
        },
        {
            "name": 'Mean mode divides by node count',
            "code": """import torch
X = torch.tensor([[2.0, 4.0], [4.0, 6.0], [10.0, 20.0]])
batch = torch.tensor([0, 0, 1])
out = {fn}(X, batch, mode='mean')
expected = torch.tensor([[3.0, 5.0], [10.0, 20.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'""",
        },
        {
            "name": 'Max mode picks element-wise maximum',
            "code": """import torch
X = torch.tensor([[1.0, 5.0], [3.0, 2.0], [4.0, 1.0]])
batch = torch.tensor([0, 0, 1])
out = {fn}(X, batch, mode='max')
expected = torch.tensor([[3.0, 5.0], [4.0, 1.0]])
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected}, got {out}'""",
        },
        {
            "name": 'Exact numerical value with three graphs',
            "code": """import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
batch = torch.tensor([0, 0, 1, 2, 2])
out_sum = {fn}(X, batch, mode='sum')
out_mean = {fn}(X, batch, mode='mean')
out_max = {fn}(X, batch, mode='max')
exp_sum = torch.tensor([[4.0, 6.0], [5.0, 6.0], [16.0, 18.0]])
exp_mean = torch.tensor([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])
exp_max = torch.tensor([[3.0, 4.0], [5.0, 6.0], [9.0, 10.0]])
assert torch.allclose(out_sum, exp_sum, atol=1e-5), f'Sum mismatch: {out_sum} vs {exp_sum}'
assert torch.allclose(out_mean, exp_mean, atol=1e-5), f'Mean mismatch: {out_mean} vs {exp_mean}'
assert torch.allclose(out_max, exp_max, atol=1e-5), f'Max mismatch: {out_max} vs {exp_max}'""",
        },
    ],
    'solution': '''def graph_readout(X, batch, mode='mean'):
    """聚合节点表示为图级向量 (Graph Readout)。"""
    num_graphs = int(batch.max().item()) + 1
    D = X.shape[1]

    # Step 1: 按图索引累加节点特征
    summed = X.new_zeros(num_graphs, D)
    summed.index_add_(0, batch, X)
    if mode == "sum":
        return summed

    # Step 2: 根据模式计算均值或逐元素最大值
    if mode == "mean":
        counts = X.new_zeros(num_graphs)
        counts.index_add_(0, batch, X.new_ones(batch.shape[0]))
        return summed / counts[:, None]
    if mode == "max":
        rows = [X[batch == graph_id].max(dim=0).values for graph_id in range(num_graphs)]
        return torch.stack(rows)

    # Step 3: 拒绝未知聚合模式
    raise ValueError(f"Unsupported mode: {mode}")''',
    'demo': """torch.manual_seed(42)
X = torch.randn(6, 4)
batch = torch.tensor([0, 0, 0, 1, 1, 1])
mean_out = graph_readout(X, batch, mode="mean")
max_out = graph_readout(X, batch, mode="max")
print("Output shape:", mean_out.shape)
print("Mean readout:", mean_out)
print("Max readout:", max_out)""",
    'explanation': """### 核心思路
依据 `batch` 将节点特征分组，分别执行求和、均值或逐元素最大值，把 `(total_nodes, D)` 压缩为 `(num_graphs, D)`。

### 代码步骤拆解
- **Step 1（分组求和）**：`index_add_` 将每行 `X` 累加到对应图的 `summed` 行。
- **Step 2（模式归约）**：均值模式除以节点数，最大值模式逐图取 `max`。
- **Step 3（参数校验）**：未知 `mode` 明确抛出 `ValueError`。

### 重点与易错点分析
- **注意点 1（广播维度）**：计数需用 `counts[:, None]` 扩展，才能按图除以每个特征。
- **注意点 2（设备与类型）**：用 `X.new_zeros` 创建累加张量，确保设备和 dtype 一致。
- **注意点 3（图编号假设）**：实现假设图编号从 0 连续排列且每个编号至少有一个节点。

### 复杂度分析
- **时间复杂度**：$O(ND)$，每个节点特征被归约一次。
- **空间复杂度**：$O(GD)$，保存 $G$ 个图的输出。""",
}

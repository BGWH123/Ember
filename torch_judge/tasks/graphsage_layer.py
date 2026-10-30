"""GraphSAGE Layer task."""

TASK = {
    'title': 'GraphSAGE Layer',
    'title_zh': 'GraphSAGE 层',
    'difficulty': 'Hard',
    'category': '图神经网络',
    'function_name': 'graphsage_layer',
    'starter_code': """def graphsage_layer(A, X, W, k=None):
    pass""",
    'hint': """1. Loop over nodes, gather neighbors from A[i].nonzero()
2. If k sampling needed: `torch.manual_seed(42)`, `perm = torch.randperm(n_neighbors)[:k]`
3. Mean-pool neighbor features (or zeros if isolated)
4. `concat = torch.cat([X, agg], dim=1)` → (N, 2*F_in)
5. `out = relu(concat @ W)`
6. `out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)`""",
    'hint_zh': """1. 遍历节点，从 A[i].nonzero() 获取邻居
2. 若需采样：`torch.manual_seed(42)`，`perm = torch.randperm(n_neighbors)[:k]`
3. 均值池化邻居特征（孤立节点用零向量）
4. `concat = torch.cat([X, agg], dim=1)` → (N, 2*F_in)
5. `out = relu(concat @ W)`
6. `out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)`""",
    'description_en': """Implement a GraphSAGE layer that aggregates neighbor features via mean pooling, concatenates with self features, and applies a linear transformation with L2 normalization.

**Signature:** `graphsage_layer(A, X, W, k=None) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (2*F_in, F_out)
- `k` — if not None, sample at most k neighbors per node (use `torch.manual_seed(42)` before sampling for determinism)

**Returns:** (N, F_out) tensor, L2-normalized along the feature dimension

**Algorithm:**
1. For each node i:
   - Get neighbor indices where A[i] > 0
   - If k is not None and num_neighbors > k: set `torch.manual_seed(42)`, use `torch.randperm(num_neighbors)[:k]` to sample
   - Compute mean of neighbor features (zero vector if no neighbors)
2. Concatenate [X, neighbor_agg] along feature dim to get (N, 2*F_in)
3. Apply linear transform: `out = ReLU(concat @ W)`
4. L2 normalize: `out = out / ||out||_2` per row (clamp norm min to 1e-8)

**Constraints:**
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` for activation
- Sampling must be deterministic with `torch.manual_seed(42)` called immediately before each `randperm`""",
    'description_zh': """实现 GraphSAGE 层：通过均值池化聚合邻居特征，与自身特征拼接后进行线性变换，最后做 L2 归一化。

**签名:** `graphsage_layer(A, X, W, k=None) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (2*F_in, F_out)
- `k` — 若不为 None，每个节点最多采样 k 个邻居（采样前调用 `torch.manual_seed(42)` 保证确定性）

**返回:** (N, F_out) 张量，沿特征维度做 L2 归一化

**算法:**
1. 对每个节点 i：
   - 获取 A[i] > 0 的邻居索引
   - 若 k 不为 None 且邻居数 > k：设置 `torch.manual_seed(42)`，用 `torch.randperm(num_neighbors)[:k]` 采样
   - 计算邻居特征均值（无邻居则为零向量）
2. 沿特征维拼接 [X, neighbor_agg] 得到 (N, 2*F_in)
3. 线性变换：`out = ReLU(concat @ W)`
4. L2 归一化：`out = out / ||out||_2`（每行归一化，范数下限 clamp 为 1e-8）

**约束:**
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 作为激活函数
- 采样必须确定性：每次 `randperm` 前立即调用 `torch.manual_seed(42)`""",
    "tests": [
        {
            "name": 'Output shape check',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected shape ({N},{F_out}), got {out.shape}'""",
        },
        {
            "name": 'Output is L2-normalized',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 6, 4, 8
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out = {fn}(A, X, W)
norms = out.norm(dim=-1)
# Rows with non-zero output should have unit norm
nonzero_mask = norms > 1e-7
assert torch.allclose(norms[nonzero_mask], torch.ones_like(norms[nonzero_mask]), atol=1e-5), f'Output rows not L2-normalized: norms = {norms}'""",
        },
        {
            "name": 'k=None uses all neighbors',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
# Node 0 connected to all others
A[0,1] = A[1,0] = 1.0
A[0,2] = A[2,0] = 1.0
A[0,3] = A[3,0] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out_none = {fn}(A, X, W, k=None)
out_large_k = {fn}(A, X, W, k=100)
assert torch.allclose(out_none, out_large_k, atol=1e-6), 'k=None and k>num_neighbors should give same result'""",
        },
        {
            "name": 'k sampling is deterministic',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 8, 4, 3
A = torch.zeros(N, N)
# Dense graph
for i in range(N):
    for j in range(i+1, N):
        A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(2*F_in, F_out)
out1 = {fn}(A, X, W, k=2)
out2 = {fn}(A, X, W, k=2)
assert torch.allclose(out1, out2, atol=1e-6), 'k-sampling should be deterministic with manual_seed(42)'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
X = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
W = torch.tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
out = {fn}(A, X, W)
# Manual computation
# Node 0 neighbors: [1,2], mean = ([0,1]+[1,1])/2 = [0.5, 1.0]
# Node 1 neighbors: [0], mean = [1, 0]
# Node 2 neighbors: [0], mean = [1, 0]
agg = torch.tensor([[0.5, 1.0],[1.0, 0.0],[1.0, 0.0]])
concat = torch.cat([X, agg], dim=1)  # (3, 4)
raw = torch.relu(concat @ W)  # (3, 2)
expected = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff = {(out - expected).abs().max().item():.6f}'""",
        },
        {
            "name": 'Gradient flow through X and W',
            "code": """import torch
torch.manual_seed(1)
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(2*F_in, F_out, requires_grad=True)
out = {fn}(A, X, W)
loss = out.sum()
loss.backward()
assert X.grad is not None, 'No gradient for X'
assert X.grad.abs().sum() > 0, 'X gradient is all zeros'
assert W.grad is not None, 'No gradient for W'
assert W.grad.abs().sum() > 0, 'W gradient is all zeros'""",
        },
        {
            "name": 'k < num_neighbors changes aggregation',
            "code": """import torch
N, F_in, F_out = 6, 4, 3
A = torch.zeros(N, N)
# Node 0 connected to all others (5 neighbors)
for j in range(1, N):
    A[0,j] = A[j,0] = 1.0
# Use distinct features so any subset gives a different mean
X = torch.tensor([[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,2.,0.,0.],[0.,0.,3.,0.],[0.,0.,0.,4.],[5.,5.,5.,5.]])
W = torch.eye(2*F_in, F_out)[:2*F_in, :F_out]
W = torch.randn(2*F_in, F_out)
torch.manual_seed(99)
W = torch.randn(2*F_in, F_out)
out_all = {fn}(A, X, W, k=None)
out_k2 = {fn}(A, X, W, k=2)
# Node 0 has 5 neighbors with distinct features; k=2 samples a subset, giving different aggregation
assert not torch.allclose(out_all[0], out_k2[0], atol=1e-5), 'k=2 with 5 distinct neighbors should differ from using all neighbors'""",
        },
        {
            "name": 'Isolated node gets zero neighbor aggregation',
            "code": """import torch
N, F_in, F_out = 3, 2, 2
A = torch.zeros(N, N)
A[0,1] = A[1,0] = 1.0
# Node 2 is isolated
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[2.0, 3.0]])
W = torch.tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
out = {fn}(A, X, W)
# Node 2: neighbors=[], agg=zeros(2), concat=[2,3,0,0]
concat_2 = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
raw_2 = torch.relu(concat_2 @ W)
expected_2 = raw_2 / raw_2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
assert torch.allclose(out[2], expected_2.squeeze(), atol=1e-5), f'Isolated node mismatch: {out[2]} vs {expected_2.squeeze()}'""",
        },
    ],
    'solution': '''def graphsage_layer(A, X, W, k=None):
    """计算均值聚合 GraphSAGE 层 (GraphSAGE Layer)。"""
    N, F_in = X.shape
    aggregated = []

    # Step 1: 为每个节点选择邻居并计算均值
    for node_id in range(N):
        neighbors = (A[node_id] > 0).nonzero(as_tuple=True)[0]
        if neighbors.numel() == 0:
            neighbor_mean = X.new_zeros(F_in)
        else:
            if k is not None and neighbors.numel() > k:
                torch.manual_seed(42)
                selected = torch.randperm(neighbors.numel(), device=neighbors.device)[:k]
                neighbors = neighbors[selected]
            neighbor_mean = X[neighbors].mean(dim=0)
        aggregated.append(neighbor_mean)

    # Step 2: 拼接自身与邻居表示并线性变换
    neighbor_features = torch.stack(aggregated)
    combined = torch.cat((X, neighbor_features), dim=1)
    out = torch.relu(combined @ W)

    # Step 3: 对每个节点输出做 L2 归一化
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)''',
    'demo': """torch.manual_seed(42)
N, F_in, F_out = 6, 4, 3
A = torch.ones(N, N) - torch.eye(N)
X = torch.randn(N, F_in)
W = torch.randn(2 * F_in, F_out)
out = graphsage_layer(A, X, W, k=2)
print("Output shape:", out.shape)
print("Row norms:", out.norm(dim=-1))
print("Sample values:", out[0])""",
    'explanation': """### 核心思路
GraphSAGE 对每个节点均值聚合邻居特征，与自身特征拼接后做线性变换和 ReLU，最后按行进行 L2 归一化。

### 代码步骤拆解
- **Step 1（邻居采样）**：每个 `neighbor_mean` 为 `(F_in,)`，孤立节点使用零向量。
- **Step 2（特征融合）**：`combined` 为 `(N, 2*F_in)`，乘 `W` 后得到 `(N, F_out)`。
- **Step 3（行归一化）**：用 `keepdim=True` 的范数逐行缩放输出。

### 重点与易错点分析
- **注意点 1（确定性采样）**：每次需要采样时都应紧邻 `randperm` 调用 `manual_seed(42)`。
- **注意点 2（孤立节点）**：零邻居不能直接求均值，否则会产生 NaN。
- **注意点 3（零输出）**：范数用 `clamp(min=1e-8)`，保证全零行仍为有限值。

### 复杂度分析
- **时间复杂度**：$O(N^2+NF_{in}F_{out})$，稠密邻接扫描和线性映射占主导。
- **空间复杂度**：$O(NF_{in}+NF_{out})$，保存聚合与输出特征。""",
}

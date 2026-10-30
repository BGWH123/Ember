"""GCN Layer (Graph Convolution) task."""

TASK = {
    'title': 'GCN Layer (Graph Convolution)',
    'title_zh': 'GCN 层（图卷积）',
    'difficulty': 'Easy',
    'category': '图神经网络',
    'function_name': 'gcn_layer',
    'starter_code': """def gcn_layer(A, X, W):
    pass""",
    'hint': """1. `A_tilde = A + torch.eye(N)`
2. `D_vec = A_tilde.sum(1)`
3. `D_inv_sqrt = torch.diag(D_vec.pow(-0.5))`
4. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. `return torch.relu(A_norm @ X @ W)`""",
    'hint_zh': """1. `A_tilde = A + torch.eye(N)`
2. `D_vec = A_tilde.sum(1)`
3. `D_inv_sqrt = torch.diag(D_vec.pow(-0.5))`
4. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. `return torch.relu(A_norm @ X @ W)`""",
    'description_en': """Implement a single Graph Convolutional Network (GCN) layer with symmetric normalization.

**Signature:** `gcn_layer(A, X, W) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (F_in, F_out)

**Returns:** output node features of shape (N, F_out)

**Algorithm:**
1. Add self-loops: `A_tilde = A + I`
2. Compute degree vector: `D_vec = A_tilde.sum(1)`
3. Symmetric normalization: `D_inv_sqrt = diag(D_vec^{-0.5})`
4. Normalized adjacency: `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. Return `ReLU(A_norm @ X @ W)`

**Constraints:**
- Weights `W` are passed in — do not create them inside the function
- Use `torch.relu` for activation""",
    'description_zh': """实现单层图卷积网络 (GCN)，采用对称归一化。

**签名:** `gcn_layer(A, X, W) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (F_in, F_out)

**返回:** 形状为 (N, F_out) 的输出节点特征

**算法:**
1. 添加自环：`A_tilde = A + I`
2. 计算度向量：`D_vec = A_tilde.sum(1)`
3. 对称归一化：`D_inv_sqrt = diag(D_vec^{-0.5})`
4. 归一化邻接矩阵：`A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
5. 返回 `ReLU(A_norm @ X @ W)`

**约束:**
- 权重 `W` 由参数传入，不要在函数内部创建
- 使用 `torch.relu` 作为激活函数""",
    "tests": [
        {
            "name": 'Output shape matches (N, F_out)',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected ({N},{F_out}), got {out.shape}'""",
        },
        {
            "name": 'Output is non-negative (ReLU applied)',
            "code": """import torch
torch.manual_seed(0)
N, F_in, F_out = 8, 6, 4
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert (out >= 0).all(), f'Output has negative values, ReLU not applied: min={out.min().item()}'""",
        },
        {
            "name": 'Isolated node gets only self-features',
            "code": """import torch
N, F_in, F_out = 3, 2, 2
A = torch.zeros(N, N)
A[0, 1] = A[1, 0] = 1.0
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
W = torch.eye(F_in)
out = {fn}(A, X, W)
expected_node2 = torch.relu(X[2:3] @ W)
assert torch.allclose(out[2], expected_node2.squeeze(), atol=1e-5), f'Isolated node should only use self-features: got {out[2]}, expected {expected_node2.squeeze()}'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
X = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
W = torch.tensor([[1.,0.],[0.,1.]])
out = {fn}(A, X, W)
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
expected = torch.relu(A_norm @ X @ W)
assert torch.allclose(out, expected, atol=1e-5), f'Numerical mismatch: max diff = {(out - expected).abs().max().item():.6f}'""",
        },
        {
            "name": 'Gradient flows through X and W',
            "code": """import torch
N, F_in, F_out = 4, 3, 2
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(F_in, F_out, requires_grad=True)
out = {fn}(A, X, W)
loss = out.sum()
loss.backward()
assert X.grad is not None and X.grad.abs().sum() > 0, 'No gradient for X'
assert W.grad is not None and W.grad.abs().sum() > 0, 'No gradient for W'""",
        },
        {
            "name": 'Works with different F_in and F_out sizes',
            "code": """import torch
torch.manual_seed(7)
N, F_in, F_out = 6, 10, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = {fn}(A, X, W)
assert out.shape == (N, F_out), f'Expected ({N},{F_out}), got {out.shape}'
assert (out >= 0).all(), 'Output has negative values'""",
        },
    ],
    'solution': '''def gcn_layer(A, X, W):
    """执行对称归一化图卷积 (GCN Layer)。"""
    N = A.shape[0]

    # Step 1: 加入自环并计算对称归一化系数
    adjacency = A + torch.eye(N, device=A.device, dtype=A.dtype)
    degree_inv_sqrt = adjacency.sum(dim=1).pow(-0.5)
    adjacency_norm = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    # Step 2: 聚合邻居特征并完成线性映射
    hidden = adjacency_norm @ X @ W

    # Step 3: 应用 ReLU 得到节点输出
    return torch.relu(hidden)''',
    'demo': """torch.manual_seed(42)
N, F_in, F_out = 5, 4, 3
A = torch.zeros(N, N)
A[torch.arange(N - 1), torch.arange(1, N)] = 1.0
A = A + A.T
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
out = gcn_layer(A, X, W)
print("Output shape:", out.shape)
print("Minimum value:", out.min().item())
print("Sample values:", out[0])""",
    'explanation': r"""### 核心思路
加入自环后构造 $\hat A=D^{-1/2}(A+I)D^{-1/2}$，再计算 $\operatorname{ReLU}(\hat AXW)$ 完成邻居聚合。

### 代码步骤拆解
- **Step 1（自环与度）**：`adjacency` 为 `(N, N)`，自环保证孤立节点保留自身特征。
- **Step 2（对称归一化）**：`degree_inv_sqrt` 在行列两侧广播得到 `adjacency_norm`。
- **Step 3（卷积输出）**：矩阵乘法将特征映射到 `(N, F_out)` 并应用 ReLU。

### 重点与易错点分析
- **注意点 1（归一化方向）**：两侧的 $D^{-1/2}$ 都不能遗漏，否则不再是标准 GCN 对称归一化。
- **注意点 2（设备一致性）**：单位阵必须继承 `A` 的设备和 dtype。
- **注意点 3（稠密实现）**：大图应改用稀疏矩阵乘法，避免显式保存 $N^2$ 邻接元素。

### 复杂度分析
- **时间复杂度**：$O(N^2F_{in}+NF_{in}F_{out})$，邻接聚合是主要瓶颈。
- **空间复杂度**：$O(N^2+NF_{out})$，主要保存归一化邻接矩阵。""",
}

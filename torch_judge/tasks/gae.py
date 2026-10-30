"""Graph Autoencoder (GAE) task."""

TASK = {
    'title': 'Graph Autoencoder (GAE)',
    'title_zh': '图自编码器（GAE）',
    'difficulty': 'Hard',
    'category': '损失函数',
    'function_name': 'gae',
    'starter_code': """def gae(A, X):
    pass""",
    'hint': """1. `A_hat = A + I` (add self-loops)
2. `D_inv_sqrt = diag(A_hat.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt`
4. `W1 = randn(F, 32) * 0.1`, `W2 = randn(32, 16) * 0.1`
5. `H = relu(A_norm @ X @ W1)`
6. `Z = A_norm @ H @ W2`
7. `return sigmoid(Z @ Z.T)`""",
    'hint_zh': """1. `A_hat = A + I`（加自环）
2. `D_inv_sqrt = diag(A_hat.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt`
4. `W1 = randn(F, 32) * 0.1`，`W2 = randn(32, 16) * 0.1`
5. `H = relu(A_norm @ X @ W1)`
6. `Z = A_norm @ H @ W2`
7. `return sigmoid(Z @ Z.T)`""",
    'description_en': """Implement a Graph Autoencoder that learns node embeddings via a two-layer GCN encoder and reconstructs the adjacency matrix with an inner-product decoder.

**Signature:** `gae(A, X) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F)

**Returns:** reconstructed adjacency matrix `A_hat` of shape (N, N), with values in [0, 1] (apply sigmoid)

**Architecture:**
1. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}` where D is the degree matrix of `A + I`
2. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)` with `W1` shape (F, 32)
3. GCN Layer 2: `Z = A_norm @ H @ W2` with `Z` shape (N, 16), `W2` shape (32, 16)
4. Inner-product decoder: `A_hat = sigmoid(Z @ Z^T)`

**Constraints:**
- Initialize `W1` and `W2` with `torch.manual_seed(0)` then `torch.randn(...) * 0.1` (W1 first, then W2)
- `W1` shape: (F, 32), `W2` shape: (32, 16)
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` and `torch.sigmoid`""",
    'description_zh': """实现图自编码器 (GAE)，通过两层 GCN 编码器学习节点嵌入，并用内积解码器重建邻接矩阵。

**签名:** `gae(A, X) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F)

**返回:** 重建的邻接矩阵 `A_hat`，形状 (N, N)，值在 [0, 1] 之间（应用 sigmoid）

**架构:**
1. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`，其中 D 是 `A + I` 的度矩阵
2. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`，`W1` 形状 (F, 32)
3. GCN 第二层：`Z = A_norm @ H @ W2`，`Z` 形状 (N, 16)，`W2` 形状 (32, 16)
4. 内积解码器：`A_hat = sigmoid(Z @ Z^T)`

**约束:**
- 用 `torch.manual_seed(0)` 初始化 `W1` 和 `W2`，然后依次 `torch.randn(...) * 0.1`（先 W1 后 W2）
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 和 `torch.sigmoid`""",
    "tests": [
        {
            "name": 'Output shape matches input',
            "code": """import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
A_hat = {fn}(A, X)
assert A_hat.shape == (N, N), f'Expected ({N},{N}), got {A_hat.shape}'""",
        },
        {
            "name": 'Output values in [0, 1]',
            "code": """import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
A_hat = {fn}(A, X)
assert A_hat.min() >= 0.0, f'Min value {A_hat.min().item()} < 0'
assert A_hat.max() <= 1.0, f'Max value {A_hat.max().item()} > 1'""",
        },
        {
            "name": 'Output is symmetric',
            "code": """import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
A_hat = {fn}(A, X)
assert torch.allclose(A_hat, A_hat.T, atol=1e-5), 'Output should be symmetric since decoder is Z@Z.T'""",
        },
        {
            "name": 'Deterministic with fixed seed',
            "code": """import torch
torch.manual_seed(42)
N, F = 5, 4
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
A_hat1 = {fn}(A, X)
A_hat2 = {fn}(A, X)
assert torch.allclose(A_hat1, A_hat2, atol=1e-6), 'Same input should give same output (seed is set inside function)'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3),(0,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(7)
X = torch.randn(N, F)
A_hat = {fn}(A, X)
# Compute expected
torch.manual_seed(0)
W1 = torch.randn(F, 32) * 0.1
W2 = torch.randn(32, 16) * 0.1
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
H = torch.relu(A_norm @ X @ W1)
Z = A_norm @ H @ W2
expected = torch.sigmoid(Z @ Z.T)
assert torch.allclose(A_hat, expected, atol=1e-5), f'Numerical mismatch: max diff = {(A_hat - expected).abs().max().item():.6f}'""",
        },
        {
            "name": 'Gradient flow through weights',
            "code": """import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges = [(0,1),(1,2),(2,3)]
for i,j in edges:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F, requires_grad=True)
A_hat = {fn}(A, X)
loss = (A_hat - A).pow(2).mean()
loss.backward()
assert X.grad is not None, 'No gradient for X'
assert X.grad.abs().sum() > 0, 'Gradient is all zeros'""",
        },
    ],
    'solution': '''def gae(A, X):
    """构建图自编码器的邻接重建结果 (Graph Autoencoder)。"""
    N, F_in = X.shape

    # Step 1: 初始化固定权重并构造对称归一化邻接矩阵
    torch.manual_seed(0)
    W1 = torch.randn(F_in, 32, device=X.device, dtype=X.dtype) * 0.1
    W2 = torch.randn(32, 16, device=X.device, dtype=X.dtype) * 0.1
    adjacency = A + torch.eye(N, device=A.device, dtype=A.dtype)
    degree_inv_sqrt = adjacency.sum(dim=1).pow(-0.5)
    adjacency_norm = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    # Step 2: 通过两层 GCN 编码节点表示
    hidden = torch.relu(adjacency_norm @ X @ W1)
    embeddings = adjacency_norm @ hidden @ W2

    # Step 3: 用节点嵌入内积解码邻接概率
    logits = embeddings @ embeddings.T
    return torch.sigmoid(logits)''',
    'demo': """torch.manual_seed(42)
N, F_in = 6, 8
A = (torch.rand(N, N) > 0.6).float()
A = torch.triu(A, diagonal=1)
A = A + A.T
X = torch.randn(N, F_in)
reconstructed = gae(A, X)
print("Output shape:", reconstructed.shape)
print("Value range:", reconstructed.min().item(), reconstructed.max().item())
print("Symmetry error:", (reconstructed - reconstructed.T).abs().max().item())""",
    'explanation': r"""### 核心思路
先计算 $\hat A=D^{-1/2}(A+I)D^{-1/2}$，再用两层 GCN 得到节点嵌入 $Z$，最后通过 $\sigma(ZZ^\top)$ 重建边概率。

### 代码步骤拆解
- **Step 1（邻接归一化）**：`adjacency` 加入自环，`degree_inv_sqrt` 通过广播在行列两侧完成对称归一化。
- **Step 2（图卷积编码）**：`X` 从 `(N, F_in)` 依次映射到 `hidden` 的 `(N, 32)` 和 `embeddings` 的 `(N, 16)`。
- **Step 3（内积解码）**：`embeddings @ embeddings.T` 形成 `(N, N)` 对称 logits，`sigmoid` 将其变为概率。

### 重点与易错点分析
- **注意点 1（确定性初始化）**：函数内必须先 `manual_seed(0)`，并严格按 `W1`、`W2` 顺序采样，才能与题目参考数值一致。
- **注意点 2（设备与类型）**：随机权重和单位阵应继承 `X`/`A` 的设备与 dtype，避免 GPU 或双精度输入发生不匹配。
- **注意点 3（数值边界）**：加入自环后每个节点度至少为 1，因此 `pow(-0.5)` 不会因零度节点产生无穷值。

### 复杂度分析
- **时间复杂度**：$O(N^2F_{in}+N^2D_z)$，稠密邻接乘法和内积解码是主要瓶颈。
- **空间复杂度**：$O(N^2+NF_{in})$，归一化邻接矩阵与重建矩阵占据主要空间。""",
}

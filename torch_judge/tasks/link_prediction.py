"""Link Prediction task."""

TASK = {
    'title': 'Link Prediction',
    'title_zh': '链接预测',
    'difficulty': 'Hard',
    'category': '图神经网络',
    'function_name': 'link_prediction',
    'starter_code': """def link_prediction(A, X, edges):
    pass""",
    'hint': """1. `A_tilde = A + I`
2. `D_inv_sqrt = diag(A_tilde.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
4. Two-layer GCN: H = relu(A_norm @ X @ W1), Z = A_norm @ H @ W2
5. Score each edge: `sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))`""",
    'hint_zh': """1. `A_tilde = A + I`（加自环）
2. `D_inv_sqrt = diag(A_tilde.sum(1))^{-0.5}`
3. `A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt`
4. 两层 GCN：H = relu(A_norm @ X @ W1)，Z = A_norm @ H @ W2
5. 对每条边打分：`sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))`""",
    'description_en': """Implement a link prediction model using a two-layer GCN encoder that predicts edge existence probabilities.

**Signature:** `link_prediction(A, X, edges) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, symmetric, no self-loops
- `X` — node feature matrix (N, F)
- `edges` — (M, 2) long tensor of node pairs to predict

**Returns:** (M,) tensor of probabilities in [0, 1], where each value is the predicted likelihood of an edge existing between the corresponding node pair

**Algorithm:**
1. Initialize weights with `torch.manual_seed(0)`: `W1 = randn(F, 16) * 0.1`, then `W2 = randn(16, 8) * 0.1`
2. Compute degree-normalized adjacency: `A_norm = D^{-1/2} (A + I) D^{-1/2}`
3. GCN Layer 1: `H = ReLU(A_norm @ X @ W1)`
4. GCN Layer 2: `Z = A_norm @ H @ W2` (no activation)
5. For each edge (i, j): `score = sigmoid(Z[i] · Z[j])`
6. Return all scores as (M,) tensor

**Constraints:**
- Do not use `nn.Module` or `nn.Linear`
- Use `torch.relu` and `torch.sigmoid`""",
    'description_zh': """实现基于两层 GCN 编码器的链接预测模型，预测节点对之间存在边的概率。

**签名:** `link_prediction(A, X, edges) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值、对称、无自环
- `X` — 节点特征矩阵 (N, F)
- `edges` — (M, 2) 长整型张量，表示待预测的节点对

**返回:** (M,) 概率张量，值在 [0, 1] 之间，每个值表示对应节点对之间存在边的预测概率

**算法:**
1. 用 `torch.manual_seed(0)` 初始化权重：`W1 = randn(F, 16) * 0.1`，然后 `W2 = randn(16, 8) * 0.1`
2. 计算度归一化邻接矩阵：`A_norm = D^{-1/2} (A + I) D^{-1/2}`
3. GCN 第一层：`H = ReLU(A_norm @ X @ W1)`
4. GCN 第二层：`Z = A_norm @ H @ W2`（无激活函数）
5. 对每条边 (i, j)：`score = sigmoid(Z[i] · Z[j])`
6. 返回所有分数组成的 (M,) 张量

**约束:**
- 不得使用 `nn.Module` 或 `nn.Linear`
- 使用 `torch.relu` 和 `torch.sigmoid`""",
    "tests": [
        {
            "name": 'Output shape is (M,)',
            "code": """import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[2,3],[0,4],[1,5]], dtype=torch.long)
out = {fn}(A, X, edges)
assert out.shape == (4,), f'Expected shape (4,), got {out.shape}'""",
        },
        {
            "name": 'Values in [0, 1]',
            "code": """import torch
torch.manual_seed(42)
N, F = 6, 8
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[2,3],[0,4],[1,5],[3,5],[0,2]], dtype=torch.long)
out = {fn}(A, X, edges)
assert out.min() >= 0.0, f'Min value {out.min().item()} < 0'
assert out.max() <= 1.0, f'Max value {out.max().item()} > 1'""",
        },
        {
            "name": 'Deterministic with same input',
            "code": """import torch
torch.manual_seed(42)
N, F = 5, 4
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F)
edges = torch.tensor([[0,2],[1,3],[0,4]], dtype=torch.long)
out1 = {fn}(A, X, edges)
out2 = {fn}(A, X, edges)
assert torch.allclose(out1, out2, atol=1e-6), 'Same input should give same output (seed is set inside function)'""",
        },
        {
            "name": 'Exact numerical value',
            "code": """import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(0,3)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(7)
X = torch.randn(N, F)
edges = torch.tensor([[0,1],[1,2],[0,2]], dtype=torch.long)
out = {fn}(A, X, edges)
# Compute expected
torch.manual_seed(0)
W1 = torch.randn(F, 16) * 0.1
W2 = torch.randn(16, 8) * 0.1
A_tilde = A + torch.eye(N)
D_vec = A_tilde.sum(1)
D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
H = torch.relu(A_norm @ X @ W1)
Z = A_norm @ H @ W2
scores = torch.sigmoid((Z[edges[:,0]] * Z[edges[:,1]]).sum(dim=1))
assert torch.allclose(out, scores, atol=1e-5), f'Numerical mismatch: max diff = {(out - scores).abs().max().item():.6f}'""",
        },
        {
            "name": 'Gradient flow through X',
            "code": """import torch
N, F = 4, 3
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
X = torch.randn(N, F, requires_grad=True)
edges = torch.tensor([[0,1],[1,2],[0,3]], dtype=torch.long)
out = {fn}(A, X, edges)
loss = out.sum()
loss.backward()
assert X.grad is not None, 'No gradient for X'
assert X.grad.abs().sum() > 0, 'Gradient is all zeros'""",
        },
        {
            "name": 'Symmetric node pairs get same score',
            "code": """import torch
N, F = 6, 4
A = torch.zeros(N, N)
edges_list = [(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
for i,j in edges_list:
    A[i,j] = A[j,i] = 1.0
torch.manual_seed(42)
X = torch.randn(N, F)
# (i,j) and (j,i) should produce the same score since decoder is dot product
edges_fwd = torch.tensor([[0,1],[2,3],[1,4]], dtype=torch.long)
edges_rev = torch.tensor([[1,0],[3,2],[4,1]], dtype=torch.long)
scores_fwd = {fn}(A, X, edges_fwd)
scores_rev = {fn}(A, X, edges_rev)
assert torch.allclose(scores_fwd, scores_rev, atol=1e-5), f'score(i,j) should equal score(j,i): max diff = {(scores_fwd - scores_rev).abs().max().item():.6f}'""",
        },
    ],
    'solution': '''def link_prediction(A, X, edges):
    """使用两层图卷积执行链接预测 (GCN Link Prediction)。"""
    N, F_in = X.shape

    # Step 1: 固定随机权重并构造对称归一化邻接矩阵
    torch.manual_seed(0)
    W1 = torch.randn(F_in, 16, dtype=X.dtype, device=X.device) * 0.1
    W2 = torch.randn(16, 8, dtype=X.dtype, device=X.device) * 0.1
    adjacency = A + torch.eye(N, dtype=A.dtype, device=A.device)
    inv_sqrt_degree = adjacency.sum(dim=1).pow(-0.5)
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )

    # Step 2: 通过两层 GCN 生成节点嵌入
    hidden = torch.relu(normalized_adjacency @ X @ W1)
    embeddings = normalized_adjacency @ hidden @ W2

    # Step 3: 对节点对做点积并映射为概率
    source = embeddings[edges[:, 0]]
    target = embeddings[edges[:, 1]]
    scores = torch.sigmoid((source * target).sum(dim=1))
    return scores''',
    'demo': """torch.manual_seed(42)
N, F_in = 5, 4
A = torch.eye(N).roll(1, dims=0) + torch.eye(N).roll(-1, dims=0)
X = torch.randn(N, F_in)
edges = torch.tensor([[0, 1], [0, 2], [3, 4]])
probs = link_prediction(A, X, edges)
print("Output shape:", probs.shape)
print("Edge probabilities:", probs)
print("Probability range:", (probs.min().item(), probs.max().item()))""",
    'explanation': r"""### 核心思路
先计算 $\hat A=D^{-1/2}(A+I)D^{-1/2}$，再用两层 GCN 得到节点嵌入 `embeddings`，最后以节点对内积的 sigmoid 作为边概率。数据由 `(N, F)` 节点特征流向 `(M,)` 预测。

### 代码步骤拆解
- **Step 1（图归一化）**：加入自环后由度向量广播构造 `normalized_adjacency`。
- **Step 2（节点编码）**：第一层 ReLU，第二层保持线性得到 8 维嵌入。
- **Step 3（边解码）**：索引两端节点，逐维乘积求和并 sigmoid。

### 重点与易错点分析
- **注意点 1（确定性）**：函数内部必须在创建 `W1`、`W2` 前设置固定随机种子。
- **注意点 2（设备对齐）**：单位阵与随机权重需跟随输入 device 和 dtype。
- **注意点 3（对称性）**：点积解码满足 $s(i,j)=s(j,i)$；若改为拼接 MLP 则不自动具备该性质。

### 复杂度分析
- **时间复杂度**：稠密实现为 $O(N^2F+MF_z)$，GCN 邻接乘法是瓶颈。
- **空间复杂度**：$O(N^2+NF_z)$，主要保存稠密归一化邻接与节点激活。""",
}

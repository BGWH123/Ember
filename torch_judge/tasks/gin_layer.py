"""GIN Layer (Graph Isomorphism Network) task."""

TASK = {
    'title': 'GIN Layer (Graph Isomorphism Network)',
    'title_zh': 'GIN 层（图同构网络）',
    'difficulty': 'Medium',
    'category': '图神经网络',
    'function_name': 'gin_layer',
    'starter_code': """def gin_layer(A, X, eps, W1, b1, W2, b2):
    pass""",
    'hint': """Aggregate with `A @ X`, combine as `(1 + eps) * X + agg`, then pass through a 2-layer MLP with ReLU.""",
    'hint_zh': '用 `A @ X` 聚合邻居，组合为 `(1 + eps) * X + agg`，然后通过带 ReLU 的两层 MLP。',
    'description_en': """Implement a Graph Isomorphism Network (GIN) layer.

GIN updates node features by aggregating neighbor features, scaling the node's own features by (1 + eps), and passing through a 2-layer MLP.

**Signature:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F_in)
- `eps` — learnable scalar tensor
- `W1` — first linear weight (F_in, F_hidden)
- `b1` — first linear bias (F_hidden,)
- `W2` — second linear weight (F_hidden, F_out)
- `b2` — second linear bias (F_out,)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. Aggregate neighbor features: `agg = A @ X`
2. Combine: `h = (1 + eps) * X + agg`
3. MLP layer 1: `h = ReLU(h @ W1 + b1)`
4. MLP layer 2: `h = h @ W2 + b2`
5. Return h""",
    'description_zh': """实现图同构网络（GIN）层。

GIN 通过聚合邻居特征、用 (1 + eps) 缩放自身特征，再经过两层 MLP 来更新节点表示。

**签名:** `gin_layer(A, X, eps, W1, b1, W2, b2) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F_in)
- `eps` — 可学习标量张量
- `W1` — 第一层线性权重 (F_in, F_hidden)
- `b1` — 第一层线性偏置 (F_hidden,)
- `W2` — 第二层线性权重 (F_hidden, F_out)
- `b2` — 第二层线性偏置 (F_out,)

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 聚合邻居特征：`agg = A @ X`
2. 组合：`h = (1 + eps) * X + agg`
3. MLP 第一层：`h = ReLU(h @ W1 + b1)`
4. MLP 第二层：`h = h @ W2 + b2`
5. 返回 h""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_hid, F_out = 5, 8, 16, 4
A = torch.ones(N, N)
X = torch.randn(N, F_in)
eps = torch.tensor(0.0)
W1 = torch.randn(F_in, F_hid)
b1 = torch.randn(F_hid)
W2 = torch.randn(F_hid, F_out)
b2 = torch.randn(F_out)
out = {fn}(A, X, eps, W1, b1, W2, b2)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'""",
        },
        {
            "name": 'eps=0 behavior',
            "code": """import torch
torch.manual_seed(10)
N, F_in, F_hid, F_out = 3, 4, 8, 2
A = torch.eye(N)
X = torch.randn(N, F_in)
eps = torch.tensor(0.0)
W1 = torch.randn(F_in, F_hid)
b1 = torch.zeros(F_hid)
W2 = torch.randn(F_hid, F_out)
b2 = torch.zeros(F_out)
out = {fn}(A, X, eps, W1, b1, W2, b2)
h = X + A @ X
h = torch.relu(h @ W1 + b1)
ref = h @ W2 + b2
assert torch.allclose(out, ref, atol=1e-5), f'eps=0 with identity A should give (1+0)*X + I@X = 2X through MLP'""",
        },
        {
            "name": 'Exact numerical value',
            "code": r"""import torch
torch.manual_seed(77)
A = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
eps = torch.tensor(0.5)
W1 = torch.tensor([[1.0, -1.0, 0.5],[0.5, 1.0, -0.5]])
b1 = torch.tensor([0.0, 0.0, 0.0])
W2 = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
b2 = torch.tensor([0.0, 0.0])
out = {fn}(A, X, eps, W1, b1, W2, b2)
agg = A @ X
h = (1 + eps) * X + agg
h = torch.relu(h @ W1 + b1)
ref = h @ W2 + b2
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\n{out}\nvs\n{ref}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_hid, F_out = 4, 6, 8, 3
A = torch.ones(N, N)
X = torch.randn(N, F_in, requires_grad=True)
eps = torch.tensor(0.1, requires_grad=True)
W1 = torch.randn(F_in, F_hid, requires_grad=True)
b1 = torch.randn(F_hid, requires_grad=True)
W2 = torch.randn(F_hid, F_out, requires_grad=True)
b2 = torch.randn(F_out, requires_grad=True)
out = {fn}(A, X, eps, W1, b1, W2, b2)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert eps.grad is not None, 'eps.grad is None'
assert W1.grad is not None, 'W1.grad is None'
assert W2.grad is not None, 'W2.grad is None'""",
        },
        {
            "name": 'Isolated node (no neighbors)',
            "code": """import torch
torch.manual_seed(55)
A = torch.tensor([[0,0,0],[0,0,1],[0,1,0]], dtype=torch.float)
X = torch.tensor([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]])
eps = torch.tensor(0.0)
W1 = torch.ones(2, 2)
b1 = torch.zeros(2)
W2 = torch.ones(2, 1)
b2 = torch.zeros(1)
out = {fn}(A, X, eps, W1, b1, W2, b2)
h0 = (1 + 0.0) * X[0] + torch.zeros(2)
h0 = torch.relu(h0 @ W1[:, :] + b1)
ref0 = h0 @ W2 + b2
assert torch.allclose(out[0], ref0, atol=1e-5), f'Isolated node output wrong: {out[0]} vs {ref0}'""",
        },
        {
            "name": 'Different dimensions',
            "code": """import torch
torch.manual_seed(33)
for F_in, F_hid, F_out in [(2, 4, 1), (8, 16, 8), (3, 6, 2)]:
    N = 4
    A = (torch.rand(N, N) > 0.3).float()
    X = torch.randn(N, F_in)
    eps = torch.tensor(0.1)
    W1 = torch.randn(F_in, F_hid)
    b1 = torch.randn(F_hid)
    W2 = torch.randn(F_hid, F_out)
    b2 = torch.randn(F_out)
    out = {fn}(A, X, eps, W1, b1, W2, b2)
    assert out.shape == (N, F_out), f'Shape mismatch for dims ({F_in},{F_hid},{F_out}): {out.shape}'""",
        },
    ],
    'solution': '''def gin_layer(A, X, eps, W1, b1, W2, b2):
    """计算图同构网络层 (GIN Layer)。"""
    # Step 1: 聚合邻居并保留加权自身特征
    neighbor_sum = A @ X
    combined = (1.0 + eps) * X + neighbor_sum

    # Step 2: 通过第一层 MLP 和 ReLU
    hidden = torch.relu(combined @ W1 + b1)

    # Step 3: 通过第二层 MLP 得到节点输出
    return hidden @ W2 + b2''',
    'demo': """torch.manual_seed(42)
N, F_in, F_hidden, F_out = 5, 4, 8, 3
A = torch.ones(N, N) - torch.eye(N)
X = torch.randn(N, F_in)
out = gin_layer(A, X, torch.tensor(0.1), torch.randn(F_in, F_hidden), torch.zeros(F_hidden), torch.randn(F_hidden, F_out), torch.zeros(F_out))
print("Output shape:", out.shape)
print("Output mean:", out.mean().item())
print("Sample values:", out[0])""",
    'explanation': r"""### 核心思路
GIN 先计算 $(1+\epsilon)X+AX$，显式区分自身与邻居贡献，再用两层 MLP 提升图结构判别能力。

### 代码步骤拆解
- **Step 1（特征聚合）**：`neighbor_sum` 和 `combined` 均为 `(N, F_in)`。
- **Step 2（隐藏映射）**：`combined @ W1 + b1` 变为 `(N, F_hidden)` 并经过 ReLU。
- **Step 3（输出映射）**：第二层映射得到 `(N, F_out)`。

### 重点与易错点分析
- **注意点 1（自环语义）**：公式已单独加入自身项，`A` 是否含自环会直接改变结果。
- **注意点 2（偏置广播）**：`b1`、`b2` 沿节点维广播，长度必须匹配对应输出维。
- **注意点 3（可学习 eps）**：不要把 `eps` 转成 Python 数值，否则会切断其梯度。

### 复杂度分析
- **时间复杂度**：$O(N^2F_{in}+NF_{in}F_h+NF_hF_{out})$。
- **空间复杂度**：$O(NF_h+NF_{out})$，不计输入邻接矩阵。""",
}

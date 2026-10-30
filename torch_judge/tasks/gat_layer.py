"""GAT Layer (Graph Attention) task."""

TASK = {
    'title': 'GAT Layer (Graph Attention)',
    'title_zh': 'GAT 层（图注意力）',
    'difficulty': 'Medium',
    'category': '图神经网络',
    'function_name': 'gat_layer',
    'starter_code': """def gat_layer(A, X, W, a, negative_slope=0.2):
    pass""",
    'hint': """Compute e as (N,N) via broadcasting: `e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`, then mask with A and softmax.""",
    'hint_zh': '利用广播计算 e 矩阵：`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`，然后用 A 掩码并 softmax。',
    'description_en': """Implement a single-head Graph Attention Network (GAT) layer.

GAT computes attention coefficients between connected nodes, then aggregates neighbor features weighted by those coefficients.

**Signature:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary, already includes self-loops
- `X` — node feature matrix (N, F_in)
- `W` — weight matrix (F_in, F_out)
- `a` — attention vector (2*F_out,)
- `negative_slope` — LeakyReLU negative slope (default 0.2)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. Project features: `H = X @ W` → (N, F_out)
2. Compute attention scores: `e_ij = LeakyReLU(a[:F_out]·H[i] + a[F_out:]·H[j])`
3. Mask non-edges: set `e[i,j] = -inf` where `A[i,j] = 0`
4. Normalize: `alpha = softmax(e, dim=-1)` over neighbors
5. Aggregate: `output = alpha @ H`

**Hint:** Compute e efficiently as an (N, N) matrix using broadcasting:
`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`""",
    'description_zh': """实现单头图注意力网络（GAT）层。

GAT 计算相连节点之间的注意力系数，然后用这些系数对邻居特征进行加权聚合。

**签名:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值，已包含自环
- `X` — 节点特征矩阵 (N, F_in)
- `W` — 权重矩阵 (F_in, F_out)
- `a` — 注意力向量 (2*F_out,)
- `negative_slope` — LeakyReLU 负斜率（默认 0.2）

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 特征投影：`H = X @ W` → (N, F_out)
2. 计算注意力分数：`e_ij = LeakyReLU(a[:F_out]·H[i] + a[F_out:]·H[j])`
3. 掩码非边：将 `A[i,j] = 0` 处的 `e[i,j]` 设为 `-inf`
4. 归一化：`alpha = softmax(e, dim=-1)` 对邻居求 softmax
5. 聚合：`output = alpha @ H`

**提示:** 利用广播高效计算 e 矩阵：
`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 8, 4
A = torch.ones(N, N)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out = {fn}(A, X, W, a)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'""",
        },
        {
            "name": 'Attention weights sum to 1 (via output verification)',
            "code": """import torch
torch.manual_seed(0)
N, F_in, F_out = 4, 6, 3
A = torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]], dtype=torch.float)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out = {fn}(A, X, W, a)
# Verify output matches expected (which implies correct attention normalization)
H = X @ W
F_o = F_out
e = (H @ a[:F_o]).unsqueeze(1) + (H @ a[F_o:]).unsqueeze(0)
e = torch.nn.functional.leaky_relu(e, 0.2)
e = e.masked_fill(A == 0, float('-inf'))
alpha = torch.softmax(e, dim=-1)
expected = alpha @ H
assert torch.allclose(out, expected, atol=1e-5), f'Output mismatch — attention normalization may be wrong'""",
        },
        {
            "name": 'Exact numerical value (small graph)',
            "code": r"""import torch
torch.manual_seed(123)
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
W = torch.tensor([[0.5, -0.5],[0.5, 0.5]])
a = torch.tensor([1.0, 0.0, 0.0, 1.0])
out = {fn}(A, X, W, a, negative_slope=0.2)
H = X @ W
F_o = F_out
e = (H @ a[:F_o]).unsqueeze(1) + (H @ a[F_o:]).unsqueeze(0)
e = torch.nn.functional.leaky_relu(e, 0.2)
e = e.masked_fill(A == 0, float('-inf'))
alpha = torch.softmax(e, dim=-1)
ref = alpha @ H
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\n{out}\nvs\n{ref}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(42)
N, F_in, F_out = 4, 6, 3
A = torch.ones(N, N)
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(F_in, F_out, requires_grad=True)
a = torch.randn(2 * F_out, requires_grad=True)
out = {fn}(A, X, W, a)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert W.grad is not None, 'W.grad is None'
assert a.grad is not None, 'a.grad is None'""",
        },
        {
            "name": 'Different graph sizes',
            "code": """import torch
torch.manual_seed(7)
for N in [2, 8, 16]:
    F_in, F_out = 4, 3
    A = (torch.rand(N, N) > 0.5).float()
    A = A + torch.eye(N)
    A = (A > 0).float()
    X = torch.randn(N, F_in)
    W = torch.randn(F_in, F_out)
    a = torch.randn(2 * F_out)
    out = {fn}(A, X, W, a)
    assert out.shape == (N, F_out), f'Shape mismatch for N={N}: {out.shape}'""",
        },
        {
            "name": 'Custom negative_slope',
            "code": """import torch
torch.manual_seed(99)
N, F_in, F_out = 3, 4, 2
A = torch.ones(N, N)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out1 = {fn}(A, X, W, a, negative_slope=0.01)
out2 = {fn}(A, X, W, a, negative_slope=0.5)
assert not torch.allclose(out1, out2, atol=1e-6), 'Different negative_slope should produce different outputs'""",
        },
    ],
    'solution': '''def gat_layer(A, X, W, a, negative_slope=0.2):
    """计算单头图注意力层 (Graph Attention Layer)。"""
    # Step 1: 投影节点特征并拆分注意力向量
    hidden = X @ W
    F_out = hidden.shape[-1]
    source_scores = hidden @ a[:F_out]
    target_scores = hidden @ a[F_out:]

    # Step 2: 构造边分数并屏蔽非边位置
    scores = source_scores[:, None] + target_scores[None, :]
    scores = F.leaky_relu(scores, negative_slope=negative_slope)
    scores = scores.masked_fill(A == 0, float("-inf"))

    # Step 3: 归一化注意力并聚合邻居特征
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ hidden''',
    'demo': """torch.manual_seed(42)
N, F_in, F_out = 5, 8, 4
A = torch.ones(N, N)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out = gat_layer(A, X, W, a)
print("Output shape:", out.shape)
print("Output norm:", out.norm().item())
print("Sample values:", out[0])""",
    'explanation': """### 核心思路
将节点特征投影为 $H=XW$，通过加性打分构造 $e_{ij}$，仅在有效边上做 softmax，再以注意力权重聚合邻居表示。

### 代码步骤拆解
- **Step 1（特征投影）**：`hidden` 为 `(N, F_out)`，`a` 被切分为源节点与目标节点两部分。
- **Step 2（边分数与掩码）**：两个一维分数广播成 `(N, N)` 的 `scores`，非边位置设为负无穷。
- **Step 3（邻居聚合）**：`attn_weights @ hidden` 返回 `(N, F_out)` 的节点表示。

### 重点与易错点分析
- **注意点 1（掩码顺序）**：必须在 softmax 前屏蔽非边，否则非边会分走概率质量；每行应至少有一个有效邻居。
- **注意点 2（广播方向）**：`source_scores[:, None]` 对应行节点，`target_scores[None, :]` 对应列节点。
- **注意点 3（梯度路径）**：全程使用张量运算，梯度可流向 `X`、`W` 与 `a`。

### 复杂度分析
- **时间复杂度**：$O(NF_{in}F_{out}+N^2F_{out})$，稠密注意力聚合占主导。
- **空间复杂度**：$O(N^2+NF_{out})$，主要保存分数和注意力权重。""",
}

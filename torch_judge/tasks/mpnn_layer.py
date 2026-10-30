"""MPNN Layer (Message Passing) task."""

TASK = {
    'title': 'MPNN Layer (Message Passing)',
    'title_zh': 'MPNN 层（消息传递）',
    'difficulty': 'Medium',
    'category': '图神经网络',
    'function_name': 'mpnn_layer',
    'starter_code': """def mpnn_layer(A, X, E, W_msg, W_upd):
    pass""",
    'hint': """Build (N, N, 2F+D_e) by broadcasting `X[i]` and `X[j]` with `E[i,j]`, matmul with W_msg, mask by A, sum over dim=1.""",
    'hint_zh': '利用广播构建 (N, N, 2F+D_e) 张量，拼接 `X[i]`、`X[j]` 和 `E[i,j]`，与 W_msg 矩阵乘，用 A 掩码后沿 dim=1 求和。',
    'description_en': """Implement a Message Passing Neural Network (MPNN) layer.

MPNN computes messages along edges using node and edge features, aggregates them per node, then updates node representations.

**Signature:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`

**Parameters:**
- `A` — adjacency matrix (N, N), binary
- `X` — node feature matrix (N, F)
- `E` — edge feature tensor (N, N, D_e)
- `W_msg` — message weight matrix (2*F + D_e, F_msg)
- `W_upd` — update weight matrix (F + F_msg, F_out)

**Returns:** updated node features (N, F_out)

**Algorithm:**
1. For each edge (i,j) where A[i,j]=1, compute message: `m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`
2. For each node i, aggregate: `agg_i = sum of m_ij over neighbors j`
3. For each node i, update: `out_i = ReLU(concat(X[i], agg_i) @ W_upd)`

**Hint:** Build an (N, N, 2F+D_e) tensor by broadcasting X, then matmul with W_msg, mask by A, and sum over dim=1.""",
    'description_zh': """实现消息传递神经网络（MPNN）层。

MPNN 利用节点和边特征沿边计算消息，按节点聚合后更新节点表示。

**签名:** `mpnn_layer(A, X, E, W_msg, W_upd) -> Tensor`

**参数:**
- `A` — 邻接矩阵 (N, N)，二值
- `X` — 节点特征矩阵 (N, F)
- `E` — 边特征张量 (N, N, D_e)
- `W_msg` — 消息权重矩阵 (2*F + D_e, F_msg)
- `W_upd` — 更新权重矩阵 (F + F_msg, F_out)

**返回:** 更新后的节点特征 (N, F_out)

**算法:**
1. 对每条边 (i,j)（A[i,j]=1），计算消息：`m_ij = ReLU(concat(X[i], X[j], E[i,j]) @ W_msg)`
2. 对每个节点 i，聚合：`agg_i = 对邻居 j 的 m_ij 求和`
3. 对每个节点 i，更新：`out_i = ReLU(concat(X[i], agg_i) @ W_upd)`

**提示:** 利用广播构建 (N, N, 2F+D_e) 张量，与 W_msg 矩阵乘，用 A 掩码后沿 dim=1 求和。""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(42)
N, F, D_e, F_msg, F_out = 5, 4, 3, 8, 6
A = (torch.rand(N, N) > 0.5).float()
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'""",
        },
        {
            "name": 'Exact numerical value',
            "code": r"""import torch
torch.manual_seed(100)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
E = torch.zeros(N, N, D_e)
E[0,1,0] = 0.5; E[1,0,0] = -0.5; E[1,2,0] = 1.0; E[2,1,0] = -1.0
E[0,0,0] = 0.1; E[1,1,0] = 0.2; E[2,2,0] = 0.3
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
# Reference
Xi = X.unsqueeze(1).expand(N, N, F)
Xj = X.unsqueeze(0).expand(N, N, F)
inp = torch.cat([Xi, Xj, E], dim=-1)
msgs = torch.relu(inp @ W_msg)
msgs = msgs * A.unsqueeze(-1)
agg = msgs.sum(dim=1)
upd_inp = torch.cat([X, agg], dim=-1)
ref = torch.relu(upd_inp @ W_upd)
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\n{out}\nvs\n{ref}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
torch.manual_seed(42)
N, F, D_e, F_msg, F_out = 4, 3, 2, 6, 3
A = torch.ones(N, N)
X = torch.randn(N, F, requires_grad=True)
E = torch.randn(N, N, D_e, requires_grad=True)
W_msg = torch.randn(2 * F + D_e, F_msg, requires_grad=True)
W_upd = torch.randn(F + F_msg, F_out, requires_grad=True)
out = {fn}(A, X, E, W_msg, W_upd)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert E.grad is not None, 'E.grad is None'
assert W_msg.grad is not None, 'W_msg.grad is None'
assert W_upd.grad is not None, 'W_upd.grad is None'""",
        },
        {
            "name": 'Single edge graph',
            "code": """import torch
torch.manual_seed(7)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.zeros(N, N)
A[0, 1] = 1.0
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
# Node 0 has one outgoing edge to node 1
msg_01 = torch.relu(torch.cat([X[0], X[1], E[0,1]]) @ W_msg)
agg_0 = msg_01
ref_0 = torch.relu(torch.cat([X[0], agg_0]) @ W_upd)
assert torch.allclose(out[0], ref_0, atol=1e-5), f'Node 0 mismatch: {out[0]} vs {ref_0}'
# Node 2 has no neighbors
agg_2 = torch.zeros(F_msg)
ref_2 = torch.relu(torch.cat([X[2], agg_2]) @ W_upd)
assert torch.allclose(out[2], ref_2, atol=1e-5), f'Node 2 (no neighbors) mismatch: {out[2]} vs {ref_2}'""",
        },
        {
            "name": 'Symmetric graph behavior',
            "code": """import torch
torch.manual_seed(88)
N, F, D_e, F_msg, F_out = 4, 3, 2, 5, 3
A = torch.ones(N, N)
X = torch.randn(N, F)
E_sym = torch.randn(N, N, D_e)
E_sym = (E_sym + E_sym.transpose(0, 1)) / 2
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E_sym, W_msg, W_upd)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape}'
assert not torch.isnan(out).any(), 'Output contains NaN'
assert not torch.isinf(out).any(), 'Output contains Inf'""",
        },
        {
            "name": 'No edges (empty graph)',
            "code": r"""import torch
torch.manual_seed(11)
N, F, D_e, F_msg, F_out = 3, 2, 1, 4, 2
A = torch.zeros(N, N)
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, F_msg)
W_upd = torch.randn(F + F_msg, F_out)
out = {fn}(A, X, E, W_msg, W_upd)
agg = torch.zeros(N, F_msg)
ref = torch.relu(torch.cat([X, agg], dim=-1) @ W_upd)
assert torch.allclose(out, ref, atol=1e-5), f'Empty graph mismatch:\n{out}\nvs\n{ref}'""",
        },
    ],
    'solution': '''def mpnn_layer(A, X, E, W_msg, W_upd):
    """MPNN 层（消息传递） (MPNN Layer (Message Passing))。"""
    # Step 1: 广播源节点与邻居节点特征，并拼接每条候选边的属性
    N, F = X.shape
    x_source = X.unsqueeze(1).expand(N, N, F)
    x_target = X.unsqueeze(0).expand(N, N, F)
    message_inputs = torch.cat([x_source, x_target, E], dim=-1)
    # Step 2: 生成逐边 ReLU 消息，用邻接矩阵屏蔽非边后按邻居求和
    messages = torch.relu(message_inputs @ W_msg)
    messages = messages * A.unsqueeze(-1)
    aggregated_messages = messages.sum(dim=1)
    # Step 3: 将节点原表示与聚合消息拼接，经更新矩阵和 ReLU 得到新表示
    return torch.relu(torch.cat([X, aggregated_messages], dim=-1) @ W_upd)''',
    'demo': """torch.manual_seed(42)
N, F, D_e = 4, 3, 2
A = torch.ones(N, N)
X = torch.randn(N, F)
E = torch.randn(N, N, D_e)
W_msg = torch.randn(2 * F + D_e, 5)
W_upd = torch.randn(F + 5, 4)
out = mpnn_layer(A, X, E, W_msg, W_upd)
print('节点/输出维度:', N, tuple(out.shape))
print('节点 0 聚合更新:', out[0].tolist())
print('激活非零比例:', f'{(out > 0).float().mean().item():.2%}')""",
    'explanation': r"""### 核心思路
该层把每个有向边 $(i,j)$ 的源节点、邻居节点和边属性联合编码：$m_{ij}=\operatorname{ReLU}([x_i,x_j,e_{ij}]W_{\mathrm{msg}})$。邻接矩阵决定哪些候选消息有效，节点 $i$ 聚合 $a_i=\sum_j A_{ij}m_{ij}$，再用 $x_i'=\operatorname{ReLU}([x_i,a_i]W_{\mathrm{upd}})$ 完成一次消息传递。

### 代码步骤拆解
- **Step 1（展开有向节点对）**：`x_source[i,j]=X[i]`、`x_target[i,j]=X[j]`，与 `E[i,j]` 拼成 `(N,N,2F+D_e)` 的 `message_inputs`。
- **Step 2（边消息与求和聚合）**：乘 `W_msg` 得到 `(N,N,F_msg)`，ReLU 后乘 `A[...,None]` 清除不存在边，沿 `dim=1` 汇总每个 i 的邻居 j。
- **Step 3（节点状态更新）**：将 `(N,F)` 的 `X` 与 `(N,F_msg)` 的 `aggregated_messages` 拼接，乘 `W_upd` 输出 `(N,F_out)`。

### 重点与易错点分析
- **注意点 1（边方向约定）**：本题定义 `A[i,j]` 为节点 i 从邻居 j 聚合的边，因此沿 `dim=1` 求和；交换两个广播张量会改变有向消息函数。
- **注意点 2（掩码位置）**：先计算候选消息再乘 A 与参考公式一致；A 为零的边不会贡献输出，也不会向相应 `E[i,j]` 传播梯度。
- **注意点 3（孤立节点行为）**：无邻居节点的聚合向量严格为零，但其输出不一定为零，因为更新仍包含自身特征 `X[i]`；这里是 sum 聚合，不会按度数归一化。

### 复杂度分析
- **时间复杂度**：稠密构造下为 $O(N^2(2F+D_e)F_{\mathrm{msg}}+N(F+F_{\mathrm{msg}})F_{\mathrm{out}})$，逐边消息投影占主导。
- **空间复杂度**：$O(N^2(2F+D_e+F_{\mathrm{msg}}))$，节点对拼接张量和全部候选消息是主要开销。""",
}

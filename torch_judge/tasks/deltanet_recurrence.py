"""DeltaNet Linear Attention Recurrence task."""

TASK = {
    'title': 'DeltaNet Linear Attention Recurrence',
    'title_zh': 'DeltaNet 线性注意力递推',
    'difficulty': 'Hard',
    'category': '注意力机制',
    'function_name': 'deltanet_recurrence',
    'starter_code': """def deltanet_recurrence(q, k, v, beta, eps=1e-6):
    pass""",
    'hint': """Normalize q/k once, then at each step compute `v_old=S@k`, apply the gated outer-product delta update, and read with `S@q`.""",
    'hint_zh': '先归一化 q/k；每一步计算 `v_old=S@k`，执行门控外积增量更新，再用 `S@q` 读取。',
    'description_en': """Implement the recurrent DeltaNet state update and return every readout together with the final matrix-valued state.""",
    'description_zh': '实现 DeltaNet 的循环状态更新，返回所有时间步输出以及最终矩阵状态。',
    'theory_en': """The state is an associative memory mapping normalized keys to values. The delta rule first removes the old prediction at a key, then writes the residual toward the new value.""",
    'theory_zh': '状态是将归一化键映射到值的联想记忆。增量规则先取得该键的旧预测，再把预测残差写入新值方向。',
    'diagram_en': """```text
S,k -> v_old -> beta*(v-v_old) outer k -> S' -> S'@q
```""",
    'diagram_zh': """```text
S,k -> 旧值 -> beta*(v-旧值) 外积 k -> S' -> S'@q
```""",
    "tests": [
        {
            "name": 'output and state shapes',
            "code": """import torch
q = torch.randn(2, 5, 4)
k = torch.randn(2, 5, 4)
v = torch.randn(2, 5, 4)
out, state = {fn}(q, k, v, torch.full((2, 5), 0.5))
assert out.shape == (2, 5, 4) and state.shape == (2, 4, 4)""",
        },
        {
            "name": 'numerical recurrence',
            "code": """import torch
torch.manual_seed(1)
q, k, v = [torch.randn(1, 3, 2) for _ in range(3)]
beta = torch.tensor([[0.2, 0.7, 0.4]])
out, state = {fn}(q, k, v, beta)
qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
s = torch.zeros(1, 2, 2)
ys = []
for t in range(3):
    old = (s @ kn[:, t, :, None]).squeeze(-1)
    s = s + beta[:, t, None, None] * (v[:, t] - old)[:, :, None] * kn[:, t, None, :]
    ys.append((s @ qn[:, t, :, None]).squeeze(-1))
ref = torch.stack(ys, 1)
assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(state, s, atol=1e-6)""",
        },
        {
            "name": 'zero gate edge case',
            "code": """import torch
q, k, v = [torch.randn(2, 4, 3) for _ in range(3)]
out, state = {fn}(q, k, v, torch.zeros(2, 4))
assert torch.count_nonzero(out) == 0 and torch.count_nonzero(state) == 0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q = torch.randn(1, 4, 3, requires_grad=True)
k = torch.randn(1, 4, 3, requires_grad=True)
v = torch.randn(1, 4, 3, requires_grad=True)
beta = torch.full((1, 4), 0.5, requires_grad=True)
out, state = {fn}(q, k, v, beta)
(out.square().sum() + state.square().sum()).backward()
assert all(t.grad is not None for t in (q, k, v, beta))""",
        },
    ],
    'solution': '''def deltanet_recurrence(q, k, v, beta, eps=1e-6):
    """执行 DeltaNet 线性注意力递推 (DeltaNet Recurrence)。"""
    B, T, D = q.shape

    # Step 1: 沿特征维归一化查询和键
    query_norm = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
    key_norm = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
    state = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)

    # Step 2: 逐时间步移除旧预测并写入门控残差
    outputs = []
    for t in range(T):
        query_t = query_norm[:, t]
        key_t = key_norm[:, t]
        value_t = v[:, t]
        old_value = (state @ key_t.unsqueeze(-1)).squeeze(-1)
        value_delta = (value_t - old_value).unsqueeze(-1)
        state = state + beta[:, t, None, None] * value_delta * key_t.unsqueeze(-2)

        # Step 3: 用更新后的状态读取当前查询
        outputs.append((state @ query_t.unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=1), state''',
    'demo': """torch.manual_seed(42)
B, T, D = 2, 10, 16
q, k, v = [torch.randn(B, T, D) for _ in range(3)]
out, final_state = deltanet_recurrence(q, k, v, torch.sigmoid(torch.randn(B, T)))
print("Output shape:", out.shape)
print("Final state shape:", final_state.shape)""",
    'explanation': r"""### 核心思路
DeltaNet 将 `(D, D)` 状态视为键到值的联想记忆，每步用 $S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top$ 写入残差，再读取 $S_tq_t$。

### 代码步骤拆解
- **Step 1（向量归一化）**：将 `(B, T, D)` 的 `q`、`k` 沿最后一维归一化，保留原始形状。
- **Step 2（增量写入）**：由 `(B, D, D)` 状态预测 `old_value`，再通过批量外积写入门控 `value_delta`。
- **Step 3（状态读取）**：用更新后的状态乘当前 `query_t`，堆叠为 `(B, T, D)` 输出。

### 重点与易错点分析
- **注意点 1（归一化稳定性）**：范数后加 `eps`，零向量不会产生 NaN；`keepdim=True` 保证广播正确。
- **注意点 2（更新顺序）**：必须先写后读，否则输出对应的是上一时刻状态，递推语义会偏移一步。
- **注意点 3（自动微分）**：使用重新赋值更新 `state`，避免对参与计算图的张量执行原地修改。

### 复杂度分析
- **时间复杂度**：$O(BTD^2)$，每步状态—向量乘法和外积是主要瓶颈。
- **空间复杂度**：$O(BD^2+BTD)$，保存矩阵状态与全部时间步输出。""",
}

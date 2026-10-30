"""DeepSeek-V3 Aux-Loss-Free MoE Router task."""

TASK = {
    'title': 'DeepSeek-V3 Aux-Loss-Free MoE Router',
    'title_zh': 'DeepSeek-V3 无辅助损失 MoE 路由',
    'difficulty': 'Hard',
    'category': '混合专家模型',
    'function_name': 'aux_free_moe_route',
    'starter_code': """def aux_free_moe_route(x, router_weight, expert_bias, top_k, bias_lr=1e-3, eps=1e-6):
    pass""",
    'hint': """Rank sigmoid affinities after adding the dynamic bias, but gather and normalize the original unbiased affinities. Use bincount for expert loads.""",
    'hint_zh': '在 sigmoid 亲和度上加入动态偏置后排序，但应收集并归一化原始无偏亲和度；用 bincount 统计专家负载。',
    'description_en': """Implement DeepSeek-V3-style auxiliary-loss-free routing. Select experts with bias-adjusted sigmoid scores, compute gates from unbiased scores, and update the per-expert bias from load imbalance.""",
    'description_zh': '实现 DeepSeek-V3 风格的无辅助损失路由：用加偏置的 sigmoid 分数选择专家，用无偏分数计算门控，并根据负载不均衡更新专家偏置。',
    'theory_en': """A non-differentiable dynamic bias steers future assignments toward underloaded experts without changing the gate values used by the current token.""",
    'theory_zh': '不可微的动态偏置将后续分配引向低负载专家，同时不改变当前 token 实际使用的门控权重。',
    'diagram_en': r"""```text
x @ W^T -> sigmoid -> + bias -> top-k indices
                    \-> gather unbiased scores -> normalize -> gates
indices -> expert counts -> signed bias update
```""",
    'diagram_zh': r"""```text
x @ W^T -> sigmoid -> + 偏置 -> top-k 索引
                    \-> 收集无偏分数 -> 归一化 -> 门控
索引 -> 专家计数 -> 符号偏置更新
```""",
    "tests": [
        {
            "name": 'shapes and normalized gates',
            "code": """import torch
x = torch.randn(5, 4)
w = torch.randn(3, 4)
b = torch.zeros(3)
idx, gates, new_b = {fn}(x, w, b, top_k=2)
assert idx.shape == gates.shape == (5, 2)
assert new_b.shape == (3,)
assert torch.allclose(gates.sum(-1), torch.ones(5), atol=2e-6)""",
        },
        {
            "name": 'exact routing reference',
            "code": """import torch
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
w = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])
b = torch.tensor([0.0, 0.2, -0.1])
idx, gates, new_b = {fn}(x, w, b, 2, bias_lr=0.05)
s = torch.sigmoid(x @ w.T)
ref_idx = torch.topk(s + b.unsqueeze(0), 2, dim=-1).indices
sel = torch.gather(s, -1, ref_idx)
ref_g = sel / (sel.sum(-1, keepdim=True) + 1e-6)
counts = torch.bincount(ref_idx.reshape(-1), minlength=3).to(s.dtype)
ref_b = b - 0.05 * torch.sign(counts - 2.0)
assert torch.equal(idx, ref_idx)
assert torch.allclose(gates, ref_g, atol=1e-7)
assert torch.equal(new_b, ref_b)""",
        },
        {
            "name": 'all experts edge case',
            "code": """import torch
torch.manual_seed(4)
x = torch.randn(2, 3)
w = torch.randn(4, 3)
b = torch.randn(4)
idx, gates, new_b = {fn}(x, w, b, top_k=4, bias_lr=0.3)
assert torch.equal(torch.sort(idx, dim=-1).values, torch.arange(4).expand(2, 4))
assert torch.equal(new_b, b)
assert torch.all(gates > 0)""",
        },
        {
            "name": 'gradient flow through gates',
            "code": """import torch
x = torch.randn(6, 5, requires_grad=True)
w = torch.randn(4, 5, requires_grad=True)
b = torch.zeros(4)
idx, gates, new_b = {fn}(x, w, b, 2)
(gates.square().sum()).backward()
assert idx.dtype == torch.long and new_b.shape == b.shape
assert x.grad is not None and w.grad is not None
assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()""",
        },
    ],
    'solution': '''def aux_free_moe_route(x, router_weight, expert_bias, top_k, bias_lr=1e-3, eps=1e-6):
    """执行无辅助损失的混合专家路由 (Aux-Loss-Free MoE Routing)。"""
    # Step 1: 计算无偏亲和度，并仅用动态偏置选择专家
    affinities = torch.sigmoid(x @ router_weight.T)
    selection_scores = affinities + expert_bias.unsqueeze(0)
    expert_indices = torch.topk(selection_scores, k=top_k, dim=-1).indices

    # Step 2: 收集原始亲和度并归一化门控权重
    selected_affinities = torch.gather(affinities, dim=-1, index=expert_indices)
    gates = selected_affinities / (selected_affinities.sum(dim=-1, keepdim=True) + eps)

    # Step 3: 根据负载偏差更新不可微的专家偏置
    num_experts = router_weight.shape[0]
    expert_counts = torch.bincount(expert_indices.reshape(-1), minlength=num_experts).to(affinities.dtype)
    average_load = x.shape[0] * top_k / num_experts
    updated_bias = expert_bias - bias_lr * torch.sign(expert_counts - average_load)
    return expert_indices, gates, updated_bias''',
    'demo': """torch.manual_seed(42)
x = torch.randn(8, 16)
router_weight = torch.randn(4, 16)
indices, gates, updated_bias = aux_free_moe_route(x, router_weight, torch.zeros(4), 2)
print("Expert indices shape:", indices.shape)
print("Gate sums:", gates.sum(dim=-1))
print("Updated bias:", updated_bias)""",
    'explanation': r"""### 核心思路
路由器用加动态偏置的 sigmoid 亲和度选择 top-k 专家，但实际门控仍来自无偏亲和度；偏置按专家负载相对平均值的符号更新。

### 代码步骤拆解
- **Step 1（专家选择）**：`x @ router_weight.T` 得到 `(B, E)` 分数，加 `(E,)` 偏置后选出 `(B, K)` 索引。
- **Step 2（门控归一化）**：按索引收集无偏 `affinities`，沿专家维归一化为和接近 1 的 `gates`。
- **Step 3（负载反馈）**：通过 `bincount` 统计每个专家的 token 数，并降低过载专家偏置、提高欠载专家偏置。

### 重点与易错点分析
- **注意点 1（选择与门控分离）**：动态偏置只影响 top-k 排名，不能混入当前 token 的门控幅值。
- **注意点 2（归一化稳定性）**：分母加 `eps` 防止极端下溢；`keepdim=True` 保证与 `(B, K)` 正确广播。
- **注意点 3（负载更新）**：索引选择和符号偏置更新不可微，梯度只通过被选中的无偏亲和度流向 `x` 与 `router_weight`。

### 复杂度分析
- **时间复杂度**：$O(BED+BE\log K)$，线性路由投影是主要计算瓶颈。
- **空间复杂度**：$O(BE+BK)$，主要保存全专家亲和度与 top-k 结果。""",
}

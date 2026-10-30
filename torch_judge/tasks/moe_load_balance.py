"""MoE Load Balancing Loss task."""

TASK = {
    'title': 'MoE Load Balancing Loss',
    'title_zh': 'MoE 负载均衡损失',
    'difficulty': 'Medium',
    'category': '混合专家模型',
    'function_name': 'moe_load_balance_loss',
    'starter_code': """def moe_load_balance_loss(router_logits, num_experts):
    pass""",
    'hint': """1. `assignments = logits.argmax(dim=-1)` → `f[e] = (assignments == e).float().mean()`
2. `P = softmax(logits, dim=-1).mean(dim=0)`
3. return `num_experts * (f * P).sum()`""",
    'hint_zh': """1. `assignments = logits.argmax(dim=-1)` → `f[e] = (assignments == e).float().mean()`
2. `P = softmax(logits, dim=-1).mean(dim=0)`
3. 返回 `num_experts * (f * P).sum()`""",
    'description_en': """Implement the auxiliary load balancing loss from Switch Transformer / DeepSeek-style Mixture-of-Experts.

Without this loss, MoE models collapse: all tokens route to the same expert, wasting capacity. The loss penalizes uneven token distribution by combining a non-differentiable routing fraction with a differentiable probability term.

**Signature:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`

**Parameters:**
- `router_logits` — raw router scores before softmax (N_tokens, num_experts)
- `num_experts` — number of experts

**Returns:** scalar auxiliary loss

**Formula:**
- `f_i` = fraction of tokens assigned to expert i (via argmax — non-differentiable)
- `P_i` = mean router probability for expert i (via softmax — differentiable)
- `L_aux = num_experts * Σ(f_i * P_i)`

**Note:** Gradient flows only through `P_i` (the softmax term). `f_i` is a stop-gradient counting term.""",
    'description_zh': """实现 Switch Transformer / DeepSeek 风格混合专家（MoE）中的辅助负载均衡损失。

没有这个损失，MoE 模型会退化：所有 token 都路由到同一个专家，浪费容量。该损失通过将不可微的路由分数与可微的概率项结合，惩罚不均匀的 token 分布。

**签名:** `moe_load_balance_loss(router_logits, num_experts) -> Tensor`

**参数:**
- `router_logits` — softmax 前的原始路由分数 (N_tokens, num_experts)
- `num_experts` — 专家数量

**返回:** 标量辅助损失

**公式:**
- `f_i` = 路由到专家 i 的 token 比例（通过 argmax，不可微）
- `P_i` = 专家 i 的平均路由概率（通过 softmax，可微）
- `L_aux = num_experts * Σ(f_i * P_i)`

**注意:** 梯度只通过 `P_i`（softmax 项）传播，`f_i` 是停止梯度的计数项。""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
N_tokens, num_experts = 16, 4
logits = torch.randn(N_tokens, num_experts)
loss = {fn}(logits, num_experts)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'""",
        },
        {
            "name": 'Uniform routing gives loss = 1.0',
            "code": """import torch
# Explicitly construct uniform routing: each expert gets N/E tokens
num_experts = 4
N_tokens = 100
logits = torch.zeros(N_tokens, num_experts)
for e in range(num_experts):
    logits[e * 25:(e + 1) * 25, e] = 10.0
# f_i = 1/4 for all i, P_i ≈ softmax-dominated by the assigned expert
# With large logit gap, P_i ≈ 1/4 on average, so L_aux ≈ 4 * 4 * (1/4)^2 = 1.0
loss = {fn}(logits, num_experts)
assert abs(loss.item() - 1.0) < 0.05, f'Uniform routing should give loss≈1.0, got {loss.item():.4f}'""",
        },
        {
            "name": 'Collapsed routing gives loss > 1',
            "code": """import torch
num_experts = 4
N_tokens = 20
# All tokens routed to expert 0 via very large logit
logits = torch.zeros(N_tokens, num_experts)
logits[:, 0] = 100.0
loss = {fn}(logits, num_experts)
assert loss.item() > 1.0, f'Collapsed routing should give loss > 1.0, got {loss.item():.4f}'""",
        },
        {
            "name": 'Gradient flows through router_logits',
            "code": """import torch
torch.manual_seed(1)
N_tokens, num_experts = 12, 3
logits = torch.randn(N_tokens, num_experts, requires_grad=True)
loss = {fn}(logits, num_experts)
loss.backward()
assert logits.grad is not None, 'Missing gradient for router_logits'
assert logits.grad.shape == logits.shape, 'Gradient shape mismatch'""",
        },
    ],
    'solution': '''def moe_load_balance_loss(router_logits, num_experts):
    """MoE 负载均衡损失 (MoE Load Balancing Loss)。"""
    # Step 1: 用 argmax 统计每个专家实际接收的 token 比例 f_i
    N_tokens = router_logits.shape[0]
    assignments = router_logits.argmax(dim=-1)
    routing_fraction = torch.zeros(num_experts, device=router_logits.device)
    for expert_index in range(num_experts):
        routing_fraction[expert_index] = (assignments == expert_index).float().mean()
    # Step 2: 对路由概率按 token 求均值得到每个专家的软重要性 P_i
    probs = torch.softmax(router_logits, dim=-1)
    mean_probabilities = probs.mean(dim=0)
    # Step 3: 计算 E 倍的 f_i 与 P_i 内积，惩罚实际流量和概率质量同时集中
    return num_experts * (routing_fraction * mean_probabilities).sum()''',
    'demo': """torch.manual_seed(42)
num_experts = 4
N_tokens = 40
logits_uniform = torch.full((N_tokens, num_experts), -8.0)
logits_uniform[torch.arange(N_tokens), torch.arange(N_tokens) % num_experts] = 8.0
loss_uniform = moe_load_balance_loss(logits_uniform, num_experts)
logits_collapsed = torch.zeros(N_tokens, num_experts)
logits_collapsed[:, 0] = 100.0
loss_collapsed = moe_load_balance_loss(logits_collapsed, num_experts)
print('均匀分配损失:', f'{loss_uniform.item():.4f}')
print('坍缩到专家 0 的损失:', f'{loss_collapsed.item():.4f}')
print('坍缩惩罚倍率:', f'{(loss_collapsed / loss_uniform).item():.2f}x')""",
    'explanation': r"""### 核心思路
该辅助项同时观察硬路由流量 $f_i=\frac1N\sum_t\mathbf1[\arg\max_j z_{t,j}=i]$ 与软概率质量 $P_i=\frac1N\sum_t\operatorname{softmax}(z_t)_i$，定义 $L_{\mathrm{aux}}=E\sum_i f_iP_i$。均匀路由时 $f_i=P_i=1/E$，损失基线为 1；若流量和置信度都集中到一个专家，损失接近 $E$。

### 代码步骤拆解
- **Step 1（统计硬负载）**：`assignments` 取每行最大 logit 的专家编号；逐专家比较后求均值，形成长度为 `num_experts` 的 `routing_fraction`。
- **Step 2（统计软重要性）**：`probs` 的形状为 `(N_tokens,E)`，沿 token 维平均得到 `mean_probabilities=P`，它保留到所有 logits 的梯度路径。
- **Step 3（耦合两种分布）**：计算 `num_experts * dot(routing_fraction, mean_probabilities)`，使高硬负载专家继续获得高概率质量时受到更大惩罚。

### 重点与易错点分析
- **注意点 1（停止梯度的计数）**：`argmax` 与相等比较不可微，所以 $f_i$ 只作为本批次负载系数；梯度只通过 $P_i$ 回到路由 logits，这正是该损失的设计。
- **注意点 2（基线不是零）**：最优均衡状态的损失为 1 而非 0，训练时通常乘一个较小辅助系数；不能用“损失应趋近零”来判断实现。
- **注意点 3（批次统计噪声）**：token 数很少或按设备局部统计时，$f_i$ 方差较大；真实分布式 MoE 常跨序列或设备聚合统计，避免局部均衡掩盖全局坍缩。

### 复杂度分析
- **时间复杂度**：$O(NE)$，主要工作是对 E 维 logits 做 argmax、softmax 和逐专家计数。
- **空间复杂度**：$O(NE+E)$，softmax 概率占主导，硬分配索引另占 $O(N)$。""",
}

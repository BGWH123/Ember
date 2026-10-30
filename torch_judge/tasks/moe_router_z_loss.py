"""MoE Router Z-Loss task."""

TASK = {
    'title': 'MoE Router Z-Loss',
    'title_zh': 'MoE 路由器 Z-Loss',
    'difficulty': 'Easy',
    'category': '混合专家模型',
    'function_name': 'moe_router_z_loss',
    'starter_code': """def moe_router_z_loss(router_logits, coef=0.001):
    pass""",
    'hint': """Use `torch.logsumexp` over the expert dimension, square each log-partition value, average, and multiply by `coef`.""",
    'hint_zh': '沿专家维使用 `torch.logsumexp`，对每个对数配分函数值平方、取均值，再乘以 `coef`。',
    'description_en': 'Compute the router z-loss for logits whose final dimension indexes experts.',
    'description_zh': '计算路由器 Z-Loss，其中 logits 的最后一维表示专家。',
    'theory_en': """The squared log-partition penalty discourages router logits from growing without bound while stable log-sum-exp avoids overflow.""",
    'theory_zh': '对数配分函数的平方惩罚可抑制路由 logits 无界增大，稳定的 log-sum-exp 可避免溢出。',
    'diagram_en': """```text
router logits (..., N) -> logsumexp over N -> square -> mean -> coefficient
```""",
    'diagram_zh': """```text
路由 logits (..., N) -> 沿 N 做 logsumexp -> 平方 -> 均值 -> 系数
```""",
    "tests": [
        {
            "name": 'scalar shape',
            "code": """import torch
logits = torch.randn(2, 3, 5)
loss = {fn}(logits)
assert loss.shape == torch.Size([])
assert loss.dtype == logits.dtype""",
        },
        {
            "name": 'exact numerical reference',
            "code": """import torch
logits = torch.tensor([[1., -2., 0.5], [3., 1., -1.]])
loss = {fn}(logits, coef=0.2)
ref = 0.2 * torch.logsumexp(logits, dim=-1).square().mean()
assert torch.equal(loss, ref)""",
        },
        {
            "name": 'large logits remain finite',
            "code": """import torch
logits = torch.tensor([[10000., 9999.], [-10000., -10001.]])
loss = {fn}(logits, coef=0.0)
assert loss.ndim == 0
assert torch.isfinite(loss) and loss.item() == 0.0""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
logits = torch.randn(4, 7, requires_grad=True)
loss = {fn}(logits, coef=1e-3)
loss.backward()
assert logits.grad is not None
assert logits.grad.shape == logits.shape and torch.isfinite(logits.grad).all()""",
        },
    ],
    'solution': '''def moe_router_z_loss(router_logits, coef=0.001):
    """MoE 路由器 Z-Loss (MoE Router Z-Loss)。"""
    # Step 1: 沿专家维计算每个 token 的稳定对数配分函数 logsumexp
    lse = torch.logsumexp(router_logits, dim=-1)
    # Step 2: 平方对数配分值，使过大的正负归一化偏移都产生惩罚
    squared_log_partition = lse ** 2
    # Step 3: 对全部 token 求均值并乘 z-loss 系数
    return coef * squared_log_partition.mean()''',
    'demo': """torch.manual_seed(42)
router_logits = torch.randn(2, 128, 8, requires_grad=True)
loss = moe_router_z_loss(router_logits)
loss.backward()
print('z-loss:', f'{loss.item():.6f}')
print('路由梯度范数:', f'{router_logits.grad.norm().item():.6f}')""",
    'explanation': r"""### 核心思路
路由 softmax 对所有 logits 同加常数不敏感，因此主任务无法约束这一平移自由度，logits 可能整体漂移并造成低精度训练不稳定。Z-Loss 对每个 token 的对数配分函数 $z_t=\log\sum_{e=1}^{E}\exp(r_{t,e})$ 施加平方惩罚：$L_z=c\,\frac1N\sum_t z_t^2$，把 softmax 的归一化常数拉回零附近。

### 代码步骤拆解
- **Step 1（求对数配分）**：`torch.logsumexp(router_logits, dim=-1)` 将任意前导维保留，只消去最后的专家维 E。
- **Step 2（施加二次约束）**：`squared_log_partition` 对每个 token 的 $z_t$ 平方，既惩罚过大的正值，也惩罚过大的负值。
- **Step 3（归约与缩放）**：在所有 token/批次位置上求均值，再乘 `coef` 得到标量辅助损失。

### 重点与易错点分析
- **注意点 1（必须使用 logsumexp）**：直接写 `log(exp(logits).sum())` 会在大正 logits 上溢出；`torch.logsumexp` 内部先减最大值，测试中的 $10^4$ 仍可计算。
- **注意点 2（它不等于 L2 正则）**：该项只约束 $\log\sum_e e^{r_e}$，不会逐元素把所有专家 logits 拉向零；它针对的是 softmax 的共同平移方向。
- **注意点 3（梯度结构）**：$\partial L_z/\partial r_{t,e}=2c\,z_t\,\operatorname{softmax}(r_t)_e/N$，高概率专家承担更大的校正；`coef=0` 时仍返回同 dtype 的可微零标量。

### 复杂度分析
- **时间复杂度**：$O(NE)$，其中 N 是 token 位置总数、E 是专家数，瓶颈是最后一维的 log-sum-exp 归约。
- **空间复杂度**：$O(N)$，需要保存每个 token 的 `lse` 与平方值；反向传播还会保留归约所需状态。""",
}

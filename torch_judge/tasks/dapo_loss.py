"""DAPO (Clip-Higher & Token-Level Policy Optimization) Loss task."""

TASK = {
    'title': 'DAPO (Clip-Higher & Token-Level Policy Optimization) Loss',
    'title_zh': 'DAPO（非对称裁剪与 Token 级策略优化）损失',
    'difficulty': 'Hard',
    'category': '强化学习',
    'function_name': 'dapo_loss',
    'starter_code': """def dapo_loss(new_logps, old_logps, advantages, action_mask, eps_low=0.2, eps_high=0.28):
    pass""",
    'hint': """Detach old log-probabilities and advantages, expand per-sequence advantages with an extra last dimension, apply asymmetric ratio clipping, and normalize over all active tokens.""",
    'hint_zh': '分离旧对数概率和优势；为序列级优势增加末维进行广播；执行非对称比率裁剪；最后在全批次有效 token 上归一化。',
    'description_en': """Implement DAPO's asymmetric clipped surrogate objective with token-level normalization. Advantages may be sequence-level or token-level, while only action-mask positions contribute.""",
    'description_zh': '实现 DAPO 的非对称裁剪代理目标和 token 级归一化。优势可以是序列级或 token 级，且仅动作掩码位置参与计算。',
    'theory_en': """Clip-higher permits larger beneficial ratio increases than decreases. Global active-token normalization prevents response lengths from receiving equal sequence weight regardless of token count.""",
    'theory_zh': '上界更宽的裁剪允许更大的有利比率增长。按全局有效 token 归一化，避免不同长度回复被强制赋予相同序列权重。',
    'diagram_en': """```mermaid
flowchart LR
N[new logp] --> R[exp difference]
O[detached old logp] --> R
R --> C[asymmetric clamp]
A[detached advantage] --> S[min surrogates]
C --> S
M[action mask] --> G[global token mean]
S --> G
```""",
    'diagram_zh': """```mermaid
flowchart LR
N[新 logp] --> R[差值取指数]
O[分离的旧 logp] --> R
R --> C[非对称裁剪]
A[分离的优势] --> S[代理项取最小]
C --> S
M[动作掩码] --> G[全局 token 均值]
S --> G
```""",
    "tests": [
        {
            "name": 'scalar with broadcast advantages',
            "code": """import torch
n = torch.zeros(2, 3, requires_grad=True)
o = torch.zeros(2, 3)
a = torch.tensor([1.0, -1.0])
m = torch.ones(2, 3)
loss = {fn}(n, o, a, m)
assert loss.shape == torch.Size([]) and torch.equal(loss, torch.tensor(0.0))""",
        },
        {
            "name": 'exact asymmetric clipping',
            "code": """import torch
n = torch.log(torch.tensor([[1.5, 0.5], [1.1, 0.9]], dtype=torch.double))
o = torch.zeros_like(n)
a = torch.tensor([[2.0, 2.0], [-3.0, -3.0]], dtype=torch.double)
m = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.double)
got = {fn}(n, o, a, m, eps_low=.2, eps_high=.3)
r = torch.exp(n - o)
ref = -(torch.minimum(r * a, r.clamp(.8, 1.3) * a) * m).sum() / m.sum().clamp(min=1.0)
assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)""",
        },
        {
            "name": 'empty action mask',
            "code": """import torch
n = torch.randn(2, 2)
o = torch.randn(2, 2)
loss = {fn}(n, o, torch.randn(2), torch.zeros(2, 2))
assert loss.shape == torch.Size([])
assert torch.equal(loss, torch.tensor(0.0)) and torch.isfinite(loss)""",
        },
        {
            "name": 'detached inputs and policy gradient',
            "code": """import torch
n = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
o = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
a = torch.ones(2, dtype=torch.double, requires_grad=True)
{fn}(n, o, a, torch.ones_like(n)).backward()
assert n.grad is not None and n.grad.shape == n.shape
assert o.grad is None and a.grad is None and torch.isfinite(n.grad).all()""",
        },
    ],
    'solution': '''def dapo_loss(new_logps, old_logps, advantages, action_mask, eps_low=0.2, eps_high=0.28):
    """计算 DAPO 非对称裁剪策略损失 (DAPO Loss)。"""
    # Step 1: 停止固定信号梯度，并将序列级优势扩展到 token 维
    advantages_fixed = advantages.detach()
    if advantages_fixed.dim() == 1:
        advantages_fixed = advantages_fixed.unsqueeze(-1)

    # Step 2: 计算重要性比率与非对称裁剪代理目标
    ratio = torch.exp(new_logps - old_logps.detach())
    clipped_ratio = ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    surrogate = torch.minimum(ratio * advantages_fixed, clipped_ratio * advantages_fixed)

    # Step 3: 仅对有效动作 token 做全局归一化
    active_tokens = action_mask.sum().clamp(min=1.0)
    loss = -(surrogate * action_mask).sum() / active_tokens
    return loss''',
    'demo': """torch.manual_seed(42)
new_logps = torch.randn(2, 4) * 0.1
old_logps = torch.zeros_like(new_logps)
action_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
loss = dapo_loss(new_logps, old_logps, torch.tensor([1.0, -0.5]), action_mask)
print("Loss:", loss.item())
print("Active tokens:", int(action_mask.sum().item()))""",
    'explanation': r"""### 核心思路
DAPO 以 $r=\exp(\log\pi_{new}-\log\pi_{old})$ 构造 PPO 类代理目标，但使用非对称区间 $[1-\epsilon_{low},1+\epsilon_{high}]$，并按全批次有效 token 数归一化。

### 代码步骤拆解
- **Step 1（优势广播）**：将 `(B,)` 的序列优势扩展为 `(B, 1)`，使其能广播到 `(B, T)`。
- **Step 2（非对称裁剪）**：比较原始与裁剪后的重要性比率代理项，逐 token 取较小值。
- **Step 3（掩码归约）**：乘 `action_mask` 后求和，再除以全局 `active_tokens` 得到标量损失。

### 重点与易错点分析
- **注意点 1（梯度截断）**：`old_logps` 与 `advantages` 必须 `.detach()`，梯度只能流向新策略 `new_logps`。
- **注意点 2（空掩码）**：分母用 `clamp(min=1.0)`，保证没有有效 token 时返回有限的零损失。
- **注意点 3（归一化口径）**：应按所有有效 token 全局平均，不能先按序列平均后再平均，否则不同长度样本权重失真。

### 复杂度分析
- **时间复杂度**：$O(BT)$，主要开销是逐 token 的指数与裁剪运算。
- **空间复杂度**：$O(BT)$，需保存比率和代理目标中间张量。""",
}

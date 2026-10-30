"""Generalized Advantage Estimation (GAE) task."""

TASK = {
    'title': 'Generalized Advantage Estimation (GAE)',
    'title_zh': '广义优势估计（GAE）',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'compute_gae',
    'starter_code': """def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    pass""",
    'hint': """Allocate advantages like rewards and scan time backward. At each step mask both bootstrap and recursive carry at episode boundaries, then add values to get returns.""",
    'hint_zh': '创建与 rewards 同形的优势张量并反向遍历时间。每步在 episode 终止处屏蔽 bootstrap 和递归项，最后加上 values 得到回报。',
    'description_en': """Compute generalized advantage estimates and value-function returns for either one trajectory or a batch of trajectories, respecting terminal boundaries.""",
    'description_zh': '为单条或批量轨迹计算广义优势估计和值函数回报，并正确处理 episode 终止边界。',
    'theory_en': """GAE exponentially weights temporal-difference residuals with gamma*lambda. Done masks stop both value bootstrapping and residual propagation across episode boundaries.""",
    'theory_zh': 'GAE 使用 gamma*lambda 对时序差分残差进行指数加权。done 掩码阻止值 bootstrap 和残差跨 episode 边界传播。',
    'diagram_en': """```mermaid
flowchart RL
VT[next value] --> D[TD residual]
R[reward] --> D
V[current value] --> D
D --> G[delta + gamma lambda mask GAE]
G --> A[advantages]
A --> Ret[returns = A + V]
```""",
    'diagram_zh': """```mermaid
flowchart RL
VT[下一状态值] --> D[TD 残差]
R[奖励] --> D
V[当前状态值] --> D
D --> G[delta + gamma lambda mask GAE]
G --> A[优势]
A --> Ret[回报 = A + V]
```""",
    "tests": [
        {
            "name": 'batched output shapes',
            "code": """import torch
r = torch.randn(2, 4)
v = torch.randn(2, 4)
a, ret = {fn}(r, v, torch.zeros(2, 4), torch.randn(2))
assert a.shape == r.shape and ret.shape == r.shape
assert torch.allclose(ret, a + v)""",
        },
        {
            "name": 'exact reverse recursion',
            "code": """import torch
r = torch.tensor([[1., 2., 3.], [0., 1., 0.]], dtype=torch.double)
v = torch.tensor([[.2, .4, .6], [.1, .3, .2]], dtype=torch.double)
d = torch.tensor([[0., 0., 1.], [0., 0., 0.]])
lv = torch.tensor([9., .5], dtype=torch.double)
got_a, got_r = {fn}(r, v, d, lv, gamma=.9, lam=.8)
ref = torch.zeros_like(r); carry = torch.zeros(2, dtype=torch.double)
for i in range(2, -1, -1):
    nv = lv if i == 2 else v[:, i+1]; m = 1-d[:, i]
    delta = r[:, i] + .9*nv*m - v[:, i]
    carry = delta + .9*.8*m*carry; ref[:, i] = carry
assert torch.allclose(got_a, ref, atol=1e-12, rtol=1e-12) and torch.allclose(got_r, ref+v)""",
        },
        {
            "name": 'one-dimensional terminal edge',
            "code": """import torch
r = torch.tensor([1.0, 2.0])
v = torch.tensor([0.5, 0.7])
a, ret = {fn}(r, v, torch.tensor([0.0, 1.0]), torch.tensor(100.0), gamma=1.0, lam=1.0)
assert a.shape == (2,) and ret.shape == (2,)
assert torch.allclose(a, torch.tensor([2.5, 1.3])) and torch.allclose(ret, torch.tensor([3.0, 2.0]))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
r = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
v = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
lv = torch.randn(2, dtype=torch.double, requires_grad=True)
a, ret = {fn}(r, v, torch.zeros(2, 3), lv)
(a.sum() + ret.sum()).backward()
assert r.grad is not None and v.grad is not None and lv.grad is not None
assert r.grad.shape == r.shape and torch.isfinite(v.grad).all()""",
        },
    ],
    'solution': '''def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """计算广义优势估计 (Generalized Advantage Estimation)。"""
    # Step 1: 初始化优势张量和反向递推状态
    advantages = torch.zeros_like(rewards)
    gae_value = torch.zeros_like(last_value)

    # Step 2: 沿时间轴反向累积 TD 残差
    T = rewards.shape[-1]
    for t in range(T - 1, -1, -1):
        next_value = last_value if t == T - 1 else values[..., t + 1]
        mask = 1.0 - dones[..., t].float()
        delta = rewards[..., t] + gamma * next_value * mask - values[..., t]
        gae_value = delta + gamma * lam * mask * gae_value
        advantages[..., t] = gae_value

    # Step 3: 将优势加回基线得到回报
    returns = advantages + values
    return advantages, returns''',
    'demo': """torch.manual_seed(42)
rewards = torch.tensor([1.0, 0.5, 2.0])
values = torch.tensor([0.2, 0.3, 0.4])
dones = torch.tensor([0.0, 0.0, 1.0])
advantages, returns = compute_gae(rewards, values, dones, torch.tensor(0.0))
print("Output shape:", advantages.shape)
print("Advantages:", advantages)
print("Returns:", returns)""",
    'explanation': r"""### 核心思路
GAE 反向递推 $A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}$，其中 $\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t$，最后令回报等于优势加价值基线。

### 代码步骤拆解
- **Step 1（状态初始化）**：`advantages` 与奖励同形，`gae_value` 与 `last_value` 同形以兼容批量轨迹。
- **Step 2（反向递推）**：沿最后一维从 `T-1` 到 0 更新 `delta` 和 `gae_value`。
- **Step 3（构造回报）**：`advantages + values` 得到值函数训练目标 `returns`。

### 重点与易错点分析
- **注意点 1（终止掩码）**：`mask` 同时屏蔽 bootstrap 和未来优势，防止跨 episode 泄漏。
- **注意点 2（时间维约定）**：始终把最后一维当时间轴，因此一维和批量二维输入共享实现。
- **注意点 3（梯度语义）**：实现未主动 `.detach()`，满足测试对奖励、价值和末值的梯度要求。

### 复杂度分析
- **时间复杂度**：$O(BT)$，每个轨迹时间步访问一次。
- **空间复杂度**：$O(BT)$，主要保存优势和回报张量。""",
}

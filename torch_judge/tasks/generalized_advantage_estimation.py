"""Generalized Advantage Estimation task."""

TASK = {
    "title": "Generalized Advantage Estimation (GAE)",
    "title_zh": "广义优势估计（GAE）",
    "difficulty": "Medium",
    "category": "强化学习",
    "function_name": "compute_gae",
    "starter_code": '''import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    pass
''',
    "hint": "Allocate advantages like rewards and scan time backward. At each step mask both bootstrap and recursive carry at episode boundaries, then add values to get returns.",
    "hint_zh": "创建与 rewards 同形的优势张量并反向遍历时间。每步在 episode 终止处屏蔽 bootstrap 和递归项，最后加上 values 得到回报。",
    "description_en": "Compute generalized advantage estimates and value-function returns for either one trajectory or a batch of trajectories, respecting terminal boundaries.",
    "description_zh": "为单条或批量轨迹计算广义优势估计和值函数回报，并正确处理 episode 终止边界。",
    "theory_en": "GAE exponentially weights temporal-difference residuals with gamma*lambda. Done masks stop both value bootstrapping and residual propagation across episode boundaries.",
    "theory_zh": "GAE 使用 gamma*lambda 对时序差分残差进行指数加权。done 掩码阻止值 bootstrap 和残差跨 episode 边界传播。",
    "diagram_en": "```mermaid\nflowchart RL\nVT[next value] --> D[TD residual]\nR[reward] --> D\nV[current value] --> D\nD --> G[delta + gamma lambda mask GAE]\nG --> A[advantages]\nA --> Ret[returns = A + V]\n```",
    "diagram_zh": "```mermaid\nflowchart RL\nVT[下一状态值] --> D[TD 残差]\nR[奖励] --> D\nV[当前状态值] --> D\nD --> G[delta + gamma lambda mask GAE]\nG --> A[优势]\nA --> Ret[回报 = A + V]\n```",
    "tests": [
        {"name": "batched output shapes", "code": """import torch
r = torch.randn(2, 4)
v = torch.randn(2, 4)
a, ret = {fn}(r, v, torch.zeros(2, 4), torch.randn(2))
assert a.shape == r.shape and ret.shape == r.shape
assert torch.allclose(ret, a + v)
"""},
        {"name": "exact reverse recursion", "code": """import torch
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
assert torch.allclose(got_a, ref, atol=1e-12, rtol=1e-12) and torch.allclose(got_r, ref+v)
"""},
        {"name": "one-dimensional terminal edge", "code": """import torch
r = torch.tensor([1.0, 2.0])
v = torch.tensor([0.5, 0.7])
a, ret = {fn}(r, v, torch.tensor([0.0, 1.0]), torch.tensor(100.0), gamma=1.0, lam=1.0)
assert a.shape == (2,) and ret.shape == (2,)
assert torch.allclose(a, torch.tensor([2.5, 1.3])) and torch.allclose(ret, torch.tensor([3.0, 2.0]))
"""},
        {"name": "gradient flow", "code": """import torch
r = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
v = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
lv = torch.randn(2, dtype=torch.double, requires_grad=True)
a, ret = {fn}(r, v, torch.zeros(2, 3), lv)
(a.sum() + ret.sum()).backward()
assert r.grad is not None and v.grad is not None and lv.grad is not None
assert r.grad.shape == r.shape and torch.isfinite(v.grad).all()
"""},
    ],
    "solution": '''import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in range(rewards.shape[-1] - 1, -1, -1):
        next_v = last_value if t == rewards.shape[-1] - 1 else values[..., t + 1]
        mask = 1.0 - dones[..., t].float()
        delta = rewards[..., t] + gamma * next_v * mask - values[..., t]
        gae = delta + gamma * lam * mask * gae
        advantages[..., t] = gae
    returns = advantages + values
    return advantages, returns
''',
    "demo": '''rewards = torch.tensor([1.0, 0.5, 2.0])
values = torch.tensor([0.2, 0.3, 0.4])
print(compute_gae(rewards, values, torch.tensor([0., 0., 1.]), torch.tensor(0.0)))
''',
    "explanation": "Using the last axis as time supports both documented input ranks. A terminal mask removes the supplied bootstrap value and clears future GAE carry.",
}

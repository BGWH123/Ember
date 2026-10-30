"""DAPO token-level clipped policy loss task."""

TASK = {
    "title": "DAPO (Clip-Higher & Token-Level Policy Optimization) Loss",
    "title_zh": "DAPO（非对称裁剪与 Token 级策略优化）损失",
    "difficulty": "Hard",
    "category": "强化学习",
    "function_name": "dapo_loss",
    "starter_code": '''import torch

def dapo_loss(new_logps, old_logps, advantages, action_mask,
              eps_low=0.2, eps_high=0.28):
    pass
''',
    "hint": "Detach old log-probabilities and advantages, expand per-sequence advantages with an extra last dimension, apply asymmetric ratio clipping, and normalize over all active tokens.",
    "hint_zh": "分离旧对数概率和优势；为序列级优势增加末维进行广播；执行非对称比率裁剪；最后在全批次有效 token 上归一化。",
    "description_en": "Implement DAPO's asymmetric clipped surrogate objective with token-level normalization. Advantages may be sequence-level or token-level, while only action-mask positions contribute.",
    "description_zh": "实现 DAPO 的非对称裁剪代理目标和 token 级归一化。优势可以是序列级或 token 级，且仅动作掩码位置参与计算。",
    "theory_en": "Clip-higher permits larger beneficial ratio increases than decreases. Global active-token normalization prevents response lengths from receiving equal sequence weight regardless of token count.",
    "theory_zh": "上界更宽的裁剪允许更大的有利比率增长。按全局有效 token 归一化，避免不同长度回复被强制赋予相同序列权重。",
    "diagram_en": "```mermaid\nflowchart LR\nN[new logp] --> R[exp difference]\nO[detached old logp] --> R\nR --> C[asymmetric clamp]\nA[detached advantage] --> S[min surrogates]\nC --> S\nM[action mask] --> G[global token mean]\nS --> G\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nN[新 logp] --> R[差值取指数]\nO[分离的旧 logp] --> R\nR --> C[非对称裁剪]\nA[分离的优势] --> S[代理项取最小]\nC --> S\nM[动作掩码] --> G[全局 token 均值]\nS --> G\n```",
    "tests": [
        {"name": "scalar with broadcast advantages", "code": """import torch
n = torch.zeros(2, 3, requires_grad=True)
o = torch.zeros(2, 3)
a = torch.tensor([1.0, -1.0])
m = torch.ones(2, 3)
loss = {fn}(n, o, a, m)
assert loss.shape == torch.Size([]) and torch.equal(loss, torch.tensor(0.0))
"""},
        {"name": "exact asymmetric clipping", "code": """import torch
n = torch.log(torch.tensor([[1.5, 0.5], [1.1, 0.9]], dtype=torch.double))
o = torch.zeros_like(n)
a = torch.tensor([[2.0, 2.0], [-3.0, -3.0]], dtype=torch.double)
m = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.double)
got = {fn}(n, o, a, m, eps_low=.2, eps_high=.3)
r = torch.exp(n - o)
ref = -(torch.minimum(r * a, r.clamp(.8, 1.3) * a) * m).sum() / m.sum().clamp(min=1.0)
assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)
"""},
        {"name": "empty action mask", "code": """import torch
n = torch.randn(2, 2)
o = torch.randn(2, 2)
loss = {fn}(n, o, torch.randn(2), torch.zeros(2, 2))
assert loss.shape == torch.Size([])
assert torch.equal(loss, torch.tensor(0.0)) and torch.isfinite(loss)
"""},
        {"name": "detached inputs and policy gradient", "code": """import torch
n = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
o = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
a = torch.ones(2, dtype=torch.double, requires_grad=True)
{fn}(n, o, a, torch.ones_like(n)).backward()
assert n.grad is not None and n.grad.shape == n.shape
assert o.grad is None and a.grad is None and torch.isfinite(n.grad).all()
"""},
    ],
    "solution": '''import torch

def dapo_loss(new_logps, old_logps, advantages, action_mask,
              eps_low=0.2, eps_high=0.28):
    adv = advantages.detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    ratio = torch.exp(new_logps - old_logps.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    surr = torch.minimum(ratio * adv, clipped_ratio * adv)
    return -(surr * action_mask).sum() / action_mask.sum().clamp(min=1.0)
''',
    "demo": '''new = torch.tensor([[0.1, -0.2]])
old = torch.zeros_like(new)
print(dapo_loss(new, old, torch.tensor([1.0]), torch.ones_like(new)))
''',
    "explanation": "Unsqueezing a one-dimensional advantage broadcasts it over sequence length. Detaching both fixed inputs confines gradients to the new policy.",
}

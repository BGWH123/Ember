"""Schulman KL estimators for RLHF."""

TASK = {
    "title": "Schulman KL Estimators (k1, k2, k3) for RLHF",
    "title_zh": "RLHF 中的 Schulman KL 估计器（k1, k2, k3）",
    "difficulty": "Medium",
    "category": "强化学习",
    "function_name": "schulman_kl_estimators",
    "starter_code": '''import torch
from torch import Tensor

def schulman_kl_estimators(logp_policy, logp_ref, mask=None):
    pass
''',
    "hint": "Form the policy/reference log-ratio, compute all three pointwise estimators, then apply either a regular mean or one shared masked-mean rule.",
    "hint_zh": "先计算策略与参考模型的对数概率比，再逐元素计算三个估计器，最后使用普通均值或统一的掩码均值。",
    "description_en": "Given tokens sampled from the policy, implement Schulman's k1, k2, and nonnegative k3 KL estimators. Return their scalar means, optionally over only positions selected by a mask.",
    "description_zh": "给定从策略模型采样的 token，实现 Schulman 的 k1、k2 和非负 k3 KL 估计器。返回三个标量均值，并可通过掩码仅统计有效位置。",
    "theory_en": "For l=log pi-log q, k1=l is unbiased but noisy, k2=l^2/2 is a local quadratic approximation, and k3=exp(-l)+l-1 is a low-variance nonnegative estimator.",
    "theory_zh": "令 l=log pi-log q。k1=l 无偏但方差较大；k2=l²/2 是局部二次近似；k3=exp(-l)+l-1 是低方差且非负的估计器。",
    "diagram_en": "```mermaid\nflowchart LR\nP[policy logp] --> L[log ratio]\nR[reference logp] --> L\nL --> K1[k1]\nL --> K2[k2]\nL --> K3[k3]\nK1 --> M[mean / masked mean]\nK2 --> M\nK3 --> M\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nP[策略 logp] --> L[对数比]\nR[参考 logp] --> L\nL --> K1[k1]\nL --> K2[k2]\nL --> K3[k3]\nK1 --> M[均值或掩码均值]\nK2 --> M\nK3 --> M\n```",
    "tests": [
        {"name": "shapes without mask", "code": """import torch
x = torch.tensor([[0.2, -0.1], [0.4, 0.0]], requires_grad=True)
out = {fn}(x, torch.zeros_like(x))
assert isinstance(out, tuple) and len(out) == 3
assert all(v.shape == torch.Size([]) for v in out)
assert torch.allclose(out[0], x.mean())
"""},
        {"name": "exact estimator reference", "code": """import torch
p = torch.tensor([0.0, -0.5, -1.0], dtype=torch.double)
r = torch.tensor([-0.2, -0.1, -1.4], dtype=torch.double)
l = p - r
got = {fn}(p, r)
ref = (l.mean(), (0.5 * l.square()).mean(), (torch.exp(-l) + l - 1.0).mean())
assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-12) for a, b in zip(got, ref))
"""},
        {"name": "masked mean and empty mask", "code": """import torch
p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
r = torch.zeros_like(p)
m = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
got = {fn}(p, r, m)
assert torch.equal(got[0], torch.tensor(2.5))
empty = {fn}(p, r, torch.zeros_like(p))
assert all(torch.equal(v, torch.tensor(0.0)) for v in empty)
"""},
        {"name": "gradient flow", "code": """import torch
p = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
r = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
vals = {fn}(p, r, torch.ones_like(p))
sum(vals).backward()
assert p.grad is not None and r.grad is not None
assert p.grad.shape == p.shape and torch.isfinite(p.grad).all()
"""},
    ],
    "solution": '''import torch

def schulman_kl_estimators(logp_policy, logp_ref, mask=None):
    log_ratio = logp_policy - logp_ref
    values = (
        log_ratio,
        0.5 * log_ratio.square(),
        torch.exp(-log_ratio) + log_ratio - 1.0,
    )
    if mask is None:
        return tuple(x.mean() for x in values)
    denominator = mask.sum().clamp(min=1.0)
    return tuple((x * mask).sum() / denominator for x in values)
''',
    "demo": '''p = torch.tensor([-0.2, -0.7])
r = torch.tensor([-0.3, -0.5])
print(schulman_kl_estimators(p, r))
''',
    "explanation": "The same mask denominator is used for every estimator. Clamping it to one makes an all-zero mask produce finite zero outputs.",
}

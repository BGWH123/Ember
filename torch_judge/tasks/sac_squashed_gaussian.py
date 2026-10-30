"""SAC squashed Gaussian policy task."""

TASK = {
    "title": "SAC Squashed Gaussian Policy & Tanh Jacobian Log-Prob",
    "title_zh": "SAC 压缩高斯策略与雅可比修正对数概率",
    "difficulty": "Medium",
    "category": "强化学习",
    "function_name": "sac_squashed_gaussian",
    "starter_code": '''import torch
import math

def sac_squashed_gaussian(mu, log_std, noise, eps=1e-6):
    pass
''',
    "hint": "Reparameterize u=mu+exp(log_std)*noise, apply tanh, compute the diagonal Gaussian log-density, subtract log(1-action^2+eps), then sum over action dimensions.",
    "hint_zh": "先重参数化 u=mu+exp(log_std)*noise，再应用 tanh；计算对角高斯对数密度，减去 log(1-action²+eps)，最后沿动作维求和。",
    "description_en": "Implement a reparameterized Gaussian SAC action sample and its log-probability after tanh squashing, including the change-of-variables Jacobian.",
    "description_zh": "实现 SAC 中重参数化高斯动作采样及 tanh 压缩后的对数概率，并包含变量变换的雅可比修正。",
    "theory_en": "Tanh maps unconstrained Gaussian samples into (-1,1). Density under this transform must subtract log|1-tanh(u)^2| for every action dimension.",
    "theory_zh": "tanh 将无约束高斯样本映射到 (-1,1)。变换后的密度必须在每个动作维减去 log|1-tanh(u)²|。",
    "diagram_en": "```mermaid\nflowchart LR\nM[mu] --> U[u = mu + std noise]\nS[log std] --> U\nN[noise] --> U\nU --> A[tanh action]\nU --> G[Gaussian log p]\nA --> J[Jacobian correction]\nG --> L[corrected log prob]\nJ --> L\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nM[均值] --> U[u = 均值 + 标准差*噪声]\nS[log 标准差] --> U\nN[噪声] --> U\nU --> A[tanh 动作]\nU --> G[高斯 log p]\nA --> J[雅可比修正]\nG --> L[修正后 log 概率]\nJ --> L\n```",
    "tests": [
        {"name": "documented output shapes", "code": """import torch
mu = torch.randn(3, 4)
action, logp = {fn}(mu, torch.zeros_like(mu), torch.randn_like(mu))
assert action.shape == (3, 4) and logp.shape == (3,)
assert (action.abs() < 1).all() and torch.isfinite(logp).all()
"""},
        {"name": "exact density reference", "code": """import torch
import math
mu = torch.tensor([[.2, -.3], [.5, .1]], dtype=torch.double)
ls = torch.tensor([[-.4, .2], [.0, -.7]], dtype=torch.double)
n = torch.tensor([[.1, -1.], [.7, .3]], dtype=torch.double)
a, lp = {fn}(mu, ls, n, eps=1e-9)
std = torch.exp(ls); u = mu + std*n; ref_a = torch.tanh(u)
base = -.5*(((u-mu)/(std+1e-9))**2 + 2*ls + math.log(2*math.pi))
ref_lp = (base-torch.log(1-ref_a.square()+1e-9)).sum(-1)
assert torch.allclose(a, ref_a, atol=1e-12, rtol=1e-12) and torch.allclose(lp, ref_lp, atol=1e-12, rtol=1e-12)
"""},
        {"name": "zero noise edge", "code": """import torch
mu = torch.tensor([[0.0, 0.5]])
ls = torch.zeros_like(mu)
a, lp = {fn}(mu, ls, torch.zeros_like(mu))
assert torch.allclose(a, torch.tanh(mu))
assert lp.shape == (1,) and torch.isfinite(lp).all()
"""},
        {"name": "reparameterization gradients", "code": """import torch
mu = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
ls = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
noise = torch.randn(2, 3, dtype=torch.double)
a, lp = {fn}(mu, ls, noise)
(a.sum() + lp.sum()).backward()
assert mu.grad is not None and ls.grad is not None
assert mu.grad.shape == mu.shape and torch.isfinite(mu.grad).all() and torch.isfinite(ls.grad).all()
"""},
    ],
    "solution": '''import torch
import math

def sac_squashed_gaussian(mu, log_std, noise, eps=1e-6):
    std = torch.exp(log_std)
    u = mu + std * noise
    action = torch.tanh(u)
    log_p_u = -0.5 * (
        ((u - mu) / (std + eps)) ** 2
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    log_prob = (
        log_p_u - torch.log(1.0 - action.pow(2) + eps)
    ).sum(dim=-1)
    return action, log_prob
''',
    "demo": '''mu = torch.zeros(2, 3)
log_std = torch.full_like(mu, -0.5)
print(sac_squashed_gaussian(mu, log_std, torch.randn_like(mu)))
''',
    "explanation": "The sample path is differentiable in both policy parameters. Summing corrected per-dimension densities gives one log-probability per batch item.",
}

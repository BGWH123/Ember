"""MoE router z-loss task."""

TASK = {
    "title": "MoE Router Z-Loss",
    "title_zh": "MoE 路由器 Z-Loss",
    "difficulty": "Easy",
    "category": "混合专家模型",
    "function_name": "moe_router_z_loss",
    "starter_code": '''def moe_router_z_loss(router_logits, coef=1e-3):
    pass
''',
    "hint": "Use `torch.logsumexp` over the expert dimension, square each log-partition value, average, and multiply by `coef`.",
    "hint_zh": "沿专家维使用 `torch.logsumexp`，对每个对数配分函数值平方、取均值，再乘以 `coef`。",
    "description_en": "Compute the router z-loss for logits whose final dimension indexes experts.",
    "description_zh": "计算路由器 Z-Loss，其中 logits 的最后一维表示专家。",
    "theory_en": "The squared log-partition penalty discourages router logits from growing without bound while stable log-sum-exp avoids overflow.",
    "theory_zh": "对数配分函数的平方惩罚可抑制路由 logits 无界增大，稳定的 log-sum-exp 可避免溢出。",
    "diagram_en": "```text\nrouter logits (..., N) -> logsumexp over N -> square -> mean -> coefficient\n```",
    "diagram_zh": "```text\n路由 logits (..., N) -> 沿 N 做 logsumexp -> 平方 -> 均值 -> 系数\n```",
    "tests": [
        {
            "name": "scalar shape",
            "code": '''import torch
logits = torch.randn(2, 3, 5)
loss = {fn}(logits)
assert loss.shape == torch.Size([])
assert loss.dtype == logits.dtype
''',
        },
        {
            "name": "exact numerical reference",
            "code": '''import torch
logits = torch.tensor([[1., -2., 0.5], [3., 1., -1.]])
loss = {fn}(logits, coef=0.2)
ref = 0.2 * torch.logsumexp(logits, dim=-1).square().mean()
assert torch.equal(loss, ref)
''',
        },
        {
            "name": "large logits remain finite",
            "code": '''import torch
logits = torch.tensor([[10000., 9999.], [-10000., -10001.]])
loss = {fn}(logits, coef=0.0)
assert loss.ndim == 0
assert torch.isfinite(loss) and loss.item() == 0.0
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
logits = torch.randn(4, 7, requires_grad=True)
loss = {fn}(logits, coef=1e-3)
loss.backward()
assert logits.grad is not None
assert logits.grad.shape == logits.shape and torch.isfinite(logits.grad).all()
''',
        },
    ],
    "solution": '''import torch

def moe_router_z_loss(router_logits, coef=1e-3):
    lse = torch.logsumexp(router_logits, dim=-1)
    return coef * (lse ** 2).mean()
''',
    "demo": '''router_logits = torch.randn(2, 128, 8, requires_grad=True)
loss = moe_router_z_loss(router_logits)
loss.backward()
print(loss.item(), router_logits.grad.norm().item())
''',
    "explanation": "The implementation reduces only the final expert axis before averaging across all remaining token and batch dimensions.",
}

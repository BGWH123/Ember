"""Stable LogSumExp and Log-Softmax task."""

TASK = {
    "title": "Numerically Stable LogSumExp & Log-Softmax",
    "title_zh": "数值稳定 LogSumExp 与 Log-Softmax",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "function_name": "logsumexp_logsoftmax",
    "starter_code": '''def logsumexp_logsoftmax(x, dim=-1):
    pass
''',
    "hint": "Subtract the maximum with `keepdim=True` before exponentiation; retain that dimension for log-probabilities and squeeze it only for LogSumExp.",
    "hint_zh": "指数运算前减去 `keepdim=True` 的最大值；计算对数概率时保留该维，仅对 LogSumExp 结果压缩该维。",
    "description_en": "Implement numerically stable LogSumExp and log-softmax together using the max-subtraction trick.",
    "description_zh": "使用减最大值技巧，同时实现数值稳定的 LogSumExp 和 log-softmax。",
    "theory_en": "Because `log(sum(exp(x)))` is shift-equivariant, subtracting the largest entry prevents overflow. Adding the maximum back recovers the exact result, while `x-lse` yields log-softmax.",
    "theory_zh": "由于 `log(sum(exp(x)))` 对平移具有等变性，先减最大值可避免溢出；随后加回最大值得到原结果，而 `x-lse` 即 log-softmax。",
    "diagram_en": "```text\nx -> max m -> exp(x-m) -> sum -> log -> +m -> lse\nx --------------------------------------------- lse -> x-lse -> log_probs\n```",
    "diagram_zh": "```text\nx -> 最大值 m -> exp(x-m) -> 求和 -> log -> +m -> lse\nx ---------------------------------------------- lse -> x-lse -> log_probs\n```",
    "tests": [
        {
            "name": "output shapes",
            "code": '''import torch
x = torch.randn(2, 3, 4)
lse, logp = {fn}(x, dim=1)
assert lse.shape == (2, 4) and logp.shape == x.shape
''',
        },
        {
            "name": "matches reference",
            "code": '''import torch
torch.manual_seed(3)
x = torch.randn(2, 5, 7, dtype=torch.float64)
lse, logp = {fn}(x, -1)
assert torch.allclose(lse, torch.logsumexp(x, -1), atol=1e-12)
assert torch.allclose(logp, torch.log_softmax(x, -1), atol=1e-12)
''',
        },
        {
            "name": "extreme shift stability",
            "code": '''import torch
x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
lse, logp = {fn}(x)
assert torch.isfinite(lse).all() and torch.isfinite(logp).all()
assert torch.allclose(logp.exp().sum(-1), torch.ones(2), atol=2e-5)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(3, 6, requires_grad=True)
lse, logp = {fn}(x, 1)
(lse.sum() + logp.square().mean()).backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
''',
        },
    ],
    "solution": '''import torch

def logsumexp_logsoftmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    lse_keep = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
    log_probs = x - lse_keep
    lse = lse_keep.squeeze(dim)
    return lse, log_probs
''',
    "demo": '''x = torch.randn(2, 5) + 1000.0
lse, log_probs = logsumexp_logsoftmax(x)
print(lse)
print(log_probs.exp().sum(dim=-1))
''',
    "explanation": "After subtracting the row maximum, every exponent is at most one, preventing positive overflow. Keeping the reduced dimension allows direct broadcasting when computing log-probabilities.",
}

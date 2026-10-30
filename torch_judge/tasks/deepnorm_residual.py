"""DeepNorm residual scaling task."""

TASK = {
    "title": "DeepNorm Residual Scaling & Init Gain",
    "title_zh": "DeepNorm 残差缩放与初始化增益",
    "difficulty": "Medium",
    "category": "归一化",
    "function_name": "deepnorm_step",
    "starter_code": '''def deepnorm_step(x, sublayer_out, gamma, beta, num_layers, eps=1e-5):
    pass
''',
    "hint": "Compute the two fourth-root constants, form `alpha*x+sublayer_out`, and manually LayerNorm the last dimension.",
    "hint_zh": "计算两个四次方根常数，构造 `alpha*x+sublayer_out`，再沿最后一维手动执行 LayerNorm。",
    "description_en": "Implement one DeepNorm residual-and-normalization step and return its recommended parameter initialization gain.",
    "description_zh": "实现一次 DeepNorm 残差与归一化步骤，并返回推荐的参数初始化增益。",
    "theory_en": "DeepNorm scales the residual stream by `alpha=(2N)^(1/4)` and uses initialization gain `(8N)^(-1/4)` to stabilize very deep Transformer stacks.",
    "theory_zh": "DeepNorm 用 `alpha=(2N)^(1/4)` 缩放残差流，并使用初始化增益 `(8N)^(-1/4)` 稳定超深 Transformer。",
    "diagram_en": "```text\nx --*alpha--\\\n              + -> LayerNorm(gamma,beta) -> output\nsublayer -----/                    beta_gain returned\n```",
    "diagram_zh": "```text\nx --*alpha--\\\n              + -> LayerNorm(gamma,beta) -> 输出\n子层输出 -----/                    同时返回 beta_gain\n```",
    "tests": [
        {
            "name": "shape and gain",
            "code": '''import torch
x = torch.randn(2, 3, 8)
out, gain = {fn}(x, torch.randn_like(x), torch.ones(8), torch.zeros(8), 12)
assert out.shape == x.shape and isinstance(gain, float)
assert abs(gain - float((8.0 * 12) ** -0.25)) < 1e-12
''',
        },
        {
            "name": "numerical reference",
            "code": '''import torch
torch.manual_seed(2)
x, sub = torch.randn(3, 6), torch.randn(3, 6)
gamma, beta = torch.randn(6), torch.randn(6)
out, gain = {fn}(x, sub, gamma, beta, 24)
u = x * ((2.0 * 24) ** 0.25) + sub
mean, var = u.mean(-1, keepdim=True), u.var(-1, keepdim=True, unbiased=False)
ref = gamma * (u - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-6) and gain == float((8.0 * 24) ** -0.25)
''',
        },
        {
            "name": "constant input edge",
            "code": '''import torch
x = torch.ones(2, 4)
sub = torch.ones(2, 4)
beta = torch.arange(4, dtype=x.dtype)
out, gain = {fn}(x, sub, torch.randn(4), beta, 1)
assert torch.allclose(out, beta.expand_as(out), atol=1e-6) and torch.isfinite(torch.tensor(gain))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2, 5, requires_grad=True)
sub = torch.randn(2, 5, requires_grad=True)
gamma = torch.ones(5, requires_grad=True)
beta = torch.zeros(5, requires_grad=True)
out, gain = {fn}(x, sub, gamma, beta, 8)
out.square().sum().backward()
assert all(t.grad is not None for t in (x, sub, gamma, beta)) and gain > 0
''',
        },
    ],
    "solution": '''import torch

def deepnorm_step(x, sublayer_out, gamma, beta, num_layers, eps=1e-5):
    alpha = (2.0 * num_layers) ** 0.25
    beta_gain = (8.0 * num_layers) ** (-0.25)
    u = x * alpha + sublayer_out
    mean = u.mean(dim=-1, keepdim=True)
    var = u.var(dim=-1, keepdim=True, unbiased=False)
    normalized = gamma * (u - mean) / torch.sqrt(var + eps) + beta
    return normalized, float(beta_gain)
''',
    "demo": '''x = torch.randn(2, 16, 64)
sub = torch.randn_like(x)
out, init_gain = deepnorm_step(x, sub, torch.ones(64), torch.zeros(64), 48)
print(out.shape, init_gain)
''',
    "explanation": "Residual scaling is applied before population-variance LayerNorm. The initialization gain is a Python float because it configures weight initialization rather than participating in this tensor computation.",
}

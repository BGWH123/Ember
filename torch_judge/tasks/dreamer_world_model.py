"""Dreamer lambda-return and symmetric logarithm task."""

TASK = {
    "title": "World Models: Symlog Transform & Dreamer Lambda-Return",
    "title_zh": "世界模型：Symlog 变换与 Dreamer Lambda 回报",
    "difficulty": "Hard",
    "category": "世界模型",
    "function_name": "dreamer_lambda_return",
    "starter_code": '''def dreamer_lambda_return(rewards, values, continues, gamma=0.99, lam=0.95):
    pass
''',
    "hint": "Initialize the final return from the final value, scan backward through the horizon, then apply the sign-preserving log and exponential elementwise.",
    "hint_zh": "用最后一个价值初始化最终回报，沿时间范围反向递推，再逐元素应用保号对数和指数变换。",
    "description_en": "Compute finite-horizon Dreamer lambda-returns together with their symlog encoding and exact symexp reconstruction.",
    "description_zh": "计算有限时域 Dreamer lambda 回报，并返回其 symlog 编码和精确 symexp 重建。",
    "theory_en": "Lambda-returns interpolate between one-step value targets and long rollouts. Symlog compresses large signed magnitudes smoothly, while symexp is its inverse.",
    "theory_zh": "Lambda 回报在一步价值目标与长时域展开之间插值。Symlog 平滑压缩大幅有符号数值，symexp 则是其逆变换。",
    "diagram_en": "```text\nlast value -> backward lambda recursion -> G -> symlog(G) -> symexp -> G\n```",
    "diagram_zh": "```text\n末端价值 -> 反向 lambda 递推 -> G -> symlog(G) -> symexp -> G\n```",
    "tests": [
        {"name": "tuple shapes", "code": '''import torch
r = torch.randn(2, 5)
v = torch.randn(2, 5)
c = torch.ones(2, 5)
G, y, x = {fn}(r, v, c)
assert G.shape == y.shape == x.shape == (2, 5)
'''},
        {"name": "exact recursion", "code": '''import torch
r = torch.tensor([[1., 2., 3.]])
v = torch.tensor([[.5, .6, .7]])
c = torch.tensor([[1., .5, 1.]])
G, y, x = {fn}(r, v, c, gamma=.9, lam=.8)
ref = torch.empty_like(r)
ref[:, 2] = v[:, 2]
ref[:, 1] = r[:, 1] + .9 * c[:, 1] * (.2 * v[:, 2] + .8 * ref[:, 2])
ref[:, 0] = r[:, 0] + .9 * c[:, 0] * (.2 * v[:, 1] + .8 * ref[:, 1])
assert torch.allclose(G, ref) and torch.allclose(y, torch.sign(ref) * torch.log1p(ref.abs())) and torch.allclose(x, ref)
'''},
        {"name": "horizon one edge case", "code": '''import torch
r = torch.tensor([[99.], [-2.]])
v = torch.tensor([[3.], [-4.]])
G, y, x = {fn}(r, v, torch.ones_like(r))
assert torch.equal(G, v) and torch.allclose(x, G, atol=1e-6)
'''},
        {"name": "gradient flow", "code": '''import torch
r = torch.randn(2, 4, requires_grad=True)
v = torch.randn(2, 4, requires_grad=True)
c = torch.sigmoid(torch.randn(2, 4)).requires_grad_()
G, y, x = {fn}(r, v, c)
(G.sum() + y.sum() + x.sum()).backward()
assert r.grad is not None and v.grad is not None and c.grad is not None
'''},
    ],
    "solution": '''import torch

def dreamer_lambda_return(rewards, values, continues, gamma=0.99, lam=0.95):
    H = rewards.shape[1]
    returns = [values[:, H - 1]]
    for t in range(H - 2, -1, -1):
        next_return = rewards[:, t] + gamma * continues[:, t] * (
            (1.0 - lam) * values[:, t + 1] + lam * returns[-1]
        )
        returns.append(next_return)
    G = torch.stack(returns[::-1], dim=1)
    symlog_G = torch.sign(G) * torch.log1p(torch.abs(G))
    reconstructed = torch.sign(symlog_G) * torch.expm1(torch.abs(symlog_G))
    return G, symlog_G, reconstructed
''',
    "demo": '''rewards = torch.randn(3, 15)
values = torch.randn(3, 15)
continues = torch.ones(3, 15)
G, encoded, decoded = dreamer_lambda_return(rewards, values, continues)
print(G.shape, (G - decoded).abs().max())
''',
    "explanation": "The final horizon position is a bootstrap value exactly as specified; rewards affect positions zero through H-2. A list-and-stack recursion preserves autograd without in-place writes.",
}

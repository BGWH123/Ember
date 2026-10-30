"""DDPM posterior and v-prediction conversion task."""

TASK = {
    "title": "DDPM Posterior Distribution & v-Prediction Conversion",
    "title_zh": "DDPM 后验分布与 v-Prediction 转换",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "ddpm_posterior_and_vpred",
    "starter_code": '''def ddpm_posterior_and_vpred(x_t, v_pred, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t):
    pass
''',
    "hint": "First recover x_0 and epsilon from v-prediction, then substitute x_0 into the closed-form DDPM posterior mean and variance.",
    "hint_zh": "先从 v-prediction 恢复 x_0 和 epsilon，再将 x_0 代入 DDPM 后验均值与方差的闭式公式。",
    "description_en": "Convert a v-prediction into clean-sample and noise predictions, then compute the DDPM posterior q(x_{t-1}|x_t,x_0). Inputs x_t and v_pred have shape (B,D); schedule tensors are scalar or (B,1). Return (x0_pred, eps_pred, posterior_mean, posterior_var).",
    "description_zh": "将 v-prediction 转换为干净样本与噪声预测，并计算 DDPM 后验 q(x_{t-1}|x_t,x_0)。x_t 和 v_pred 形状为 (B,D)，调度张量为标量或 (B,1)。返回 (x0_pred, eps_pred, posterior_mean, posterior_var)。",
    "theory_en": "The v parameterization is an orthogonal rotation of (x_0, epsilon). Its inverse gives x_0=sqrt(alpha_bar_t)x_t-sqrt(1-alpha_bar_t)v and epsilon=sqrt(1-alpha_bar_t)x_t+sqrt(alpha_bar_t)v. The Gaussian forward process then gives an analytic posterior.",
    "theory_zh": "v 参数化是 (x_0, epsilon) 的正交旋转。逆变换为 x_0=sqrt(alpha_bar_t)x_t-sqrt(1-alpha_bar_t)v、epsilon=sqrt(1-alpha_bar_t)x_t+sqrt(alpha_bar_t)v，随后高斯前向过程给出解析后验。",
    "diagram_en": "```text\n(x_t, v) -> inverse rotation -> (x0_pred, eps_pred) -> DDPM posterior -> (mean, variance)\n```",
    "diagram_zh": "```text\n(x_t, v) -> 逆旋转 -> (x0_pred, eps_pred) -> DDPM 后验 -> (均值, 方差)\n```",
    "tests": [
        {
            "name": "shapes and scalar schedules",
            "code": '''import torch
x = torch.randn(3, 5)
v = torch.randn(3, 5)
out = {fn}(x, v, torch.tensor(.8), torch.tensor(.6), torch.tensor(.75), torch.tensor(.2))
assert [t.shape for t in out] == [(3, 5), (3, 5), (3, 5), torch.Size([])]
''',
        },
        {
            "name": "exact batched reference",
            "code": '''import torch
torch.manual_seed(1)
x, v = torch.randn(2, 4), torch.randn(2, 4)
a = torch.tensor([[.8], [.7]]); ab = torch.tensor([[.6], [.5]])
abp = torch.tensor([[.75], [.65]]); b = torch.tensor([[.2], [.3]])
got = {fn}(x, v, a, ab, abp, b)
x0 = torch.sqrt(ab)*x-torch.sqrt(1-ab)*v
ep = torch.sqrt(1-ab)*x+torch.sqrt(ab)*v
mean = torch.sqrt(abp)*b/(1-ab)*x0+torch.sqrt(a)*(1-abp)/(1-ab)*x
var = (1-abp)/(1-ab)*b
assert all(torch.equal(g, r) for g, r in zip(got, (x0, ep, mean, var)))
''',
        },
        {
            "name": "zero v edge case",
            "code": '''import torch
x = torch.tensor([[2., -1.]])
z = torch.zeros_like(x)
ab = torch.tensor([[.25]])
x0, ep, mean, var = {fn}(x, z, torch.tensor([[.5]]), ab, torch.tensor([[.5]]), torch.tensor([[.5]]))
assert torch.allclose(x0, .5*x) and torch.allclose(ep, torch.sqrt(torch.tensor(.75))*x)
assert mean.shape == x.shape and var.shape == (1, 1) and torch.isfinite(mean).all()
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2, 3, requires_grad=True)
v = torch.randn(2, 3, requires_grad=True)
ab = torch.full((2, 1), .6, requires_grad=True)
out = {fn}(x, v, torch.full((2,1),.8), ab, torch.full((2,1),.75), torch.full((2,1),.2))
sum(t.sum() for t in out).backward()
assert x.grad is not None and v.grad is not None and ab.grad is not None
''',
        },
    ],
    "solution": '''import torch

def ddpm_posterior_and_vpred(x_t, v_pred, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t):
    x0_pred = torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1.0 - alpha_bar_t) * v_pred
    eps_pred = torch.sqrt(1.0 - alpha_bar_t) * x_t + torch.sqrt(alpha_bar_t) * v_pred
    posterior_mean = (
        torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * x0_pred
        + torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * x_t
    )
    posterior_var = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
    return x0_pred, eps_pred, posterior_mean, posterior_var
''',
    "demo": '''x_t = torch.randn(2, 4)
v = torch.randn_like(x_t)
schedule = [torch.tensor(.8), torch.tensor(.6), torch.tensor(.75), torch.tensor(.2)]
print([x.shape for x in ddpm_posterior_and_vpred(x_t, v, *schedule)])
''',
    "explanation": "All schedule values broadcast over (B,D). The variance remains scalar or (B,1), while the other three outputs have the state shape.",
}

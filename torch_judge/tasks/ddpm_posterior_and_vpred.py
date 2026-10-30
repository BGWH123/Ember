"""DDPM Posterior Distribution & v-Prediction Conversion task."""

TASK = {
    'title': 'DDPM Posterior Distribution & v-Prediction Conversion',
    'title_zh': 'DDPM 后验分布与 v-Prediction 转换',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'ddpm_posterior_and_vpred',
    'starter_code': """def ddpm_posterior_and_vpred(x_t, v_pred, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t):
    pass""",
    'hint': """First recover x_0 and epsilon from v-prediction, then substitute x_0 into the closed-form DDPM posterior mean and variance.""",
    'hint_zh': '先从 v-prediction 恢复 x_0 和 epsilon，再将 x_0 代入 DDPM 后验均值与方差的闭式公式。',
    'description_en': """Convert a v-prediction into clean-sample and noise predictions, then compute the DDPM posterior q(x_{t-1}|x_t,x_0). Inputs x_t and v_pred have shape (B,D); schedule tensors are scalar or (B,1). Return (x0_pred, eps_pred, posterior_mean, posterior_var).""",
    'description_zh': """将 v-prediction 转换为干净样本与噪声预测，并计算 DDPM 后验 q(x_{t-1}|x_t,x_0)。x_t 和 v_pred 形状为 (B,D)，调度张量为标量或 (B,1)。返回 (x0_pred, eps_pred, posterior_mean, posterior_var)。""",
    'theory_en': """The v parameterization is an orthogonal rotation of (x_0, epsilon). Its inverse gives x_0=sqrt(alpha_bar_t)x_t-sqrt(1-alpha_bar_t)v and epsilon=sqrt(1-alpha_bar_t)x_t+sqrt(alpha_bar_t)v. The Gaussian forward process then gives an analytic posterior.""",
    'theory_zh': """v 参数化是 (x_0, epsilon) 的正交旋转。逆变换为 x_0=sqrt(alpha_bar_t)x_t-sqrt(1-alpha_bar_t)v、epsilon=sqrt(1-alpha_bar_t)x_t+sqrt(alpha_bar_t)v，随后高斯前向过程给出解析后验。""",
    'diagram_en': """```text
(x_t, v) -> inverse rotation -> (x0_pred, eps_pred) -> DDPM posterior -> (mean, variance)
```""",
    'diagram_zh': """```text
(x_t, v) -> 逆旋转 -> (x0_pred, eps_pred) -> DDPM 后验 -> (均值, 方差)
```""",
    "tests": [
        {
            "name": 'shapes and scalar schedules',
            "code": """import torch
x = torch.randn(3, 5)
v = torch.randn(3, 5)
out = {fn}(x, v, torch.tensor(.8), torch.tensor(.6), torch.tensor(.75), torch.tensor(.2))
assert [t.shape for t in out] == [(3, 5), (3, 5), (3, 5), torch.Size([])]""",
        },
        {
            "name": 'exact batched reference',
            "code": """import torch
torch.manual_seed(1)
x, v = torch.randn(2, 4), torch.randn(2, 4)
a = torch.tensor([[.8], [.7]]); ab = torch.tensor([[.6], [.5]])
abp = torch.tensor([[.75], [.65]]); b = torch.tensor([[.2], [.3]])
got = {fn}(x, v, a, ab, abp, b)
x0 = torch.sqrt(ab)*x-torch.sqrt(1-ab)*v
ep = torch.sqrt(1-ab)*x+torch.sqrt(ab)*v
mean = torch.sqrt(abp)*b/(1-ab)*x0+torch.sqrt(a)*(1-abp)/(1-ab)*x
var = (1-abp)/(1-ab)*b
assert all(torch.equal(g, r) for g, r in zip(got, (x0, ep, mean, var)))""",
        },
        {
            "name": 'zero v edge case',
            "code": """import torch
x = torch.tensor([[2., -1.]])
z = torch.zeros_like(x)
ab = torch.tensor([[.25]])
x0, ep, mean, var = {fn}(x, z, torch.tensor([[.5]]), ab, torch.tensor([[.5]]), torch.tensor([[.5]]))
assert torch.allclose(x0, .5*x) and torch.allclose(ep, torch.sqrt(torch.tensor(.75))*x)
assert mean.shape == x.shape and var.shape == (1, 1) and torch.isfinite(mean).all()""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2, 3, requires_grad=True)
v = torch.randn(2, 3, requires_grad=True)
ab = torch.full((2, 1), .6, requires_grad=True)
out = {fn}(x, v, torch.full((2,1),.8), ab, torch.full((2,1),.75), torch.full((2,1),.2))
sum(t.sum() for t in out).backward()
assert x.grad is not None and v.grad is not None and ab.grad is not None""",
        },
    ],
    'solution': '''def ddpm_posterior_and_vpred(x_t, v_pred, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t):
    """转换 v-prediction 并计算 DDPM 后验 (DDPM Posterior and v-Prediction)。"""
    # Step 1: 对 v 参数化执行逆旋转，恢复干净样本与噪声
    x0_pred = torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1.0 - alpha_bar_t) * v_pred
    eps_pred = torch.sqrt(1.0 - alpha_bar_t) * x_t + torch.sqrt(alpha_bar_t) * v_pred

    # Step 2: 计算 DDPM 后验均值的两个线性系数
    denominator = 1.0 - alpha_bar_t
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / denominator
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / denominator
    posterior_mean = coef_x0 * x0_pred + coef_xt * x_t

    # Step 3: 计算解析后验方差
    posterior_var = (1.0 - alpha_bar_prev) / denominator * beta_t
    return x0_pred, eps_pred, posterior_mean, posterior_var''',
    'demo': """torch.manual_seed(42)
x_t = torch.randn(2, 4)
v_pred = torch.randn_like(x_t)
outputs = ddpm_posterior_and_vpred(x_t, v_pred, torch.tensor(0.8), torch.tensor(0.6), torch.tensor(0.75), torch.tensor(0.2))
print("State output shapes:", [value.shape for value in outputs[:3]])
print("Posterior variance:", outputs[3])""",
    'explanation': r"""### 核心思路
`v_pred` 是 $(x_0,\epsilon)$ 的正交参数化；先逆变换得到 `x0_pred` 与 `eps_pred`，再代入高斯前向过程的解析后验 $q(x_{t-1}\mid x_t,x_0)$。

### 代码步骤拆解
- **Step 1（逆参数化）**：调度张量广播到 `(B, D)`，生成与 `x_t` 同形状的两个预测。
- **Step 2（后验均值）**：计算 `coef_x0` 与 `coef_xt`，线性组合 `x0_pred` 和 `x_t`。
- **Step 3（后验方差）**：由 `beta_t` 与累计调度比值得到标量或 `(B, 1)` 的 `posterior_var`。

### 重点与易错点分析
- **注意点 1（广播维度）**：调度值应为标量或 `(B, 1)`，这样才能沿特征维正确广播。
- **注意点 2（分母稳定性）**：`alpha_bar_t` 接近 1 时分母很小；实际调度需避免在该公式中使用退化端点。
- **注意点 3（参数化符号）**：`x0_pred` 的 v 项为负，`eps_pred` 的 v 项为正，符号颠倒会破坏正交逆变换。

### 复杂度分析
- **时间复杂度**：$O(BD)$，所有输出均由逐元素运算生成。
- **空间复杂度**：$O(BD)$，主要保存三个状态形状的中间与输出张量。""",
}

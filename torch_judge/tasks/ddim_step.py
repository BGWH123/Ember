"""DDIM Sampling Step task."""

TASK = {
    'title': 'DDIM Sampling Step',
    'title_zh': 'DDIM 采样步骤',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'ddim_step',
    'starter_code': """def ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev):
    pass""",
    'hint': """1. x0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
2. noise_dir = √(1-ᾱ_{t-1})·ε
3. x_{t-1} = √ᾱ_{t-1}·x0_pred + noise_dir
   Use ** 0.5 for square roots.""",
    'hint_zh': """1. x0_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
2. noise_dir = √(1-ᾱ_{t-1})·ε
3. x_{t-1} = √ᾱ_{t-1}·x0_pred + noise_dir
   用 ** 0.5 计算平方根。""",
    'description_en': """Implement one step of DDIM (Denoising Diffusion Implicit Models) deterministic sampling.

Given the current noisy sample and the model's noise prediction, compute the previous (less noisy) sample without any stochastic noise injection.

**Signature:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`

**Parameters:**
- `x_t` — current noisy sample at timestep t, shape (B, ...)
- `noise_pred` — model's predicted noise ε_θ(x_t, t), same shape as x_t
- `alpha_bar_t` — scalar, cumulative noise schedule ᾱ_t at current step
- `alpha_bar_prev` — scalar, cumulative noise schedule ᾱ_{t-1} at previous step

**Returns:** x_{t-1} with same shape as x_t

**Formula:**
1. Predict x0: `x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`
2. Direction toward x_t: `noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`
3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`""",
    'description_zh': """实现 DDIM（去噪扩散隐式模型）的单步确定性采样。

给定当前含噪样本和模型的噪声预测，在不注入随机噪声的情况下计算上一步（噪声更少的）样本。

**签名:** `ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev) -> Tensor`

**参数:**
- `x_t` — 时间步 t 的含噪样本，形状 (B, ...)
- `noise_pred` — 模型预测的噪声 ε_θ(x_t, t)，形状与 x_t 相同
- `alpha_bar_t` — 标量，当前步的累积噪声调度 ᾱ_t
- `alpha_bar_prev` — 标量，上一步的累积噪声调度 ᾱ_{t-1}

**返回:** x_{t-1}，形状与 x_t 相同

**公式:**
1. 预测 x0：`x0_pred = (x_t - sqrt(1 - ᾱ_t) * noise_pred) / sqrt(ᾱ_t)`
2. 指向 x_t 的方向：`noise_direction = sqrt(1 - ᾱ_{t-1}) * noise_pred`
3. `x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + noise_direction`""",
    "tests": [
        {
            "name": 'Output shape matches x_t',
            "code": """import torch
torch.manual_seed(0)
x_t = torch.randn(4, 3, 8, 8)
noise_pred = torch.randn(4, 3, 8, 8)
out = {fn}(x_t, noise_pred, alpha_bar_t=0.5, alpha_bar_prev=0.7)
assert out.shape == x_t.shape, f'Expected {x_t.shape}, got {out.shape}'""",
        },
        {
            "name": 'alpha_bar_prev=1 recovers x0_pred',
            "code": """import torch
torch.manual_seed(1)
x_t = torch.randn(2, 16)
noise_pred = torch.randn(2, 16)
alpha_bar_t = 0.4
# When alpha_bar_prev=1: noise_direction=0, x_prev = 1*x0_pred = x0_pred
x_prev = {fn}(x_t, noise_pred, alpha_bar_t=alpha_bar_t, alpha_bar_prev=1.0)
x0_pred = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
assert torch.allclose(x_prev, x0_pred, atol=1e-5), 'With alpha_bar_prev=1, output should equal x0_pred'""",
        },
        {
            "name": 'alpha_bar_t == alpha_bar_prev gives x_t back',
            "code": """import torch
torch.manual_seed(2)
x_t = torch.randn(3, 32)
noise_pred = torch.randn(3, 32)
alpha_bar = 0.6
x_prev = {fn}(x_t, noise_pred, alpha_bar_t=alpha_bar, alpha_bar_prev=alpha_bar)
assert torch.allclose(x_prev, x_t, atol=1e-5), 'When alpha_bar_t == alpha_bar_prev, output should equal x_t'""",
        },
        {
            "name": 'Gradient flow through x_t and noise_pred',
            "code": """import torch
x_t = torch.randn(2, 8, requires_grad=True)
noise_pred = torch.randn(2, 8, requires_grad=True)
out = {fn}(x_t, noise_pred, alpha_bar_t=0.5, alpha_bar_prev=0.3)
out.sum().backward()
assert x_t.grad is not None, 'No gradient for x_t'
assert noise_pred.grad is not None, 'No gradient for noise_pred'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch
torch.manual_seed(3)
x_t = torch.randn(1, 4)
noise_pred = torch.randn(1, 4)
abt, abp = 0.64, 0.81
x0_pred = (x_t - (1 - abt) ** 0.5 * noise_pred) / (abt ** 0.5)
noise_dir = (1 - abp) ** 0.5 * noise_pred
expected = abp ** 0.5 * x0_pred + noise_dir
result = {fn}(x_t, noise_pred, alpha_bar_t=abt, alpha_bar_prev=abp)
assert torch.allclose(result, expected, atol=1e-5), f'Numerical mismatch: {result} vs {expected}'""",
        },
    ],
    'solution': '''def ddim_step(x_t, noise_pred, alpha_bar_t, alpha_bar_prev):
    """执行确定性 DDIM 单步采样 (DDIM Sampling Step)。"""
    # Step 1: 从当前样本和噪声预测恢复干净样本
    x0_pred = (x_t - (1.0 - alpha_bar_t) ** 0.5 * noise_pred) / alpha_bar_t ** 0.5

    # Step 2: 计算前一时刻保留的噪声方向
    noise_direction = (1.0 - alpha_bar_prev) ** 0.5 * noise_pred

    # Step 3: 按前一时刻信噪比例重组样本
    x_prev = alpha_bar_prev ** 0.5 * x0_pred + noise_direction
    return x_prev''',
    'demo': """torch.manual_seed(42)
x_t = torch.randn(2, 4)
noise_pred = torch.randn_like(x_t)
x_prev = ddim_step(x_t, noise_pred, 0.5, 0.7)
print("Output shape:", x_prev.shape)
print("Sample values:", x_prev[0])""",
    'explanation': r"""### 核心思路
DDIM 先由 $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$ 恢复干净样本，再按上一时刻的系数确定性重组 $x_{t-1}$。

### 代码步骤拆解
- **Step 1（预测干净样本）**：`x_t` 与 `noise_pred` 同形状，通过当前累计噪声系数得到 `x0_pred`。
- **Step 2（噪声方向）**：使用 `alpha_bar_prev` 缩放同一个噪声预测，得到 `noise_direction`。
- **Step 3（确定性重组）**：将干净分量与噪声方向相加，输出形状不变的 `x_prev`。

### 重点与易错点分析
- **注意点 1（边界条件）**：`alpha_bar_t` 必须大于零，否则恢复 `x0_pred` 时会除零。
- **注意点 2（确定性）**：本题对应 $\eta=0$，不能额外采样随机噪声；给定输入后输出唯一。
- **注意点 3（系数对应）**：恢复干净样本用当前 `alpha_bar_t`，重组必须用 `alpha_bar_prev`，两者不可混用。

### 复杂度分析
- **时间复杂度**：$O(N)$，其中 $N$ 为样本张量元素数，全部操作均为逐元素计算。
- **空间复杂度**：$O(N)$，主要保存 `x0_pred` 与 `noise_direction`。""",
}

"""SAC Squashed Gaussian Policy & Tanh Jacobian Log-Prob task."""

TASK = {
    'title': 'SAC Squashed Gaussian Policy & Tanh Jacobian Log-Prob',
    'title_zh': 'SAC 压缩高斯策略与雅可比修正对数概率',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'sac_squashed_gaussian',
    'starter_code': """def sac_squashed_gaussian(mu, log_std, noise, eps=1e-06):
    pass""",
    'hint': """Reparameterize u=mu+exp(log_std)*noise, apply tanh, compute the diagonal Gaussian log-density, subtract log(1-action^2+eps), then sum over action dimensions.""",
    'hint_zh': '先重参数化 u=mu+exp(log_std)*noise，再应用 tanh；计算对角高斯对数密度，减去 log(1-action²+eps)，最后沿动作维求和。',
    'description_en': """Implement a reparameterized Gaussian SAC action sample and its log-probability after tanh squashing, including the change-of-variables Jacobian.""",
    'description_zh': '实现 SAC 中重参数化高斯动作采样及 tanh 压缩后的对数概率，并包含变量变换的雅可比修正。',
    'theory_en': """Tanh maps unconstrained Gaussian samples into (-1,1). Density under this transform must subtract log|1-tanh(u)^2| for every action dimension.""",
    'theory_zh': 'tanh 将无约束高斯样本映射到 (-1,1)。变换后的密度必须在每个动作维减去 log|1-tanh(u)²|。',
    'diagram_en': """```mermaid
flowchart LR
M[mu] --> U[u = mu + std noise]
S[log std] --> U
N[noise] --> U
U --> A[tanh action]
U --> G[Gaussian log p]
A --> J[Jacobian correction]
G --> L[corrected log prob]
J --> L
```""",
    'diagram_zh': """```mermaid
flowchart LR
M[均值] --> U[u = 均值 + 标准差*噪声]
S[log 标准差] --> U
N[噪声] --> U
U --> A[tanh 动作]
U --> G[高斯 log p]
A --> J[雅可比修正]
G --> L[修正后 log 概率]
J --> L
```""",
    "tests": [
        {
            "name": 'documented output shapes',
            "code": """import torch
mu = torch.randn(3, 4)
action, logp = {fn}(mu, torch.zeros_like(mu), torch.randn_like(mu))
assert action.shape == (3, 4) and logp.shape == (3,)
assert (action.abs() < 1).all() and torch.isfinite(logp).all()""",
        },
        {
            "name": 'exact density reference',
            "code": """import torch
import math
mu = torch.tensor([[.2, -.3], [.5, .1]], dtype=torch.double)
ls = torch.tensor([[-.4, .2], [.0, -.7]], dtype=torch.double)
n = torch.tensor([[.1, -1.], [.7, .3]], dtype=torch.double)
a, lp = {fn}(mu, ls, n, eps=1e-9)
std = torch.exp(ls); u = mu + std*n; ref_a = torch.tanh(u)
base = -.5*(((u-mu)/(std+1e-9))**2 + 2*ls + math.log(2*math.pi))
ref_lp = (base-torch.log(1-ref_a.square()+1e-9)).sum(-1)
assert torch.allclose(a, ref_a, atol=1e-12, rtol=1e-12) and torch.allclose(lp, ref_lp, atol=1e-12, rtol=1e-12)""",
        },
        {
            "name": 'zero noise edge',
            "code": """import torch
mu = torch.tensor([[0.0, 0.5]])
ls = torch.zeros_like(mu)
a, lp = {fn}(mu, ls, torch.zeros_like(mu))
assert torch.allclose(a, torch.tanh(mu))
assert lp.shape == (1,) and torch.isfinite(lp).all()""",
        },
        {
            "name": 'reparameterization gradients',
            "code": """import torch
mu = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
ls = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
noise = torch.randn(2, 3, dtype=torch.double)
a, lp = {fn}(mu, ls, noise)
(a.sum() + lp.sum()).backward()
assert mu.grad is not None and ls.grad is not None
assert mu.grad.shape == mu.shape and torch.isfinite(mu.grad).all() and torch.isfinite(ls.grad).all()""",
        },
    ],
    'solution': '''def sac_squashed_gaussian(mu, log_std, noise, eps=1e-06):
    """实现SAC 压缩高斯策略与雅可比修正对数概率（SAC Squashed Gaussian Policy & Tanh Jacobian Log-Prob）。"""
    # Step 1: 用可微重参数化从对角高斯得到压缩前动作
    std = torch.exp(log_std)
    u = mu + std * noise
    # Step 2: 通过 tanh 将无界样本压缩到合法动作区间
    action = torch.tanh(u)
    # Step 3: 从高斯密度减去 tanh 雅可比并汇总动作维
    log_p_u = -0.5 * (((u - mu) / (std + eps)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    log_prob = (log_p_u - torch.log(1.0 - action.pow(2) + eps)).sum(dim=-1)
    return (action, log_prob)''',
    'demo': """torch.manual_seed(42)
mu = torch.zeros(2, 3)
log_std = torch.full_like(mu, -0.5)
noise = torch.randn_like(mu)
action, log_prob = sac_squashed_gaussian(mu, log_std, noise)
print("Action range:", action.min().item(), action.max().item())
print("Per-sample corrected log-probability:", log_prob)
print("Output shapes:", action.shape, log_prob.shape)""",
    'explanation': r"""### 核心思路
SAC 先以重参数化 $u=\mu+\exp(\log\sigma)\odot\epsilon$ 生成无界高斯样本，再令 $a=\tanh(u)\in(-1,1)$。变量替换要求动作密度满足 $\log\pi(a)=\log\mathcal N(u;\mu,\sigma^2)-\sum_j\log(1-a_j^2)$；若漏掉第二项，靠近边界的动作概率会被系统性算错。

### 代码步骤拆解
- **Step 1（重参数化高斯样本）**：`std = exp(log_std)` 与 `mu`、`noise` 均为 $(B,A)$，`u = mu + std * noise` 让动作样本对策略参数保持可微。
- **Step 2（压缩动作范围）**：`action = tanh(u)` 逐维把 $u$ 映射到环境常用的 $(-1,1)$ 动作区间。
- **Step 3（修正变换后密度）**：先求每个动作维的 `log_p_u`，再减去 `log(1 - action.pow(2) + eps)`，最后沿 $A$ 维求和得到 $(B,)$ 的 `log_prob`。

### 重点与易错点分析
- **注意点 1（雅可比方向）**：因为 $da/du=1-\tanh^2(u)$，从 $u$ 的密度变到 $a$ 的密度要减去对数行列式，而不是加上；各动作维的项必须全部累加。
- **注意点 2（饱和区数值）**：当 $|u|$ 很大时 `action` 接近 $\pm1$，`eps` 防止 $\log(0)$；本实现还按题目参考在高斯标准化分母 `std + eps` 中加保护。
- **注意点 3（与直接采样对比）**：使用外部 `noise` 而非从不可微采样结果回传，使 $\nabla_\mu u=1$、$\nabla_{\log\sigma}u=\sigma\epsilon$；这正是 SAC 策略梯度可穿过采样步骤的关键。

### 复杂度分析
- **时间复杂度**：$O(BA)$，指数、双曲正切、高斯密度和雅可比修正均逐动作元素计算。
- **空间复杂度**：$O(BA)$，主要临时张量是 `std`、`u`、`action` 与逐维 `log_p_u`。""",
}

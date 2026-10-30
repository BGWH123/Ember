"""Schulman KL Estimators (k1, k2, k3) for RLHF task."""

TASK = {
    'title': 'Schulman KL Estimators (k1, k2, k3) for RLHF',
    'title_zh': 'RLHF 中的 Schulman KL 估计器（k1, k2, k3）',
    'difficulty': 'Medium',
    'category': '强化学习',
    'function_name': 'schulman_kl_estimators',
    'starter_code': """def schulman_kl_estimators(logp_policy, logp_ref, mask=None):
    pass""",
    'hint': """Form the policy/reference log-ratio, compute all three pointwise estimators, then apply either a regular mean or one shared masked-mean rule.""",
    'hint_zh': '先计算策略与参考模型的对数概率比，再逐元素计算三个估计器，最后使用普通均值或统一的掩码均值。',
    'description_en': """Given tokens sampled from the policy, implement Schulman's k1, k2, and nonnegative k3 KL estimators. Return their scalar means, optionally over only positions selected by a mask.""",
    'description_zh': '给定从策略模型采样的 token，实现 Schulman 的 k1、k2 和非负 k3 KL 估计器。返回三个标量均值，并可通过掩码仅统计有效位置。',
    'theory_en': """For l=log pi-log q, k1=l is unbiased but noisy, k2=l^2/2 is a local quadratic approximation, and k3=exp(-l)+l-1 is a low-variance nonnegative estimator.""",
    'theory_zh': '令 l=log pi-log q。k1=l 无偏但方差较大；k2=l²/2 是局部二次近似；k3=exp(-l)+l-1 是低方差且非负的估计器。',
    'diagram_en': """```mermaid
flowchart LR
P[policy logp] --> L[log ratio]
R[reference logp] --> L
L --> K1[k1]
L --> K2[k2]
L --> K3[k3]
K1 --> M[mean / masked mean]
K2 --> M
K3 --> M
```""",
    'diagram_zh': """```mermaid
flowchart LR
P[策略 logp] --> L[对数比]
R[参考 logp] --> L
L --> K1[k1]
L --> K2[k2]
L --> K3[k3]
K1 --> M[均值或掩码均值]
K2 --> M
K3 --> M
```""",
    "tests": [
        {
            "name": 'shapes without mask',
            "code": """import torch
x = torch.tensor([[0.2, -0.1], [0.4, 0.0]], requires_grad=True)
out = {fn}(x, torch.zeros_like(x))
assert isinstance(out, tuple) and len(out) == 3
assert all(v.shape == torch.Size([]) for v in out)
assert torch.allclose(out[0], x.mean())""",
        },
        {
            "name": 'exact estimator reference',
            "code": """import torch
p = torch.tensor([0.0, -0.5, -1.0], dtype=torch.double)
r = torch.tensor([-0.2, -0.1, -1.4], dtype=torch.double)
l = p - r
got = {fn}(p, r)
ref = (l.mean(), (0.5 * l.square()).mean(), (torch.exp(-l) + l - 1.0).mean())
assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-12) for a, b in zip(got, ref))""",
        },
        {
            "name": 'masked mean and empty mask',
            "code": """import torch
p = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
r = torch.zeros_like(p)
m = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
got = {fn}(p, r, m)
assert torch.equal(got[0], torch.tensor(2.5))
empty = {fn}(p, r, torch.zeros_like(p))
assert all(torch.equal(v, torch.tensor(0.0)) for v in empty)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
p = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
r = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
vals = {fn}(p, r, torch.ones_like(p))
sum(vals).backward()
assert p.grad is not None and r.grad is not None
assert p.grad.shape == p.shape and torch.isfinite(p.grad).all()""",
        },
    ],
    'solution': '''def schulman_kl_estimators(logp_policy, logp_ref, mask=None):
    """计算 Schulman 的三种 KL 估计器 (KL Estimators)。"""
    # Step 1: 构造策略相对参考模型的逐 token 对数比
    log_ratio = logp_policy - logp_ref

    # Step 2: 计算 k1、k2 与非负 k3
    estimators = (
        log_ratio,
        0.5 * log_ratio.square(),
        torch.exp(-log_ratio) + log_ratio - 1.0,
    )

    # Step 3: 使用普通均值或统一掩码均值归约
    if mask is None:
        return tuple(value.mean() for value in estimators)
    denominator = mask.sum().clamp(min=1.0)
    return tuple((value * mask).sum() / denominator for value in estimators)''',
    'demo': """torch.manual_seed(42)
logp_policy = torch.randn(2, 4)
logp_ref = torch.randn(2, 4)
mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
k1, k2, k3 = schulman_kl_estimators(logp_policy, logp_ref, mask)
print("Output shapes:", k1.shape, k2.shape, k3.shape)
print("Estimator values:", k1.item(), k2.item(), k3.item())
print("k3 nonnegative:", (k3 >= 0).item())""",
    'explanation': r"""### 核心思路
令 $l=\log\pi-\log q$，分别计算 $k_1=l$、$k_2=l^2/2$ 和 $k_3=e^{-l}+l-1$，再对有效 token 求均值。

### 代码步骤拆解
- **Step 1（对数比）**：`log_ratio` 与 token log-prob 张量同形。
- **Step 2（逐点估计）**：三个 `estimators` 保持相同形状，便于共享归约逻辑。
- **Step 3（掩码均值）**：有掩码时统一用 `mask.sum()` 作为分母。

### 重点与易错点分析
- **注意点 1（符号约定）**：k3 中指数项是 `exp(-log_ratio)`，符号错误会破坏非负性。
- **注意点 2（空掩码）**：分母下界设为 1，使全零掩码返回有限的零。
- **注意点 3（梯度流）**：策略与参考 log-prob 均未分离，测试要求两者都能接收梯度。

### 复杂度分析
- **时间复杂度**：$O(M)$，每个 token 计算常数次逐元素运算。
- **空间复杂度**：$O(M)$，保存对数比与三个估计张量。""",
}

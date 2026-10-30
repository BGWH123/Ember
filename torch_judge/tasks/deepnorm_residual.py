"""DeepNorm Residual Scaling & Init Gain task."""

TASK = {
    'title': 'DeepNorm Residual Scaling & Init Gain',
    'title_zh': 'DeepNorm 残差缩放与初始化增益',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'deepnorm_step',
    'starter_code': """def deepnorm_step(x, sublayer_out, gamma, beta, num_layers, eps=1e-5):
    pass""",
    'hint': """Compute the two fourth-root constants, form `alpha*x+sublayer_out`, and manually LayerNorm the last dimension.""",
    'hint_zh': '计算两个四次方根常数，构造 `alpha*x+sublayer_out`，再沿最后一维手动执行 LayerNorm。',
    'description_en': """Implement one DeepNorm residual-and-normalization step and return its recommended parameter initialization gain.""",
    'description_zh': '实现一次 DeepNorm 残差与归一化步骤，并返回推荐的参数初始化增益。',
    'theory_en': """DeepNorm scales the residual stream by `alpha=(2N)^(1/4)` and uses initialization gain `(8N)^(-1/4)` to stabilize very deep Transformer stacks.""",
    'theory_zh': 'DeepNorm 用 `alpha=(2N)^(1/4)` 缩放残差流，并使用初始化增益 `(8N)^(-1/4)` 稳定超深 Transformer。',
    'diagram_en': r"""```text
x --*alpha--\
              + -> LayerNorm(gamma,beta) -> output
sublayer -----/                    beta_gain returned
```""",
    'diagram_zh': r"""```text
x --*alpha--\
              + -> LayerNorm(gamma,beta) -> 输出
子层输出 -----/                    同时返回 beta_gain
```""",
    "tests": [
        {
            "name": 'shape and gain',
            "code": """import torch
x = torch.randn(2, 3, 8)
out, gain = {fn}(x, torch.randn_like(x), torch.ones(8), torch.zeros(8), 12)
assert out.shape == x.shape and isinstance(gain, float)
assert abs(gain - float((8.0 * 12) ** -0.25)) < 1e-12""",
        },
        {
            "name": 'numerical reference',
            "code": """import torch
torch.manual_seed(2)
x, sub = torch.randn(3, 6), torch.randn(3, 6)
gamma, beta = torch.randn(6), torch.randn(6)
out, gain = {fn}(x, sub, gamma, beta, 24)
u = x * ((2.0 * 24) ** 0.25) + sub
mean, var = u.mean(-1, keepdim=True), u.var(-1, keepdim=True, unbiased=False)
ref = gamma * (u - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-6) and gain == float((8.0 * 24) ** -0.25)""",
        },
        {
            "name": 'constant input edge',
            "code": """import torch
x = torch.ones(2, 4)
sub = torch.ones(2, 4)
beta = torch.arange(4, dtype=x.dtype)
out, gain = {fn}(x, sub, torch.randn(4), beta, 1)
assert torch.allclose(out, beta.expand_as(out), atol=1e-6) and torch.isfinite(torch.tensor(gain))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(2, 5, requires_grad=True)
sub = torch.randn(2, 5, requires_grad=True)
gamma = torch.ones(5, requires_grad=True)
beta = torch.zeros(5, requires_grad=True)
out, gain = {fn}(x, sub, gamma, beta, 8)
out.square().sum().backward()
assert all(t.grad is not None for t in (x, sub, gamma, beta)) and gain > 0""",
        },
    ],
    'solution': '''def deepnorm_step(x, sublayer_out, gamma, beta, num_layers, eps=1e-5):
    """执行 DeepNorm 残差缩放与层归一化 (DeepNorm Residual Scaling)。"""
    # Step 1: 计算残差缩放和推荐初始化增益
    alpha = (2.0 * num_layers) ** 0.25
    init_gain = (8.0 * num_layers) ** -0.25
    residual = alpha * x + sublayer_out

    # Step 2: 沿最后一维计算总体方差的 LayerNorm
    mean = residual.mean(dim=-1, keepdim=True)
    variance = residual.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (residual - mean) / torch.sqrt(variance + eps)

    # Step 3: 应用逐特征仿射变换
    out = gamma * normalized + beta
    return out, float(init_gain)''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 16, 64)
sublayer_out = torch.randn_like(x)
out, init_gain = deepnorm_step(x, sublayer_out, torch.ones(64), torch.zeros(64), 48)
print("Output shape:", out.shape)
print("Initialization gain:", init_gain)""",
    'explanation': r"""### 核心思路
DeepNorm 在归一化前构造 $u=\alpha x+f(x)$，其中 $\alpha=(2N)^{1/4}$，并返回推荐初始化增益 $(8N)^{-1/4}$ 以稳定深层网络。

### 代码步骤拆解
- **Step 1（残差缩放）**：按层数计算 `alpha` 和 `init_gain`，并得到与 `x` 同形状的 `residual`。
- **Step 2（层归一化）**：沿最后一维以 `keepdim=True` 计算均值和总体方差，保持广播维度对齐。
- **Step 3（仿射恢复）**：用 `(D,)` 的 `gamma`、`beta` 对归一化结果逐特征缩放和平移。

### 重点与易错点分析
- **注意点 1（方差定义）**：LayerNorm 使用 `unbiased=False` 的总体方差，不应采用默认无偏估计。
- **注意点 2（稳定常数）**：`eps` 必须加在方差内部再开方，常量输入时才能避免除零。
- **注意点 3（返回类型）**：初始化增益用于参数初始化配置，按题意返回 Python `float` 而非参与梯度的张量。

### 复杂度分析
- **时间复杂度**：$O(N_x)$，主要开销是最后一维的均值和方差归约。
- **空间复杂度**：$O(N_x)$，需保存残差与归一化激活。""",
}

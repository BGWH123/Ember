"""Adaptive LayerNorm Zero (adaLN-Zero) task."""

TASK = {
    'title': 'Adaptive LayerNorm Zero (adaLN-Zero)',
    'title_zh': '自适应层归一化 Zero',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'adaln_zero',
    'starter_code': """def adaln_zero(x, cond, W_ada, b_ada):
    pass""",
    'hint': """1. `params = cond @ W_ada + b_ada`  shape `(B, 6*D)`
2. `γ1,β1,α1,... = params.chunk(6, dim=-1)`
3. Manual LayerNorm: `mean/var` over `dim=-1`, `unbiased=False`, `eps=1e-6`
4. `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`""",
    'hint_zh': """1. `params = cond @ W_ada + b_ada`  形状 `(B, 6*D)`
2. `γ1,β1,α1,... = params.chunk(6, dim=-1)`
3. 手动 LayerNorm：`dim=-1` 求 `mean/var`，`unbiased=False`，`eps=1e-6`
4. `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`""",
    'description_en': """Implement the adaLN-Zero conditioning mechanism from DiT (Diffusion Transformer, ICCV 2023).

A linear projection of the conditioning embedding (timestep + class label) regresses scale (γ), shift (β), and gate (α) parameters. These modulate the layer-normed input. The gate is zero-initialized so the block starts as an identity function.

**Signature:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`

**Parameters:**
- `x` — input tokens (B, N, D)
- `cond` — conditioning embedding (B, D)
- `W_ada` — linear weight (D, 6*D), regresses [γ1, β1, α1, γ2, β2, α2]
- `b_ada` — bias (6*D,), zero-initialized in practice

**Returns:** modulated tensor (B, N, D) using the first set of params (γ1, β1, α1)

**Steps:**
1. `params = cond @ W_ada + b_ada` — shape (B, 6*D)
2. Split into 6 chunks of size D: γ1, β1, α1, γ2, β2, α2
3. LayerNorm x manually (no learnable affine): subtract mean, divide by std
4. Modulate: `out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`
5. Return out""",
    'description_zh': """实现 DiT（扩散变换器，ICCV 2023）中的 adaLN-Zero 条件机制。

通过对条件嵌入（时间步 + 类别标签）进行线性投影，回归出缩放（γ）、偏移（β）和门控（α）参数，用于调制经过层归一化的输入。门控参数零初始化，使得模块初始时等价于恒等映射。

**签名:** `adaln_zero(x, cond, W_ada, b_ada) -> Tensor`

**参数:**
- `x` — 输入 token (B, N, D)
- `cond` — 条件嵌入 (B, D)
- `W_ada` — 线性权重 (D, 6*D)，回归 [γ1, β1, α1, γ2, β2, α2]
- `b_ada` — 偏置 (6*D,)，实践中零初始化

**返回:** 使用第一组参数 (γ1, β1, α1) 调制后的张量 (B, N, D)

**步骤:**
1. `params = cond @ W_ada + b_ada` — 形状 (B, 6*D)
2. 沿最后一维切分为 6 块，每块大小 D：γ1, β1, α1, γ2, β2, α2
3. 手动计算 LayerNorm（无可学习仿射参数）：减均值，除以标准差
4. 调制：`out = α1.unsqueeze(1) * (γ1.unsqueeze(1) * x_norm + β1.unsqueeze(1))`
5. 返回 out""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
torch.manual_seed(0)
B, N, D = 2, 16, 64
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
W_ada = torch.randn(D, 6 * D) * 0.02
b_ada = torch.zeros(6 * D)
out = {fn}(x, cond, W_ada, b_ada)
assert out.shape == (B, N, D), f'Expected ({B}, {N}, {D}), got {out.shape}'""",
        },
        {
            "name": 'Zero gate gives zero output',
            "code": """import torch
torch.manual_seed(1)
B, N, D = 2, 8, 32
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
W_ada = torch.zeros(D, 6 * D)
b_ada = torch.zeros(6 * D)
out = {fn}(x, cond, W_ada, b_ada)
assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), 'With zero W_ada and b_ada, gate=0 so output must be all zeros'""",
        },
        {
            "name": 'Identity modulation (alpha=1, gamma=1, beta=0)',
            "code": """import torch
torch.manual_seed(2)
B, N, D = 2, 10, 16
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
# Construct b_ada so that alpha1=1, gamma1=1, beta1=0 for all dims
# params = cond @ 0 + b_ada, chunks: gamma1=b[0:D], beta1=b[D:2D], alpha1=b[2D:3D]
b_ada = torch.zeros(6 * D)
b_ada[0:D] = 1.0      # gamma1 = 1
b_ada[D:2*D] = 0.0    # beta1 = 0
b_ada[2*D:3*D] = 1.0  # alpha1 = 1
W_ada = torch.zeros(D, 6 * D)
out = {fn}(x, cond, W_ada, b_ada)
# Compute expected: manual layer norm of x
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / (var + 1e-6).sqrt()
assert torch.allclose(out, x_norm, atol=1e-5), 'With alpha=1, gamma=1, beta=0, output should equal layer-normed x'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
B, N, D = 2, 6, 8
x = torch.randn(B, N, D, requires_grad=True)
cond = torch.randn(B, D, requires_grad=True)
W_ada = torch.nn.Parameter(torch.randn(D, 6 * D))  # leaf tensor
b_ada = torch.zeros(6 * D, requires_grad=True)
out = {fn}(x, cond, W_ada, b_ada)
out.sum().backward()
assert x.grad is not None, 'No gradient for x'
assert cond.grad is not None, 'No gradient for cond'
assert W_ada.grad is not None, 'No gradient for W_ada'""",
        },
    ],
    'solution': '''def adaln_zero(x, cond, W_ada, b_ada):
    """自适应零初始化层归一化 (adaLN-Zero)。"""
    # Step 1: 从条件向量回归六组调制参数
    params = cond @ W_ada + b_ada
    gamma_1, beta_1, alpha_1, _, _, _ = params.chunk(6, dim=-1)

    # Step 2: 在特征维执行无仿射 LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-6)

    # Step 3: 广播缩放、偏移与门控参数到 token 维
    modulated = gamma_1.unsqueeze(1) * x_norm + beta_1.unsqueeze(1)
    return alpha_1.unsqueeze(1) * modulated''',
    'demo': """torch.manual_seed(42)
B, S, D = 2, 8, 16
x = torch.randn(B, S, D)
cond = torch.randn(B, D)
W_ada = torch.randn(D, 6 * D) * 0.02
b_ada = torch.zeros(6 * D)
out = adaln_zero(x, cond, W_ada, b_ada)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :3])""",
    'explanation': r"""### 核心思路
条件向量经线性映射产生缩放、偏移和门控，输出为 $\alpha_1(\gamma_1\hat{x}+\beta_1)$，其中 $\hat{x}$ 是逐 token 归一化结果。

### 代码步骤拆解
- **Step 1（条件投影）**：`cond @ W_ada + b_ada` 得到 `(B, 6D)`，再切分出第一组调制参数。
- **Step 2（特征归一化）**：沿最后一维计算 `mean` 和 `var`，保持 `(B, S, 1)` 便于广播。
- **Step 3（调制门控）**：三组 `(B, D)` 参数扩展 token 维，输出保持 `(B, S, D)`。

### 重点与易错点分析
- **注意点 1（方差定义）**：LayerNorm 使用总体方差，必须设置 `unbiased=False`，且 `eps` 加在开方前。
- **注意点 2（广播维度）**：参数应在维度 1 使用 `unsqueeze(1)`，不能误扩展批次或特征维。
- **注意点 3（零初始化）**：当门控 `alpha_1` 为零时本函数输出为零，供外部残差分支保持恒等映射。

### 复杂度分析
- **时间复杂度**：$O(BD^2+BSD)$，条件线性投影通常是主要计算。
- **空间复杂度**：$O(BD+BSD)$，保存调制参数和归一化激活。""",
}

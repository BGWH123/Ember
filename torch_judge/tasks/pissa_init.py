"""PiSSA (Principal Singular Value Adaptation) Initialization task."""

TASK = {
    'title': 'PiSSA (Principal Singular Value Adaptation) Initialization',
    'title_zh': 'PiSSA 主奇异值低秩初始化',
    'difficulty': 'Medium',
    'category': '参数高效微调',
    'function_name': 'pissa_init',
    'starter_code': """def pissa_init(W, rank, alpha=1.0):
    pass""",
    'hint': """Split each retained singular value equally between B and A after dividing by LoRA scaling. Subtract the resulting scaled product from W.""",
    'hint_zh': '先除以 LoRA 缩放，再将每个保留奇异值的平方根分别放入 B 和 A，最后从 W 中减去缩放后的乘积。',
    'description_en': """Initialize PiSSA factors from the leading singular components of a weight matrix. Return a residual base weight and low-rank factors whose scaled sum exactly reconstructs the original matrix.""",
    'description_zh': '使用权重矩阵的主奇异分量初始化 PiSSA 因子。返回残差基础权重和低秩因子，使其缩放后的和精确重建原矩阵。',
    'theory_en': """The truncated SVD is the optimal rank-r approximation in Frobenius norm. PiSSA assigns that principal component to the adapter at initialization and leaves the complementary residual in the frozen base.""",
    'theory_zh': '截断 SVD 是 Frobenius 范数下的最优秩 r 近似。PiSSA 在初始化时将主分量分配给适配器，并把其余残差留在冻结基础权重中。',
    'diagram_en': """```mermaid
flowchart LR
W[W] --> S[SVD]
S --> P[top-r principal part]
P --> BA[B and A factors]
W --> R[residual Wres]
BA --> C[Wres+sBA=W]
R --> C
```""",
    'diagram_zh': """```mermaid
flowchart LR
W[W] --> S[SVD]
S --> P[前 r 个主分量]
P --> BA[B 与 A 因子]
W --> R[残差 Wres]
BA --> C[Wres+sBA=W]
R --> C
```""",
    "tests": [
        {
            "name": 'shapes and reconstruction',
            "code": """import torch
torch.manual_seed(0)
W = torch.randn(6, 4)
W_res, B, A = {fn}(W, rank=2, alpha=4.0)
assert W_res.shape == W.shape and B.shape == (6, 2) and A.shape == (2, 4)
reconstructed = W_res + 2.0 * (B @ A)
assert torch.allclose(reconstructed, W, atol=1e-6, rtol=1e-6)""",
        },
        {
            "name": 'principal product exact reference',
            "code": """import torch
torch.manual_seed(1)
W = torch.randn(5, 3, dtype=torch.double)
W_res, B, A = {fn}(W, rank=2, alpha=1.0)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
s = 0.5
expected = (U[:, :2] * S[:2].unsqueeze(0)) @ Vh[:2]
assert B.shape == (5, 2) and A.shape == (2, 3)
assert torch.allclose(s * (B @ A), expected, atol=1e-10)
assert torch.allclose(W_res, W - expected, atol=1e-10)""",
        },
        {
            "name": 'full rank leaves zero residual',
            "code": """import torch
W = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
W_res, B, A = {fn}(W, rank=3, alpha=3.0)
assert B.shape == (3, 3) and A.shape == (3, 3)
assert torch.allclose(W_res, torch.zeros_like(W), atol=1e-6)
assert torch.allclose(B @ A, W, atol=1e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
torch.manual_seed(2)
W = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
W_res, B, A = {fn}(W, rank=2, alpha=2.0)
loss = W_res.square().sum() + B.square().sum() + A.square().sum()
loss.backward()
assert W_res.shape == W.shape
assert W.grad is not None and W.grad.shape == W.shape
assert torch.isfinite(W.grad).all()""",
        },
    ],
    'solution': '''def pissa_init(W, rank, alpha=1.0):
    """PiSSA 主奇异值低秩初始化 (PiSSA (Principal Singular Value Adaptation) Initialization)。"""
    # Step 1: 对基础权重做紧致 SVD，并截取能量最大的前 rank 个奇异分量
    left_vectors, singular_values, right_vectors = torch.linalg.svd(W, full_matrices=False)
    left_rank = left_vectors[:, :rank]
    values_rank = singular_values[:rank]
    right_rank = right_vectors[:rank, :]
    # Step 2: 将每个主奇异值的平方根对称分配给 LoRA 的 B、A 因子
    lora_scale = alpha / rank
    sqrt_values = torch.sqrt(values_rank / lora_scale)
    factor_b = left_rank * sqrt_values.unsqueeze(0)
    factor_a = sqrt_values.unsqueeze(1) * right_rank
    # Step 3: 从 W 中扣除缩放后的 BA 主分量，得到可精确重构 W 的残差基座
    residual_weight = W - lora_scale * (factor_b @ factor_a)
    return (residual_weight, factor_b, factor_a)''',
    'demo': """torch.manual_seed(42)
W = torch.randn(8, 6)
W_res, B, A = pissa_init(W, rank=2, alpha=4.0)
lora_scale = 4.0 / 2
reconstructed = W_res + lora_scale * (B @ A)
print('Factor shapes:', B.shape, A.shape)
print('Reconstruction max error:', (reconstructed - W).abs().max().item())
print('Residual Frobenius norm:', W_res.norm().item())""",
    'explanation': r"""### 核心思路
对 $W\in\mathbb R^{M\times N}$ 做 $W=U\Sigma V^\top$，把前 $r$ 个主奇异分量交给适配器。令缩放 $s=\alpha/r$、$B=U_r\sqrt{\Sigma_r/s}$、$A=\sqrt{\Sigma_r/s}V_r^\top$，则 $sBA=U_r\Sigma_rV_r^\top$；基础权重改为 $W_{res}=W-sBA$ 后，初始化瞬间仍严格满足 $W=W_{res}+sBA$。

### 代码步骤拆解
- **Step 1（提取主奇异子空间）**：`torch.linalg.svd(W, full_matrices=False)` 返回紧致分解，`left_rank`、`values_rank`、`right_rank` 分别截取前 `rank` 列、值和行。
- **Step 2（构造对称低秩因子）**：`sqrt(values_rank/lora_scale)` 同时缩放 `left_rank` 的列和 `right_rank` 的行，使 `factor_b @ factor_a` 在乘 `alpha/rank` 后恢复截断 SVD。
- **Step 3（保留初始函数）**：`residual_weight=W-lora_scale*(factor_b@factor_a)` 把主分量从冻结基座移入可训练适配器，并返回三者。

### 重点与易错点分析
- **注意点 1（LoRA 缩放必须入平方根）**：若只用 $\sqrt{\Sigma_r}$ 构造因子，当 $\alpha/r\ne1$ 时适配器主分量会被额外缩放，重构不再等于原权重。
- **注意点 2（SVD 方向与形状）**：PyTorch 返回的是 `Vh` 而不是 `V`；`factor_b` 应为 `(M,r)`、`factor_a` 应为 `(r,N)`，不能再次转置 `right_vectors[:rank]`。
- **注意点 3（为何不同于零初始化 LoRA）**：普通 LoRA 常把一侧初始化为零；PiSSA 从一开始让适配器承载谱能量最大的方向，基座则保存正交残差。重奇异值附近 SVD 向量不唯一，反向梯度也可能敏感。

### 复杂度分析
- **时间复杂度**：完整紧致 SVD 为 $O(MN\min(M,N))$，截取和重构主分量的 $O(MNr)$ 次于它。
- **空间复杂度**：SVD 因子与残差共需 $O(MN+(M+N)\min(M,N))$；返回的低秩因子额外为 $O((M+N)r)$。""",
}

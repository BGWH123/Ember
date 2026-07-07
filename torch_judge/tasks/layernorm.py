"""Implement LayerNorm task."""

TASK = {
    "title": "Implement LayerNorm",
    "title_zh": "实现 LayerNorm",
    "difficulty": "Medium",
    "category": "归一化",
    "description_en": (
        "Implement Layer Normalization.\n\nLayerNorm normalizes each sample across the feature dimension, stabilizing training without dependence on batch size.\n\n**Signature:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (..., D)\n- `gamma` — scale parameter (D,)\n- `beta` — shift parameter (D,)\n- `eps` — epsilon for numerical stability\n\n**Returns:** normalized tensor, same shape as x\n\n**Constraints:**\n- Normalize over the last dimension\n- Use `unbiased=False` for variance\n- Must match `F.layer_norm`"
    ),
    "description_zh": (
        "实现层归一化。\n\nLayerNorm 对每个样本沿特征维度进行归一化，不依赖批大小即可稳定训练。\n\n**签名:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (..., D)\n- `gamma` — 缩放参数 (D,)\n- `beta` — 偏移参数 (D,)\n- `eps` — 数值稳定性的 epsilon\n\n**返回:** 归一化后的张量，形状与 x 相同\n\n**约束:**\n- 沿最后一个维度归一化\n- 方差使用 `unbiased=False`\n- 必须与 `F.layer_norm` 一致"
    ),
    "function_name": "my_layer_norm",
    "hint": (
        "1. `mean = x.mean(dim=-1, keepdim=True)`\n2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`\n3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`"
    ),
    "hint_zh": (
        "1. `mean = x.mean(dim=-1, keepdim=True)`\n2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`\n3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`"
    ),
    "theory_en": (
        "Layer Normalization normalizes each sample independently across the feature dimension, unlike BatchNorm which normalizes across the batch.\n\n**Formula:**\n$$\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$\nwhere $\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i$ and $\\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2$\n\n**Why LayerNorm over BatchNorm in Transformers?**\n- Sequence lengths vary; batch statistics become unreliable\n- Independent per-sample normalization avoids batch-size dependencies\n- Pre-norm (LN before attention/FFN) is more stable than post-norm for deep networks"
    ),
    "theory_zh": (
        "层归一化对每个样本沿特征维度独立归一化，与 BatchNorm 沿批次维度归一化不同。\n\n**公式：**\n$$\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$\n其中 $\\mu = \\frac{1}{D} \\sum_{i=1}^D x_i$，$\\sigma^2 = \\frac{1}{D} \\sum_{i=1}^D (x_i - \\mu)^2$\n\n**为什么 Transformer 中使用 LayerNorm 而非 BatchNorm？**\n- 序列长度变化；批次统计量不可靠\n- 逐样本独立归一化避免依赖批大小\n- Pre-norm（在注意力/FFN 之前做 LN）比 post-norm 对深层网络更稳定"
    ),
    "diagram_en": (
        "```mermaid\nflowchart TD\n    X[x] --> MEAN[Compute mean<br/>mu = mean(x, dim=-1)]\n    X --> VAR[Compute variance<br/>sigma^2 = var(x, dim=-1)]\n    MEAN --> NORM[Normalize<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]\n    VAR --> NORM\n    GAMMA[gamma] --> SCALE[Scale & Shift<br/>gamma * x_norm + beta]\n    BETA[beta] --> SCALE\n    NORM --> SCALE\n```"
    ),
    "diagram_zh": (
        "```mermaid\nflowchart TD\n    X[x] --> MEAN[计算均值<br/>mu = mean(x, dim=-1)]\n    X --> VAR[计算方差<br/>sigma^2 = var(x, dim=-1)]\n    MEAN --> NORM[归一化<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]\n    VAR --> NORM\n    GAMMA[gamma] --> SCALE[缩放与偏移<br/>gamma * x_norm + beta]\n    BETA[beta] --> SCALE\n    NORM --> SCALE\n```"
    ),
    "tests": [
        {
            "name": "Shape and basic behavior",
            "code": """









import torch
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch vs F.layer_norm'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "With learned parameters",
            "code": """









import torch
x = torch.randn(4, 16)
gamma = torch.randn(16)
beta = torch.randn(16)
out = {fn}(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [16], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
x = torch.randn(2, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''


def my_layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization implementation.
    对每个样本沿特征维度独立归一化，再应用可学习的缩放 γ 和偏移 β。
    与 BatchNorm 不同，LayerNorm 不依赖 batch size，适合变长序列（Transformer）。
    """
    # 沿最后一个维度（特征维度）计算均值
    # x: (B, S, D) 或 (N, D)，mean 结果: (B, S, 1) 或 (N, 1)
    mean = x.mean(dim=-1, keepdim=True)                # μ = (1/D) Σ_i x_i

    # 计算方差（无偏估计，unbiased=False 对应 PyTorch 默认）
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # σ² = (1/D) Σ_i (x_i - μ)²

    # 标准化: (x - μ) / √(σ² + ε)
    # ε 防止除零，保证数值稳定
    x_norm = (x - mean) / torch.sqrt(var + eps)        # (B, S, D)，均值为0，方差为1

    # 可学习的仿射变换: γ * x_norm + β
    # γ 控制输出缩放，β 控制输出偏移。允许网络学习到"不需要归一化"（γ=1, β=0 即恒等）
    return gamma * x_norm + beta                       # (B, S, D)

    
    
    ''',
    "demo": '''








x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))
    
    
    
    
    
    
    
    
    ''',
}

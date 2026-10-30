"""SmoothQuant Activation-Weight Migration task."""

TASK = {
    'title': 'SmoothQuant Activation-Weight Migration',
    'title_zh': 'SmoothQuant 激活-权重迁移',
    'difficulty': 'Medium',
    'category': '量化',
    'function_name': 'smoothquant_transform',
    'starter_code': """def smoothquant_transform(X, W, alpha=0.5, eps=1e-05):
    pass""",
    'hint': """Compute channel maxima over `(B,S)` for activations and over output channels for weights. Divide X and multiply W by the same channel scale.""",
    'hint_zh': '激活沿 `(B,S)` 求逐通道最大值，权重沿输出通道求最大值；用同一通道缩放除激活、乘权重。',
    'description_en': "Implement SmoothQuant's exact activation-to-weight scale migration for a linear layer.",
    'description_zh': '实现 SmoothQuant 在线性层中的精确激活到权重缩放迁移。',
    'theory_en': """Reciprocal channel scaling leaves the linear result unchanged while moving difficult activation magnitude into weights, which are easier to quantize per channel.""",
    'theory_zh': '互逆的通道缩放保持线性层结果不变，同时把难量化的激活幅值迁移到更易逐通道量化的权重中。',
    'diagram_en': """```text
X maxima, W maxima -> channel scale -> X / scale, W * scale -> unchanged XW^T
```""",
    'diagram_zh': """```text
X 最大值、W 最大值 -> 通道尺度 -> X / scale、W * scale -> XW^T 不变
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
X = torch.randn(2, 4, 6)
W = torch.randn(3, 6)
Xs, Ws, scale = {fn}(X, W)
assert Xs.shape == X.shape and Ws.shape == W.shape
assert scale.shape == (6,)""",
        },
        {
            "name": 'exact scale reference',
            "code": """import torch
X = torch.tensor([[[1., -2.], [4., 1.]], [[-3., 0.5], [2., -1.]]])
W = torch.tensor([[2., -8.], [-1., 4.], [0.5, -2.]])
Xs, Ws, scale = {fn}(X, W, alpha=0.25)
am = X.abs().amax(dim=(0, 1)).clamp(min=1e-5)
wm = W.abs().amax(dim=0).clamp(min=1e-5)
ref = am.pow(0.25) / wm.pow(0.75)
assert torch.equal(scale, ref)
assert torch.equal(Xs, X / ref) and torch.equal(Ws, W * ref.unsqueeze(0))""",
        },
        {
            "name": 'zero channels and invariant output',
            "code": """import torch
torch.manual_seed(2)
X = torch.randn(2, 3, 4)
W = torch.randn(5, 4)
X[..., 1] = 0
W[:, 2] = 0
Xs, Ws, scale = {fn}(X, W, eps=1e-4)
assert torch.isfinite(scale).all() and (scale > 0).all()
assert torch.allclose(Xs @ Ws.T, X @ W.T, atol=2e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
X = torch.randn(2, 3, 4, requires_grad=True)
W = torch.randn(5, 4, requires_grad=True)
Xs, Ws, scale = {fn}(X, W, alpha=0.5)
(Xs.square().mean() + Ws.square().mean() + scale.mean()).backward()
assert X.grad is not None and W.grad is not None
assert X.grad.shape == X.shape and W.grad.shape == W.shape""",
        },
    ],
    'solution': '''def smoothquant_transform(X, W, alpha=0.5, eps=1e-05):
    """实现SmoothQuant 激活-权重迁移（SmoothQuant Activation-Weight Migration）。"""
    # Step 1: 分别统计每个输入通道的激活峰值和权重峰值
    act_max = X.abs().amax(dim=(0, 1)).clamp(min=eps)
    weight_max = W.abs().amax(dim=0).clamp(min=eps)
    # Step 2: 按 alpha 平衡激活与权重动态范围，得到逐输入通道迁移尺度
    scale = act_max ** alpha / weight_max ** (1.0 - alpha)
    # Step 3: 对激活和权重施加互逆缩放以保持线性层输出不变
    x_smooth = X / scale
    w_smooth = W * scale.unsqueeze(0)
    return (x_smooth, w_smooth, scale)''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 16, 32)
weight = torch.randn(64, 32)
x_smooth, weight_smooth, scale = smoothquant_transform(x, weight)
max_error = (x_smooth @ weight_smooth.T - x @ weight.T).abs().max()
print("Scale shape:", scale.shape)
print("Scale min/max:", scale.min().item(), scale.max().item())
print("Maximum migration error:", max_error.item())""",
    'explanation': r"""### 核心思路
对线性层 $Y=XW^\top$ 的每个输入通道 $j$，SmoothQuant 选取 $s_j=a_j^\alpha/w_j^{1-\alpha}$，其中 $a_j=\max_{b,s}|X_{bsj}|$、$w_j=\max_o|W_{oj}|$。令 $\widetilde X_j=X_j/s_j$、$\widetilde W_{:j}=W_{:j}s_j$，则通道尺度在乘积中精确抵消：$\widetilde X\widetilde W^\top=XW^\top$。

### 代码步骤拆解
- **Step 1（逐输入通道统计峰值）**：`act_max` 沿批次和序列轴归约 $(B,S,D)$，`weight_max` 沿输出轴归约 $(D_{\text{out}},D)$，两者都得到 $(D,)$。
- **Step 2（计算迁移尺度）**：`scale = act_max ** alpha / weight_max ** (1 - alpha)` 在激活离群值与权重动态范围之间分配量化难度。
- **Step 3（互逆通道缩放）**：`X / scale` 在最后一维广播，`W * scale.unsqueeze(0)` 沿权重列广播，返回变换后的两因子及校准尺度。

### 重点与易错点分析
- **注意点 1（轴必须对应）**：尺度索引对应线性层输入通道，激活要在 $(B,S)$ 上取最大，权重要在输出通道上取最大；若按权重行缩放，乘积无法抵消。
- **注意点 2（零通道保护）**：全零激活列或权重列会产生 0 的幂或除零，先用 `eps` 截断峰值可保证 `scale` 有限且为正，同时不破坏互逆缩放恒等式。
- **注意点 3（$\alpha$ 的含义）**：$\alpha$ 越大越积极地压低激活峰值、把动态范围迁入权重；这有利于 per-tensor 激活量化，而权重通常可用更细的 per-channel 量化吸收差异。

### 复杂度分析
- **时间复杂度**：$O(BSD+D_{\text{out}}D)$，峰值扫描和对两个张量逐元素缩放各访问一次数据。
- **空间复杂度**：输出 `x_smooth`、`w_smooth` 需要 $O(BSD+D_{\text{out}}D)$，通道统计和 `scale` 仅需 $O(D)$。""",
}

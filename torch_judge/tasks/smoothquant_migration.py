"""SmoothQuant activation-weight migration task."""

TASK = {
    "title": "SmoothQuant Activation-Weight Migration",
    "title_zh": "SmoothQuant 激活-权重迁移",
    "difficulty": "Medium",
    "category": "量化",
    "function_name": "smoothquant_transform",
    "starter_code": '''def smoothquant_transform(X, W, alpha=0.5, eps=1e-5):
    pass
''',
    "hint": "Compute channel maxima over `(B,S)` for activations and over output channels for weights. Divide X and multiply W by the same channel scale.",
    "hint_zh": "激活沿 `(B,S)` 求逐通道最大值，权重沿输出通道求最大值；用同一通道缩放除激活、乘权重。",
    "description_en": "Implement SmoothQuant's exact activation-to-weight scale migration for a linear layer.",
    "description_zh": "实现 SmoothQuant 在线性层中的精确激活到权重缩放迁移。",
    "theory_en": "Reciprocal channel scaling leaves the linear result unchanged while moving difficult activation magnitude into weights, which are easier to quantize per channel.",
    "theory_zh": "互逆的通道缩放保持线性层结果不变，同时把难量化的激活幅值迁移到更易逐通道量化的权重中。",
    "diagram_en": "```text\nX maxima, W maxima -> channel scale -> X / scale, W * scale -> unchanged XW^T\n```",
    "diagram_zh": "```text\nX 最大值、W 最大值 -> 通道尺度 -> X / scale、W * scale -> XW^T 不变\n```",
    "tests": [
        {
            "name": "output shapes",
            "code": '''import torch
X = torch.randn(2, 4, 6)
W = torch.randn(3, 6)
Xs, Ws, scale = {fn}(X, W)
assert Xs.shape == X.shape and Ws.shape == W.shape
assert scale.shape == (6,)
''',
        },
        {
            "name": "exact scale reference",
            "code": '''import torch
X = torch.tensor([[[1., -2.], [4., 1.]], [[-3., 0.5], [2., -1.]]])
W = torch.tensor([[2., -8.], [-1., 4.], [0.5, -2.]])
Xs, Ws, scale = {fn}(X, W, alpha=0.25)
am = X.abs().amax(dim=(0, 1)).clamp(min=1e-5)
wm = W.abs().amax(dim=0).clamp(min=1e-5)
ref = am.pow(0.25) / wm.pow(0.75)
assert torch.equal(scale, ref)
assert torch.equal(Xs, X / ref) and torch.equal(Ws, W * ref.unsqueeze(0))
''',
        },
        {
            "name": "zero channels and invariant output",
            "code": '''import torch
torch.manual_seed(2)
X = torch.randn(2, 3, 4)
W = torch.randn(5, 4)
X[..., 1] = 0
W[:, 2] = 0
Xs, Ws, scale = {fn}(X, W, eps=1e-4)
assert torch.isfinite(scale).all() and (scale > 0).all()
assert torch.allclose(Xs @ Ws.T, X @ W.T, atol=2e-6)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
X = torch.randn(2, 3, 4, requires_grad=True)
W = torch.randn(5, 4, requires_grad=True)
Xs, Ws, scale = {fn}(X, W, alpha=0.5)
(Xs.square().mean() + Ws.square().mean() + scale.mean()).backward()
assert X.grad is not None and W.grad is not None
assert X.grad.shape == X.shape and W.grad.shape == W.shape
''',
        },
    ],
    "solution": '''import torch

def smoothquant_transform(X, W, alpha=0.5, eps=1e-5):
    act_max = X.abs().amax(dim=(0, 1)).clamp(min=eps)
    weight_max = W.abs().amax(dim=0).clamp(min=eps)
    scale = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
    X_smooth = X / scale
    W_smooth = W * scale.unsqueeze(0)
    return X_smooth, W_smooth, scale
''',
    "demo": '''X = torch.randn(2, 128, 256)
W = torch.randn(512, 256)
X_smooth, W_smooth, scale = smoothquant_transform(X, W)
print((X_smooth @ W_smooth.T - X @ W.T).abs().max().item())
''',
    "explanation": "Clamping channel maxima prevents division by zero. Because each input channel is divided and multiplied by the same scale, its contribution to the linear output is preserved.",
}

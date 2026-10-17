"""Tanh Activation task."""

TASK = {
    "title": "Tanh Activation",
    "title_zh": "Tanh 激活函数",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": "Implement the hyperbolic tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Output range is (-1, 1). Use numerically stable computation.",
    "description_zh": "实现双曲正切激活函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))。输出范围为 (-1, 1)。使用数值稳定计算。",
    "function_name": "tanh",
    "hint": "For large |x|, tanh(x) approaches ±1. You can also use the identity tanh(x) = 2*sigmoid(2x) - 1.",
    "hint_zh": "|x| 很大时 tanh(x) → ±1。也可以用恒等式 tanh(x) = 2*σ(2x) - 1。",
    "theory_en": "Tanh is zero-centered (output mean ≈ 0), which helps gradients flow better than sigmoid in hidden layers. It still suffers from vanishing gradient for large |x|.",
    "theory_zh": "Tanh 是零中心化的（输出均值≈0），比 Sigmoid 更适合隐藏层。但在 |x| 很大时仍有梯度消失问题。",
    "tests": [
    {
        "name": "Values across signs and dimensions",
        "code": "x = torch.tensor([[-3., -0.2, 0.], [0.5, 2., 5.]], dtype=torch.float64)\nout = {fn}(x)\nassert out.shape == x.shape\nassert torch.allclose(out, torch.tanh(x), atol=1e-10)"
    },
    {
        "name": "Large magnitudes remain finite",
        "code": "x = torch.tensor([-1000., -100., 0., 100., 1000.])\nout = {fn}(x)\nassert torch.isfinite(out).all()\nassert torch.allclose(out, torch.tanh(x), atol=1e-6)"
    },
    {
        "name": "Finite backward including extreme inputs",
        "code": "x = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64, requires_grad=True)\n{fn}(x).sum().backward()\nexpected = torch.tanh(x.detach())\nassert torch.isfinite(x.grad).all()\nassert torch.allclose(x.grad, 1 - expected.square(), atol=1e-10)"
    }
],
    "solution": '''

import torch

def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    Zero-centered activation with range (-1, 1).
    """
    # 使用 torch.tanh 是最佳实践（内部有CUDNN优化）
    # Using torch.tanh is best practice (CUDNN optimized)
    return torch.tanh(x)

    
    ''',
    "demo": "",
}

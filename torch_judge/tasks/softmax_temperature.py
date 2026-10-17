"""Softmax with Temperature task."""

TASK = {
    "title": "Softmax with Temperature",
    "title_zh": "带温度的 Softmax",
    "difficulty": "Easy",
    "category": "采样与解码",
    "description_en": "Implement softmax with temperature scaling: p_i = exp(z_i / T) / Σ_j exp(z_j / T), where T > 0 is the temperature. Higher T makes distribution more uniform (less confident), lower T makes it sharper (more confident).",
    "description_zh": "实现带温度缩放的 Softmax：p_i = exp(z_i / T) / Σ_j exp(z_j / T)，其中 T > 0 是温度。T 越大分布越均匀（置信度低），T 越小分布越尖锐（置信度高）。",
    "function_name": "softmax_temperature",
    "hint": "Divide logits by T before computing softmax. Use the standard max-subtraction trick for numerical stability.",
    "hint_zh": "计算 softmax 前先将 logits 除以 T。使用标准的 max 减法技巧保证数值稳定。",
    "theory_en": "Temperature scaling controls the sharpness of a probability distribution. It is used in text generation (sampling diversity) and knowledge distillation (softening teacher labels).",
    "theory_zh": "温度缩放控制概率分布的尖锐程度。用于文本生成（控制采样多样性）和知识蒸馏（软化教师标签）。",
    "tests": [
    {
        "name": "Temperature and final dimension",
        "code": "x = torch.tensor([[1., 3., -2.], [2., 1., 0.]], dtype=torch.float64)\nfor temperature in [0.25, 1., 4.]:\n    out = {fn}(x, temperature)\n    assert out.shape == x.shape\n    assert torch.allclose(out, torch.softmax(x / temperature, dim=-1), atol=1e-10)"
    },
    {
        "name": "Large logits and shift invariance",
        "code": "x = torch.tensor([[10000., 10001., 9999.], [-10000., -10000., -10000.]])\nout = {fn}(x, 0.5)\nassert torch.isfinite(out).all()\nassert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-6)\nassert torch.allclose(out, {fn}(x - 10000., 0.5), atol=1e-6)"
    },
    {
        "name": "Backward with nonconstant upstream gradient",
        "code": "x = torch.tensor([0.2, -0.5, 1.], dtype=torch.float64, requires_grad=True)\nreference = x.detach().clone().requires_grad_()\nweights = torch.tensor([1., -2., 3.], dtype=torch.float64)\n({fn}(x, 0.7) * weights).sum().backward()\n(torch.softmax(reference / 0.7, dim=-1) * weights).sum().backward()\nassert torch.allclose(x.grad, reference.grad, atol=1e-10)"
    }
],
    "solution": '''

import torch

def softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Softmax with temperature scaling.
    p_i = exp(z_i / T) / sum(exp(z_j / T))
    T > 1: smoother (more uniform), T < 1: sharper (more peaky).
    """
    # 先将 logits 除以温度，再做数值稳定的 softmax
    # Divide logits by temperature, then numerically stable softmax
    scaled = logits / temperature
    max_val = scaled.max(dim=-1, keepdim=True).values
    exp_scaled = torch.exp(scaled - max_val)
    probs = exp_scaled / exp_scaled.sum(dim=-1, keepdim=True)
    return probs

    
    ''',
    "demo": "",
}

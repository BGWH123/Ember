"""L2 Regularization (Weight Decay) task."""

TASK = {
    "title": "L2 Regularization (Weight Decay)",
    "title_zh": "L2 正则化（Weight Decay）",
    "difficulty": "Easy",
    "category": "正则化",
    "description_en": "Implement L2 regularization (weight decay). Given a weight tensor `w`, compute the L2 penalty as half the sum of squared values: L2 = 0.5 * λ * Σw_i². Returns the penalty (scalar). The 0.5 factor simplifies the gradient to λ*w.",
    "description_zh": "实现 L2 正则化（Weight Decay）。给定权重张量 `w`，计算平方和的一半作为惩罚项：L2 = 0.5 * λ * Σw_i²。返回标量惩罚值。0.5 因子使梯度简化为 λ*w。",
    "function_name": "l2_regularization",
    "hint": "Use `torch.pow()` or `**2`, then `.sum()`. Multiply by 0.5 * lambda.",
    "hint_zh": "使用 `torch.pow()` 或 `**2`，然后 `.sum()`，再乘以 0.5 * λ。",
    "theory_en": "L2 regularization penalizes large weights by adding the sum of squared weights to the loss. It shrinks all weights uniformly toward zero but rarely makes them exactly zero.",
    "theory_zh": "L2 正则化通过惩罚权重的平方和来抑制大权重。它均匀地将所有权重向零收缩，但很少精确为零。",
    "tests": [
    {
        "name": "Signed multidimensional penalty",
        "code": "w = torch.tensor([[-2., 0., 3.], [4., -1., 2.]], dtype=torch.float64)\nexpected = 0.3 * 0.5 * w.abs().pow(2).sum()\nactual = {fn}(w, 0.3)\nassert actual.ndim == 0\nassert torch.allclose(actual, expected, atol=1e-10)"
    },
    {
        "name": "Zero coefficient and zero weights",
        "code": "assert {fn}(torch.tensor([-3., 2.]), 0.).item() == 0.\nassert {fn}(torch.zeros(2, 3), 0.7).item() == 0."
    },
    {
        "name": "Gradient and input preservation",
        "code": "w = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)\nbefore = w.detach().clone()\n{fn}(w, 0.4).backward()\nassert torch.equal(w.detach(), before)\nassert torch.allclose(w.grad, 0.4 * w.detach(), atol=1e-10)"
    }
],
    "solution": '''

import torch

def l2_regularization(w: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """
    Compute L2 regularization penalty (weight decay).
    L2 = 0.5 * lambda_reg * sum(w_i^2)
    The 0.5 factor makes gradient = lambda_reg * w.
    """
    # 计算权重平方和的一半，乘以正则化系数
    # Compute half the sum of squared weights, scaled by lambda
    penalty = 0.5 * lambda_reg * torch.pow(w, 2).sum()
    return penalty

    
    ''',
    "demo": "",
}

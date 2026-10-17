"""L1 Regularization (Lasso) task."""

TASK = {
    "title": "L1 Regularization (Lasso)",
    "title_zh": "L1 正则化（Lasso）",
    "difficulty": "Easy",
    "category": "正则化",
    "description_en": "Implement L1 regularization. Given a weight tensor `w`, compute the L1 penalty as the sum of absolute values: L1 = λ * Σ|w_i|. Returns the penalty (scalar).",
    "description_zh": "实现 L1 正则化。给定权重张量 `w`，计算绝对值之和作为惩罚项：L1 = λ * Σ|w_i|。返回标量惩罚值。",
    "function_name": "l1_regularization",
    "hint": "Use `torch.abs()` and `.sum()`.",
    "hint_zh": "使用 `torch.abs()` 和 `.sum()`。",
    "theory_en": "L1 regularization adds the sum of absolute values of weights to the loss. It promotes sparsity (many weights become exactly zero).",
    "theory_zh": "L1 正则化将权重绝对值之和加入损失函数，促进稀疏性（许多权重精确为0）。",
    "tests": [
    {
        "name": "Signed multidimensional penalty",
        "code": "w = torch.tensor([[-2., 0., 3.], [4., -1., 2.]], dtype=torch.float64)\nexpected = 0.3 * 1.0 * w.abs().pow(1).sum()\nactual = {fn}(w, 0.3)\nassert actual.ndim == 0\nassert torch.allclose(actual, expected, atol=1e-10)"
    },
    {
        "name": "Zero coefficient and zero weights",
        "code": "assert {fn}(torch.tensor([-3., 2.]), 0.).item() == 0.\nassert {fn}(torch.zeros(2, 3), 0.7).item() == 0."
    },
    {
        "name": "Gradient and input preservation",
        "code": "w = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)\nbefore = w.detach().clone()\n{fn}(w, 0.4).backward()\nassert torch.equal(w.detach(), before)\nassert torch.allclose(w.grad, 0.4 * w.detach().sign(), atol=1e-10)"
    }
],
    "solution": '''

import torch

def l1_regularization(w: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """
    Compute L1 regularization penalty.
    L1 = lambda_reg * sum(|w_i|)
    """
    # 计算所有权重元素的绝对值之和
    # Compute the sum of absolute values of all weight elements
    penalty = lambda_reg * torch.abs(w).sum()
    return penalty

    
    ''',
    "demo": "",
}

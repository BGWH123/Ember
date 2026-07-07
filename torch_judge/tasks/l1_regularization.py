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
            "name": "basic",
            "code": "",
        },
        {
            "name": "basic",
            "code": "",
        },
        {
            "name": "basic",
            "code": "",
        },
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

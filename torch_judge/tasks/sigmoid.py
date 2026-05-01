"""Sigmoid Activation task."""

TASK = {
    "title": "Sigmoid Activation",
    "title_zh": "Sigmoid 激活函数",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": "Implement the sigmoid activation function: σ(x) = 1 / (1 + exp(-x)). Use numerically stable computation for large negative inputs (return values in (0, 1)).",
    "description_zh": "实现 Sigmoid 激活函数：σ(x) = 1 / (1 + exp(-x))。对大负数输入使用数值稳定计算，返回值在 (0, 1) 区间。",
    "function_name": "sigmoid",
    "hint": "For stability with large negative x, use `z = torch.exp(-torch.abs(x))` pattern, or clamp inputs before exp.",
    "hint_zh": "对大负数 x，先用 `torch.abs(x)` 取绝对值再做 exp，或对输入做 clamp。",
    "theory_en": "Sigmoid squashes any real number to (0, 1), making it useful for binary classification output layers. However, it suffers from vanishing gradient when |x| is large.",
    "theory_zh": "Sigmoid 将任意实数压缩到 (0, 1) 区间，适用于二分类输出层。但在 |x| 很大时会出现梯度消失问题。",
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

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x))
    For x >= 0: compute as 1 / (1 + exp(-x))
    For x < 0:  compute as exp(x) / (1 + exp(x)) to avoid overflow.
    """
    # 数值稳定的实现：根据输入符号选择计算路径
    # Numerically stable: choose computation path based on sign
    return torch.where(
        x >= 0,
        1.0 / (1.0 + torch.exp(-x)),
        torch.exp(x) / (1.0 + torch.exp(x))
    )

    
    ''',
    "demo": "",
}

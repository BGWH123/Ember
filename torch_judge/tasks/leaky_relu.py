"""Leaky ReLU task."""

TASK = {
    "title": "Leaky ReLU",
    "title_zh": "Leaky ReLU",
    "difficulty": "Easy",
    "category": "激活函数",
    "description_en": "Implement Leaky ReLU: f(x) = x if x > 0, else f(x) = alpha * x, where alpha is a small negative slope (default 0.01). This addresses the 'dying ReLU' problem where neurons can become permanently inactive.",
    "description_zh": "实现 Leaky ReLU：f(x) = x（若 x > 0），否则 f(x) = alpha * x，其中 alpha 是一个小的负斜率（默认 0.01）。这解决了「死亡 ReLU」问题，即神经元可能永久失活。",
    "function_name": "leaky_relu",
    "hint": "Use `torch.where(x > 0, x, alpha * x)`.",
    "hint_zh": "使用 `torch.where(x > 0, x, alpha * x)`。",
    "theory_en": "Leaky ReLU allows a small gradient for negative inputs, preventing neurons from dying. It is a simple and effective improvement over standard ReLU.",
    "theory_zh": "Leaky ReLU 允许负输入有小的梯度，防止神经元死亡。是对标准 ReLU 简单而有效的改进。",
    "tests": [
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

def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x.
    Prevents dying ReLU by allowing small negative gradients.
    """
    # 正数保持原样，负数乘以 alpha（小斜率）
    # Positive: keep as is; Negative: multiply by small slope alpha
    return torch.where(x > 0, x, alpha * x)

    
    ''',
    "demo": "",
}

"""Binary Cross-Entropy Loss task."""

TASK = {
    "title": "Binary Cross-Entropy Loss",
    "title_zh": "二元交叉熵损失",
    "difficulty": "Easy",
    "category": "损失函数",
    "description_en": "Implement binary cross-entropy loss with logits: BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))], where z are logits and y are binary targets in {0,1}. Use numerically stable computation via `logsigmoid`.",
    "description_zh": "实现带 logits 的二元交叉熵损失：BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))]，其中 z 是 logits，y 是 {0,1} 二元目标。使用 `logsigmoid` 做数值稳定计算。",
    "function_name": "binary_cross_entropy",
    "hint": "Use `F.logsigmoid(z)` for log(σ(z)) and `F.logsigmoid(-z)` for log(1-σ(z)). Avoid computing sigmoid explicitly.",
    "hint_zh": "使用 `F.logsigmoid(z)` 计算 log(σ(z))，`F.logsigmoid(-z)` 计算 log(1-σ(z))，避免显式计算 sigmoid。",
    "theory_en": "Binary cross-entropy measures the divergence between predicted probabilities and binary targets. Using logits directly (with logsigmoid) avoids numerical instability from computing exp of large numbers.",
    "theory_zh": "二元交叉熵衡量预测概率与二元目标的差异。直接使用 logits（配合 logsigmoid）可避免大数 exp 的数值不稳定问题。",
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
import torch.nn.functional as F

def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy with logits (numerically stable).
    BCE = -mean[y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))]
    Using logsigmoid avoids explicit sigmoid computation.
    """
    # 使用 logsigmoid 实现数值稳定的 BCE
    # log(σ(z)) = logsigmoid(z), log(1-σ(z)) = logsigmoid(-z)
    log_p = F.logsigmoid(logits)
    log_1p = F.logsigmoid(-logits)
    loss = -(targets * log_p + (1.0 - targets) * log_1p).mean()
    return loss

    
    ''',
    "demo": "",
}

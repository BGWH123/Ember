"""Mean Squared Error (MSE) task."""

TASK = {
    "title": "Mean Squared Error (MSE)",
    "title_zh": "均方误差损失（MSE）",
    "difficulty": "Easy",
    "category": "损失函数",
    "description_en": "Implement mean squared error loss: MSE = (1/n) * Σ(pred_i - target_i)². Compute element-wise squared difference, then mean over all elements.",
    "description_zh": "实现均方误差损失：MSE = (1/n) * Σ(pred_i - target_i)²。先逐元素计算差的平方，再对所有元素求平均。",
    "function_name": "mse_loss",
    "hint": "Use `(pred - target) ** 2` then `.mean()`.",
    "hint_zh": "使用 `(pred - target) ** 2` 然后 `.mean()`。",
    "theory_en": "MSE penalizes large errors quadratically. It is the standard loss for regression tasks. The gradient is linear in the error, making optimization stable.",
    "theory_zh": "MSE 对大误差的惩罚是二次增长。是回归任务的标准损失函数。梯度与误差线性相关，优化过程稳定。",
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

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error: MSE = mean((pred - target)^2)
    Standard regression loss.
    """
    # 计算预测值与目标值的逐元素差，平方后取平均
    # Element-wise difference, square, then mean
    loss = torch.pow(pred - target, 2).mean()
    return loss

    
    ''',
    "demo": "",
}

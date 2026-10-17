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
        "name": "Mean over all tensor elements",
        "code": "p = torch.tensor([[1., -2., 4.], [0., 3., 8.]], dtype=torch.float64)\nt = torch.tensor([[0., 2., 2.], [-1., 1., 5.]], dtype=torch.float64)\nactual = {fn}(p, t)\nassert actual.ndim == 0\nassert torch.allclose(actual, F.mse_loss(p, t), atol=1e-10)"
    },
    {
        "name": "Perfect predictions and scalar",
        "code": "x = torch.tensor([1., 2., 3.])\nassert {fn}(x, x).item() == 0.\nassert torch.allclose({fn}(torch.tensor(4.), torch.tensor(1.)), torch.tensor(9.))"
    },
    {
        "name": "Backward gradient",
        "code": "p = torch.tensor([1., -2., 3.], dtype=torch.float64, requires_grad=True)\nt = torch.tensor([0., 4., 1.], dtype=torch.float64)\n{fn}(p, t).backward()\nassert torch.allclose(p.grad, 2 * (p.detach() - t) / p.numel(), atol=1e-10)"
    }
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

"""SGD with Momentum task."""

TASK = {
    "title": "SGD with Momentum",
    "title_zh": "带动量的 SGD",
    "difficulty": "Easy",
    "category": "优化器与学习率",
    "description_en": "Implement SGD with momentum update step. Given parameter p, gradient g, and velocity v: v_new = momentum * v + g, p_new = p - lr * v_new. This accumulates velocity in directions of consistent gradient, accelerating convergence in relevant directions and dampening oscillations.",
    "description_zh": "实现带动量的 SGD 更新步。给定参数 p、梯度 g 和速度 v：v_new = momentum * v + g，p_new = p - lr * v_new。这会在梯度一致的方向上积累速度，加速收敛并抑制震荡。",
    "function_name": "sgd_momentum",
    "hint": "Update velocity first, then update parameter using the new velocity. Momentum is typically 0.9.",
    "hint_zh": "先更新速度，再用新速度更新参数。动量通常取 0.9。",
    "theory_en": "Momentum simulates the inertia of a heavy ball rolling down the loss landscape. It accelerates in directions of consistent gradient and reduces oscillations in directions with high curvature.",
    "theory_zh": "动量模拟重球在损失 landscape 上滚动的惯性。在梯度一致的方向上加速，在高曲率方向上减少震荡。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch

def sgd_momentum(param: torch.Tensor, grad: torch.Tensor, velocity: torch.Tensor,
                 lr: float, momentum: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SGD with momentum update step.
    v_new = momentum * v + grad
    p_new = p - lr * v_new
    Returns: (param_new, velocity_new)
    """
    # 更新速度：保留上一时刻速度的一部分 + 当前梯度
    # Update velocity: keep part of previous velocity + current gradient
    velocity_new = momentum * velocity + grad

    # 用新速度更新参数
    # Update parameter with new velocity
    param_new = param - lr * velocity_new

    return param_new, velocity_new

    
    ''',
    "demo": "",
}

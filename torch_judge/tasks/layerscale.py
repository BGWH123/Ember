"""LayerScale task."""

TASK = {
    "title": "LayerScale",
    "title_zh": "LayerScale",
    "difficulty": "Medium",
    "category": "训练技巧",
    "description_en": "Implement LayerScale as used in DeiT III and modern Transformers. LayerScale multiplies the output of a sub-layer (e.g., attention or FFN) by a learnable diagonal matrix: output = x + diag(γ) * f(x), where γ is a learnable vector initialized to a small value (e.g., 1e-6). This stabilizes training of deep Transformers.",
    "description_zh": "实现 DeiT III 和现代 Transformer 中使用的 LayerScale。LayerScale 将子层输出（如注意力或 FFN）乘以一个可学习的对角矩阵：output = x + diag(γ) * f(x)，其中 γ 是可学习向量，初始化为小值（如 1e-6）。这稳定了深层 Transformer 的训练。",
    "function_name": "layerscale",
    "hint": "Initialize γ as a small constant (e.g., 1e-6). Multiply f(x) element-wise by γ before adding to the residual.",
    "hint_zh": "将 γ 初始化为小常数（如 1e-6）。在加入残差前，将 f(x) 逐元素乘以 γ。",
    "theory_en": "LayerScale prevents early training instability by initially suppressing the residual branch. As training progresses, γ learns to scale the sub-layer output appropriately, allowing deeper networks to train without warm-up.",
    "theory_zh": "LayerScale 通过初始时抑制残差分支来防止早期训练不稳定。随着训练进行，γ 学习适当缩放子层输出，使深层网络无需 warm-up 即可训练。",
    "tests": [
        {
            "name": "basic",
            "code": "",
        },
    ],
    "solution": '''

import torch
import torch.nn as nn

class LayerScale(nn.Module):
    """
    LayerScale: output = x + gamma * f(x)
    gamma is a learnable per-channel scaling factor, initialized small.
    """
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        # 可学习的逐通道缩放因子，初始化为很小的值
        # Learnable per-channel scale, initialized to a tiny value
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        # gamma 逐元素缩放子层输出，再加回残差
        # Scale sublayer output element-wise, add back residual
        return x + self.gamma * f_x

    
    ''',
    "demo": "",
}

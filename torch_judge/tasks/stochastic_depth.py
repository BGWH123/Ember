"""Stochastic Depth (DropPath) task."""

TASK = {
    "title": "Stochastic Depth (DropPath)",
    "title_zh": "随机深度（DropPath）",
    "difficulty": "Medium",
    "category": "正则化",
    "description_en": "Implement Stochastic Depth (DropPath) regularization. During training, randomly drop entire residual branches with probability `drop_prob`: output = x + b * f(x), where b ~ Bernoulli(1 - drop_prob). During evaluation, use the full network: output = x + (1 - drop_prob) * f(x) (expectation over Bernoulli).",
    "description_zh": "实现随机深度（DropPath）正则化。训练时以概率 `drop_prob` 随机丢弃整个残差分支：output = x + b * f(x)，其中 b ~ Bernoulli(1 - drop_prob)。推理时使用完整网络：output = x + (1 - drop_prob) * f(x)（Bernoulli 的期望）。",
    "function_name": "StochasticDepth",
    "hint": "Use `torch.rand()` to sample a Bernoulli mask. In eval mode, scale f(x) by (1 - drop_prob).",
    "hint_zh": "使用 `torch.rand()` 采样 Bernoulli 掩码。推理模式下将 f(x) 缩放 (1 - drop_prob)。",
    "theory_en": "Stochastic Depth randomly drops layers during training, effectively training an ensemble of networks with varying depth. It reduces overfitting and improves generalization, especially in deep networks like Vision Transformers.",
    "theory_zh": "随机深度在训练时随机丢弃层，相当于训练了一个具有不同深度的网络集成。它减少过拟合、提升泛化，尤其在深层网络（如 ViT）中效果显著。",
    "tests": [
    {
        "name": "Evaluation expectation",
        "code": "layer = {fn}(0.25)\nlayer.eval()\nx = torch.ones(3, 2, 4); branch = torch.full_like(x, 2.)\nassert torch.allclose(layer(x, branch), x + 0.75 * branch)"
    },
    {
        "name": "Per-sample branch dropping",
        "code": "torch.manual_seed(1234)\nlayer = {fn}(0.5)\nlayer.train()\nx = torch.zeros(2048, 3, 4); branch = torch.full_like(x, 2.)\nout = layer(x, branch)\nassert torch.all((out == 0) | (out == 2))\nassert torch.equal(out, out[:, :1, :1].expand_as(out))\nassert abs(out.mean().item() - 1.) < 0.1"
    },
    {
        "name": "Zero and complete drop probabilities",
        "code": "x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)\nbranch = torch.ones_like(x, requires_grad=True)\nfor p in [0., 1.]:\n    layer = {fn}(p)\n    for training in [True, False]:\n        layer.train(training)\n        out = layer(x, branch)\n        assert torch.isfinite(out).all()\n        assert torch.allclose(out, x + (1 - p) * branch)"
    }
],
    "solution": '''

import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath): randomly drop residual branches during training.
    output = x + b * f(x) where b ~ Bernoulli(1 - drop_prob)
    """
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.drop_prob == 0.0:
                return x + f_x
            # 训练时：以概率 drop_prob 丢弃整个分支
            # Training: drop the entire branch with probability drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
            mask = torch.rand(shape, device=x.device) < self.keep_prob
            # 按题目约定保留未缩放的分支；推理时乘以存活概率取期望。
            # Keep surviving branches unscaled; evaluation uses their expectation.
            return x + f_x * mask
        else:
            # 推理时：取期望，输出 = x + (1-p) * f(x)
            # Inference: expectation, output = x + (1-p) * f(x)
            return x + self.keep_prob * f_x

    
    ''',
    "demo": "",
}

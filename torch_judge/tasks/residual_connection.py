"""Residual Connection task."""

TASK = {
    "title": "Residual Connection",
    "title_zh": "残差连接（Residual Connection）",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": "Implement a residual connection: output = x + f(x), where f(x) is a sub-layer (e.g., linear transformation). If the input and output shapes differ, project x with a linear layer before adding. In this exercise, use a bias-free identity-like rectangular projection: copy overlapping channels and fill additional output channels with zero.",
    "description_zh": "实现残差连接：output = x + f(x)，其中 f(x) 是子层（如线性变换）。如果输入和输出维度不同，先对 x 做线性投影再相加。 本题使用不含偏置的矩形单位投影：复制共有通道，新增输出通道填零。",
    "function_name": "residual_connection",
    "hint": "Check if shapes match. If not, create a projection matrix W such that x @ W has the same shape as f(x). Initialize the entire rectangular projection with nn.init.eye_, including zeroing off-diagonal entries.",
    "hint_zh": "检查形状是否匹配。如果不匹配，创建投影矩阵 W 使得 x @ W 与 f(x) 形状相同。 使用 nn.init.eye_ 初始化完整矩形投影，包括将对角线以外的位置置零。",
    "theory_en": "Residual connections allow gradients to flow directly through the network via shortcut paths, enabling training of very deep networks (ResNet, Transformer).",
    "theory_zh": "残差连接通过捷径路径让梯度直接回流，使得极深网络（ResNet、Transformer）可以训练。",
    "tests": [
    {
        "name": "Equal shapes and gradients",
        "code": "x = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], requires_grad=True)\nbranch = torch.tensor([[2., -1., 4.], [3., 2., 1.]], requires_grad=True)\nout = {fn}(x, branch)\nassert torch.allclose(out, x + branch)\nout.sum().backward()\nassert torch.equal(x.grad, torch.ones_like(x))\nassert torch.equal(branch.grad, torch.ones_like(branch))"
    },
    {
        "name": "Projection to narrower channels",
        "code": "x = torch.tensor([[1., 2., 3.], [-2., 4., 1.]])\nbranch = torch.tensor([[0.5, 1.], [2., -1.]])\n# The documented demo initializes the square portion of the projection as identity.\nout = {fn}(x, branch)\nassert out.shape == (2, 2)\nassert torch.allclose(out, x[:, :2] + branch)"
    }
],
    "solution": '''

import torch
import torch.nn as nn

def residual_connection(x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
    """
    Residual connection: output = x + f(x)
    If shapes differ, project x to match f(x) shape.
    """
    # 检查输入和子层输出的形状是否一致
    # Check if input and sublayer output shapes match
    if x.shape == f_x.shape:
        return x + f_x
    else:
        # 维度不匹配时，使用线性投影（这里用随机初始化演示）
        # When dims mismatch, use linear projection (random init for demo)
        in_features = x.shape[-1]
        out_features = f_x.shape[-1]
        proj = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            # Initialize as identity-like for stability demonstration
            nn.init.eye_(proj.weight)
        return proj(x) + f_x

    
    ''',
    "demo": "",
}

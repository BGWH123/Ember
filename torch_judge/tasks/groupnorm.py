"""Group Normalization task."""

TASK = {
    "title": "Group Normalization",
    "title_zh": "组归一化（GroupNorm）",
    "difficulty": "Medium",
    "category": "归一化",
    "function_name": "my_group_norm",
    "starter_code": '''def my_group_norm(x, num_groups, gamma, beta, eps=1e-5):
    pass
''',
    "hint": "Reshape to `(B,G,C/G,H,W)`, compute population mean/variance over the last three dimensions, restore BCHW, and apply channel-wise affine parameters.",
    "hint_zh": "重塑为 `(B,G,C/G,H,W)`，在后三维计算总体均值/方差，恢复 BCHW 后应用逐通道仿射参数。",
    "description_en": "Implement Group Normalization manually for BCHW inputs with channel-wise scale and bias.",
    "description_zh": "为 BCHW 输入手动实现组归一化，并应用逐通道缩放和偏置。",
    "theory_en": "GroupNorm partitions channels into groups and normalizes each group independently over channels and spatial positions. Unlike BatchNorm, it does not depend on other batch samples.",
    "theory_zh": "GroupNorm 将通道划分为若干组，在每组的通道和空间位置上独立归一化；与 BatchNorm 不同，它不依赖其他批次样本。",
    "diagram_en": "```text\nB,C,H,W -> B,G,C/G,H,W -> group mean/var -> B,C,H,W -> gamma,beta\n```",
    "diagram_zh": "```text\nB,C,H,W -> B,G,C/G,H,W -> 组均值/方差 -> B,C,H,W -> gamma,beta\n```",
    "tests": [
        {
            "name": "output shape",
            "code": '''import torch
x = torch.randn(2, 8, 3, 5)
out = {fn}(x, 4, torch.ones(8), torch.zeros(8))
assert out.shape == x.shape
''',
        },
        {
            "name": "matches PyTorch reference",
            "code": '''import torch
torch.manual_seed(0)
x = torch.randn(3, 6, 4, 2)
gamma, beta = torch.randn(6), torch.randn(6)
out = {fn}(x, 3, gamma, beta, 1e-5)
ref = torch.nn.functional.group_norm(x, 3, gamma, beta, 1e-5)
assert torch.allclose(out, ref, atol=1e-6)
''',
        },
        {
            "name": "one channel per group edge",
            "code": '''import torch
x = torch.randn(1, 4, 2, 3)
out = {fn}(x, 4, torch.ones(4), torch.zeros(4))
means = out.mean(dim=(2, 3))
vars = out.var(dim=(2, 3), unbiased=False)
assert torch.allclose(means, torch.zeros_like(means), atol=1e-5) and torch.allclose(vars, torch.ones_like(vars), atol=2e-4)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2, 4, 3, 3, requires_grad=True)
gamma = torch.ones(4, requires_grad=True)
beta = torch.zeros(4, requires_grad=True)
{fn}(x, 2, gamma, beta).square().sum().backward()
assert x.grad is not None and gamma.grad is not None and beta.grad is not None
''',
        },
    ],
    "solution": '''import torch

def my_group_norm(x, num_groups, gamma, beta, eps=1e-5):
    B, C, H, W = x.shape
    grouped = x.reshape(B, num_groups, C // num_groups, H, W)
    mean = grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    normalized = ((grouped - mean) / torch.sqrt(var + eps)).reshape(B, C, H, W)
    return gamma.view(1, C, 1, 1) * normalized + beta.view(1, C, 1, 1)
''',
    "demo": '''x = torch.randn(2, 32, 16, 16)
out = my_group_norm(x, 8, torch.ones(32), torch.zeros(32))
print(out.shape)
''',
    "explanation": "Each group uses one mean and variance per sample. The `(C,)` affine vectors are reshaped so broadcasting applies one learned scale and bias to each channel.",
}

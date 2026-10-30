"""Dynamic Tanh (DyT) task."""

TASK = {
    "title": "Dynamic Tanh (DyT)",
    "title_zh": "动态 Tanh 归一化替代层（DyT）",
    "difficulty": "Easy",
    "category": "归一化",
    "function_name": "dynamic_tanh",
    "starter_code": '''def dynamic_tanh(x, alpha, gamma, beta):
    pass
''',
    "hint": "Scale x by the scalar alpha, apply tanh, then use the broadcastable feature-wise gamma and beta.",
    "hint_zh": "先用标量 alpha 缩放 x，再应用 tanh，最后使用可广播的逐特征 gamma 和 beta。",
    "description_en": "Implement Dynamic Tanh, a simple normalization-layer alternative with a learned input scale and feature-wise affine transform.",
    "description_zh": "实现 Dynamic Tanh：一种使用可学习输入缩放和逐特征仿射变换的归一化层替代方案。",
    "theory_en": "DyT bounds activations with tanh instead of computing sample statistics. Alpha controls saturation globally, while gamma and beta restore per-feature scale and offset.",
    "theory_zh": "DyT 用 tanh 限制激活而不计算样本统计量。alpha 全局控制饱和程度，gamma 和 beta 恢复逐特征尺度与偏移。",
    "diagram_en": "```text\nx -> alpha*x -> tanh -> gamma*(.)+beta\n```",
    "diagram_zh": "```text\nx -> alpha*x -> tanh -> gamma*(.)+beta\n```",
    "tests": [
        {
            "name": "output shape",
            "code": '''import torch
x = torch.randn(2, 3, 7)
out = {fn}(x, torch.tensor([0.5]), torch.ones(7), torch.zeros(7))
assert out.shape == x.shape
''',
        },
        {
            "name": "exact formula",
            "code": '''import torch
torch.manual_seed(1)
x = torch.randn(4, 5)
alpha, gamma, beta = torch.tensor(0.7), torch.randn(5), torch.randn(5)
out = {fn}(x, alpha, gamma, beta)
ref = gamma * torch.tanh(alpha * x) + beta
assert torch.equal(out, ref)
''',
        },
        {
            "name": "zero alpha edge",
            "code": '''import torch
x = torch.tensor([[1e6, -1e6, 3.0]])
beta = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, torch.tensor(0.0), torch.randn(3), beta)
assert torch.equal(out, beta.unsqueeze(0))
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(2, 4, requires_grad=True)
alpha = torch.tensor([0.5], requires_grad=True)
gamma = torch.ones(4, requires_grad=True)
beta = torch.zeros(4, requires_grad=True)
{fn}(x, alpha, gamma, beta).square().sum().backward()
assert all(t.grad is not None for t in (x, alpha, gamma, beta))
''',
        },
    ],
    "solution": '''import torch

def dynamic_tanh(x, alpha, gamma, beta):
    return gamma * torch.tanh(alpha * x) + beta
''',
    "demo": '''x = torch.randn(2, 10, 64)
out = dynamic_tanh(x, torch.tensor(0.5), torch.ones(64), torch.zeros(64))
print(out.shape)
''',
    "explanation": "PyTorch broadcasting applies scalar alpha to every element and the `(D,)` affine parameters along the final feature dimension.",
}

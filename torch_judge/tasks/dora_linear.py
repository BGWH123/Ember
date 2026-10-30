"""DoRA linear layer task."""

TASK = {
    "title": "DoRA (Weight-Decomposed Low-Rank Adaptation)",
    "title_zh": "DoRA（权重分解低秩适配）",
    "difficulty": "Hard",
    "category": "参数高效微调",
    "function_name": "DoRALinear",
    "starter_code": '''import torch
import torch.nn as nn

class DoRALinear(nn.Module):
    def __init__(self, W_0, rank, alpha=1.0, eps=1e-6):
        super().__init__()

    def forward(self, x):
        pass
''',
    "hint": "Register the frozen base matrix as a buffer. Form the adapted direction, normalize each output row, rescale by the trainable magnitude, and apply it to x.",
    "hint_zh": "将冻结基矩阵注册为 buffer。构造适配后的方向，逐输出行归一化，再用可训练幅值缩放并作用于 x。",
    "description_en": "Implement a DoRA linear layer that separates each weight row into a trainable magnitude and a direction adapted by low-rank matrices. The base weight is frozen and there is no bias.",
    "description_zh": "实现 DoRA 线性层：将每个权重行分解为可训练幅值与由低秩矩阵适配的方向。基础权重冻结且不使用偏置。",
    "theory_en": "DoRA writes a weight row as magnitude times a unit direction. LoRA modifies only the direction candidate `V`, after which row normalization and the learned magnitude reconstruct the effective weight.",
    "theory_zh": "DoRA 将权重行表示为幅值乘单位方向。LoRA 只修改方向候选 `V`，随后通过逐行归一化和学习到的幅值重建有效权重。",
    "diagram_en": "```mermaid\nflowchart LR\nW[frozen W0] --> V[V=W0+sBA]\nA[A] --> V\nB[B] --> V\nV --> N[row normalize]\nM[magnitude m] --> R[rescale]\nN --> R\nX[x] --> O[x Wdora^T]\nR --> O\n```",
    "diagram_zh": "```mermaid\nflowchart LR\nW[冻结 W0] --> V[V=W0+sBA]\nA[A] --> V\nB[B] --> V\nV --> N[逐行归一化]\nM[幅值 m] --> R[重新缩放]\nN --> R\nX[x] --> O[x Wdora^T]\nR --> O\n```",
    "tests": [
        {"name": "parameter shapes and freezing", "code": """import torch
import torch.nn as nn
W = torch.randn(5, 7, requires_grad=True)
layer = {fn}(W, rank=3, alpha=2.0)
assert isinstance(layer, nn.Module)
assert layer.W_0.shape == (5, 7) and not layer.W_0.requires_grad
assert layer.m.shape == (5,) and layer.A.shape == (3, 7) and layer.B.shape == (5, 3)
"""},
        {"name": "initial output equals base numerically", "code": """import torch
torch.manual_seed(0)
W = torch.randn(4, 6)
x = torch.randn(3, 6)
layer = {fn}(W, rank=2, eps=1e-8)
out = layer(x)
expected = x @ W.T
assert out.shape == (3, 4)
assert torch.allclose(out, expected, atol=1e-7, rtol=1e-6)
"""},
        {"name": "exact forward formula", "code": """import torch
torch.manual_seed(1)
W = torch.randn(3, 4)
layer = {fn}(W, rank=2, alpha=4.0, eps=1e-6)
with torch.no_grad():
    layer.B.normal_()
x = torch.randn(2, 4)
V = layer.W_0 + 2.0 * (layer.B @ layer.A)
expected = x @ (((layer.m[:, None] / (V.norm(dim=1, keepdim=True) + 1e-6)) * V).T)
assert layer(x).shape == (2, 3)
assert torch.allclose(layer(x), expected, atol=1e-7)
"""},
        {"name": "gradient flow", "code": """import torch
W = torch.randn(4, 5)
layer = {fn}(W, rank=2)
x = torch.randn(3, 5, requires_grad=True)
layer(x).sum().backward()
assert x.grad is not None and x.grad.shape == x.shape
assert layer.m.grad is not None and layer.A.grad is not None and layer.B.grad is not None
assert layer.W_0.grad is None
"""},
    ],
    "solution": '''import torch
import torch.nn as nn

class DoRALinear(nn.Module):
    def __init__(self, W_0, rank, alpha=1.0, eps=1e-6):
        super().__init__()
        frozen = W_0.detach().clone()
        self.register_buffer('W_0', frozen)
        self.m = nn.Parameter(W_0.detach().norm(p=2, dim=1).clone())
        self.A = nn.Parameter(torch.randn(rank, W_0.shape[1], device=W_0.device, dtype=W_0.dtype) * 0.02)
        self.B = nn.Parameter(torch.zeros(W_0.shape[0], rank, device=W_0.device, dtype=W_0.dtype))
        self.scaling = alpha / rank
        self.eps = eps

    def forward(self, x):
        V = self.W_0 + self.scaling * (self.B @ self.A)
        col_norm = V.norm(p=2, dim=1, keepdim=True) + self.eps
        W_dora = (self.m.unsqueeze(1) / col_norm) * V
        return x @ W_dora.T
''',
    "demo": '''W = torch.randn(8, 16)
layer = DoRALinear(W, rank=4)
print(layer(torch.randn(2, 16)).shape)
''',
    "explanation": "Registering `W_0` as a buffer freezes it while preserving device moves and serialization. `m`, `A`, and `B` are the only trainable layer parameters.",
}

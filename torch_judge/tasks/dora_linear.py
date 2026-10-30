"""DoRA (Weight-Decomposed Low-Rank Adaptation) task."""

TASK = {
    'title': 'DoRA (Weight-Decomposed Low-Rank Adaptation)',
    'title_zh': 'DoRA（权重分解低秩适配）',
    'difficulty': 'Hard',
    'category': '参数高效微调',
    'function_name': 'DoRALinear',
    'starter_code': """class DoRALinear(nn.Module):
    pass""",
    'hint': """Register the frozen base matrix as a buffer. Form the adapted direction, normalize each output row, rescale by the trainable magnitude, and apply it to x.""",
    'hint_zh': '将冻结基矩阵注册为 buffer。构造适配后的方向，逐输出行归一化，再用可训练幅值缩放并作用于 x。',
    'description_en': """Implement a DoRA linear layer that separates each weight row into a trainable magnitude and a direction adapted by low-rank matrices. The base weight is frozen and there is no bias.""",
    'description_zh': '实现 DoRA 线性层：将每个权重行分解为可训练幅值与由低秩矩阵适配的方向。基础权重冻结且不使用偏置。',
    'theory_en': """DoRA writes a weight row as magnitude times a unit direction. LoRA modifies only the direction candidate `V`, after which row normalization and the learned magnitude reconstruct the effective weight.""",
    'theory_zh': 'DoRA 将权重行表示为幅值乘单位方向。LoRA 只修改方向候选 `V`，随后通过逐行归一化和学习到的幅值重建有效权重。',
    'diagram_en': """```mermaid
flowchart LR
W[frozen W0] --> V[V=W0+sBA]
A[A] --> V
B[B] --> V
V --> N[row normalize]
M[magnitude m] --> R[rescale]
N --> R
X[x] --> O[x Wdora^T]
R --> O
```""",
    'diagram_zh': """```mermaid
flowchart LR
W[冻结 W0] --> V[V=W0+sBA]
A[A] --> V
B[B] --> V
V --> N[逐行归一化]
M[幅值 m] --> R[重新缩放]
N --> R
X[x] --> O[x Wdora^T]
R --> O
```""",
    "tests": [
        {
            "name": 'parameter shapes and freezing',
            "code": """import torch
import torch.nn as nn
W = torch.randn(5, 7, requires_grad=True)
layer = {fn}(W, rank=3, alpha=2.0)
assert isinstance(layer, nn.Module)
assert layer.W_0.shape == (5, 7) and not layer.W_0.requires_grad
assert layer.m.shape == (5,) and layer.A.shape == (3, 7) and layer.B.shape == (5, 3)""",
        },
        {
            "name": 'initial output equals base numerically',
            "code": """import torch
torch.manual_seed(0)
W = torch.randn(4, 6)
x = torch.randn(3, 6)
layer = {fn}(W, rank=2, eps=1e-8)
out = layer(x)
expected = x @ W.T
assert out.shape == (3, 4)
assert torch.allclose(out, expected, atol=1e-7, rtol=1e-6)""",
        },
        {
            "name": 'exact forward formula',
            "code": """import torch
torch.manual_seed(1)
W = torch.randn(3, 4)
layer = {fn}(W, rank=2, alpha=4.0, eps=1e-6)
with torch.no_grad():
    layer.B.normal_()
x = torch.randn(2, 4)
V = layer.W_0 + 2.0 * (layer.B @ layer.A)
expected = x @ (((layer.m[:, None] / (V.norm(dim=1, keepdim=True) + 1e-6)) * V).T)
assert layer(x).shape == (2, 3)
assert torch.allclose(layer(x), expected, atol=1e-7)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
W = torch.randn(4, 5)
layer = {fn}(W, rank=2)
x = torch.randn(3, 5, requires_grad=True)
layer(x).sum().backward()
assert x.grad is not None and x.grad.shape == x.shape
assert layer.m.grad is not None and layer.A.grad is not None and layer.B.grad is not None
assert layer.W_0.grad is None""",
        },
    ],
    'solution': '''class DoRALinear(nn.Module):
    """实现权重分解低秩线性层 (DoRA Linear Layer)。"""
    def __init__(self, W_0, rank, alpha=1.0, eps=1e-6):
        super().__init__()
        self.register_buffer("W_0", W_0.detach().clone())
        self.m = nn.Parameter(W_0.detach().norm(dim=1).clone())
        self.A = nn.Parameter(torch.randn(rank, W_0.shape[1], device=W_0.device, dtype=W_0.dtype) * 0.02)
        self.B = nn.Parameter(torch.zeros(W_0.shape[0], rank, device=W_0.device, dtype=W_0.dtype))
        self.scaling = alpha / rank
        self.eps = eps

    def forward(self, x):
        # Step 1: 将冻结基权重与低秩方向增量相加
        direction = self.W_0 + self.scaling * (self.B @ self.A)

        # Step 2: 逐输出行归一化方向并恢复可训练幅值
        row_norm = direction.norm(dim=1, keepdim=True) + self.eps
        effective_weight = self.m.unsqueeze(1) * direction / row_norm

        # Step 3: 使用重建权重执行无偏置线性变换
        out = x @ effective_weight.T
        return out''',
    'demo': """torch.manual_seed(42)
base_weight = torch.randn(8, 16)
layer = DoRALinear(base_weight, rank=4)
out = layer(torch.randn(2, 16))
print("Output shape:", out.shape)
print("Trainable parameters:", [name for name, _ in layer.named_parameters()])""",
    'explanation': r"""### 核心思路
DoRA 将每个权重行写成可训练幅值乘单位方向，方向候选为 $V=W_0+(\alpha/r)BA$，最终权重为 $m\,V/(\lVert V
Vert_2+\epsilon)$。

### 代码步骤拆解
- **Step 1（低秩适配）**：`B @ A` 从 `(out, r)` 与 `(r, in)` 生成完整方向增量，并加到冻结 `W_0`。
- **Step 2（幅值方向分解）**：沿输入维计算每个输出行的 `row_norm`，用 `(out,)` 的 `m` 恢复幅值。
- **Step 3（线性映射）**：输入最后一维与 `effective_weight.T` 相乘，输出最后一维为 `out_features`。

### 重点与易错点分析
- **注意点 1（冻结基权重）**：`W_0` 应注册为 buffer，既不接收梯度，又能随模块迁移设备并进入状态字典。
- **注意点 2（初始化等价）**：`B` 初始化为零、`m` 初始化为基权重行范数，使初始映射近似原始线性层。
- **注意点 3（归一化轴）**：应按权重行即 `dim=1` 归一化；按列归一化会改变 DoRA 的输出通道语义。

### 复杂度分析
- **时间复杂度**：$O(rIO+B I O)$，重建低秩增量和输入线性变换是主要开销。
- **空间复杂度**：$O(IO+r(I+O))$，包含冻结基权重、临时完整方向和低秩参数。""",
}

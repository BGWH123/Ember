"""Simple Linear Layer task."""

TASK = {
    'title': 'Simple Linear Layer',
    'title_zh': '简单线性层',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'SimpleLinear',
    'starter_code': """class SimpleLinear:
    def __init__(self, in_features, out_features):
        pass

    def forward(self, x):
        pass""",
    'hint': """`weight` shape `(out, in)`, init `randn * 1/sqrt(in_features)`
`bias` shape `(out,)`, init zeros
Forward: `x @ weight.T + bias`""",
    'hint_zh': """`weight` 形状 `(out, in)`，初始化 `randn * 1/sqrt(in_features)`
`bias` 形状 `(out,)`，初始化为零
前向：`x @ weight.T + bias`""",
    'description_en': """Implement a simple linear (fully connected) layer from scratch.

A linear layer computes `y = x @ W^T + b` with learnable weight and bias tensors.

**Signature:** `SimpleLinear(in_features, out_features)` (class)

**Method:** `forward(x) -> Tensor`
- `x` — input tensor (*, in_features)

**Returns:** output tensor (*, out_features)

**Constraints:**
- Weight shape: (out_features, in_features) with Kaiming scaling
- Bias shape: (out_features,) initialized to zeros
- Both must have `requires_grad=True`""",
    'description_zh': """从零实现简单线性（全连接）层。

线性层使用可学习的权重和偏置张量计算 `y = x @ W^T + b`。

**签名:** `SimpleLinear(in_features, out_features)`（类）

**方法:** `forward(x) -> Tensor`
- `x` — 输入张量 (*, in_features)

**返回:** 输出张量 (*, out_features)

**约束:**
- 权重形状：(out_features, in_features)，使用 Kaiming 缩放
- 偏置形状：(out_features,)，初始化为零
- 两者都必须 `requires_grad=True`""",
    "tests": [
        {
            "name": 'Weight & bias shape',
            "code": """import torch
layer = {fn}(8, 4)
assert layer.weight.shape == (4, 8), f'Weight shape: {layer.weight.shape}'
assert layer.bias.shape == (4,), f'Bias shape: {layer.bias.shape}'
assert layer.weight.requires_grad, 'weight must require grad'
assert layer.bias.requires_grad, 'bias must require grad'""",
        },
        {
            "name": 'Forward pass',
            "code": """import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.T + layer.bias
assert torch.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
y.sum().backward()
assert layer.weight.grad is not None, 'weight.grad is None'
assert layer.bias.grad is not None, 'bias.grad is None'""",
        },
    ],
    'solution': '''class SimpleLinear:
    """从零实现全连接线性层 (Linear Layer)。"""

    def __init__(self, in_features, out_features):
        # Step 1: 按输入维度缩放初始化可训练权重
        scale = 1.0 / math.sqrt(in_features)
        self.weight = (torch.randn(out_features, in_features) * scale).requires_grad_()
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x):
        # Step 2: 将输入与转置权重相乘
        projected = x @ self.weight.T

        # Step 3: 广播加上输出偏置
        out = projected + self.bias
        return out''',
    'demo': """torch.manual_seed(42)
layer = SimpleLinear(in_features=8, out_features=4)
x = torch.randn(2, 8)
out = layer.forward(x)
print("Weight shape:", layer.weight.shape)
print("Output shape:", out.shape)
print("Sample values:", out[0])""",
    'explanation': r"""### 核心思路
线性层计算 $y=xW^\top+b$，其中 `weight` 为 `(out_features, in_features)`，`bias` 为 `(out_features,)`。矩阵乘法保留输入前导维，仅替换最后一维。

### 代码步骤拆解
- **Step 1（参数初始化）**：用 $1/\sqrt{in\_features}$ 缩放随机权重并创建零偏置。
- **Step 2（线性投影）**：将 `x` 与 `weight.T` 相乘得到输出特征。
- **Step 3（偏置广播）**：把 `bias` 加到每个样本位置。

### 重点与易错点分析
- **注意点 1（叶子参数）**：缩放后再调用 `.requires_grad_()`，确保 `weight` 是可直接读取梯度的叶子张量。
- **注意点 2（权重方向）**：存储布局是 `(out, in)`，前向必须使用 `.T`。
- **注意点 3（模块差异）**：该题类不继承 `nn.Module`，因此参数需手动交给优化器。

### 复杂度分析
- **时间复杂度**：$O(M\,in\_features\,out\_features)$，瓶颈是矩阵乘法。
- **空间复杂度**：$O(M\,out\_features)$，主要为输出激活。""",
}

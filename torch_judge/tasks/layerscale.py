"""LayerScale task."""

TASK = {
    'title': 'LayerScale',
    'title_zh': 'LayerScale',
    'difficulty': 'Medium',
    'category': '训练技巧',
    'function_name': 'LayerScale',
    'starter_code': """class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        pass

    def forward(self, x, f_x):
        pass""",
    'hint': """Initialize γ as a small constant (e.g., 1e-6). Multiply f(x) element-wise by γ before adding to the residual.""",
    'hint_zh': '将 γ 初始化为小常数（如 1e-6）。在加入残差前，将 f(x) 逐元素乘以 γ。',
    'description_en': """Implement LayerScale as used in DeiT III and modern Transformers. LayerScale multiplies the output of a sub-layer (e.g., attention or FFN) by a learnable diagonal matrix: output = x + diag(γ) * f(x), where γ is a learnable vector initialized to a small value (e.g., 1e-6). This stabilizes training of deep Transformers.""",
    'description_zh': """实现 DeiT III 和现代 Transformer 中使用的 LayerScale。LayerScale 将子层输出（如注意力或 FFN）乘以一个可学习的对角矩阵：output = x + diag(γ) * f(x)，其中 γ 是可学习向量，初始化为小值（如 1e-6）。这稳定了深层 Transformer 的训练。""",
    'theory_en': """LayerScale prevents early training instability by initially suppressing the residual branch. As training progresses, γ learns to scale the sub-layer output appropriately, allowing deeper networks to train without warm-up.""",
    'theory_zh': 'LayerScale 通过初始时抑制残差分支来防止早期训练不稳定。随着训练进行，γ 学习适当缩放子层输出，使深层网络无需 warm-up 即可训练。',
    "tests": [
        {
            "name": 'Learnable channel scale and residual',
            "code": """layer = {fn}(3, init_value=0.2)
assert isinstance(layer, nn.Module)
assert isinstance(layer.gamma, nn.Parameter)
assert layer.gamma.shape == (3,)
assert torch.allclose(layer.gamma, torch.full((3,), 0.2))
x = torch.arange(24.).reshape(2, 4, 3)
branch = torch.ones_like(x) * 2
assert torch.allclose(layer(x, branch), x + 0.4)""",
        },
        {
            "name": 'Independent channels and parameter gradient',
            "code": """layer = {fn}(3)
assert torch.allclose(layer.gamma, torch.full((3,), 1e-6))
with torch.no_grad():
    layer.gamma.copy_(torch.tensor([0.1, 0.2, -0.3]))
x = torch.ones(2, 3, requires_grad=True)
branch = torch.tensor([[2., 3., 4.], [1., 2., 3.]], requires_grad=True)
out = layer(x, branch)
assert torch.allclose(out, x + branch * torch.tensor([0.1, 0.2, -0.3]))
out.sum().backward()
assert torch.allclose(layer.gamma.grad, branch.detach().sum(0))
assert torch.allclose(x.grad, torch.ones_like(x))
assert torch.allclose(branch.grad, layer.gamma.detach().expand_as(branch))""",
        },
    ],
    'solution': '''class LayerScale(nn.Module):
    """实现逐通道残差缩放 (LayerScale)。"""

    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        # Step 1: 创建小值初始化的可学习通道系数
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x, f_x):
        # Step 2: 广播缩放残差分支
        scaled_branch = self.gamma * f_x

        # Step 3: 将缩放后的分支加回恒等路径
        out = x + scaled_branch
        return out''',
    'demo': """torch.manual_seed(42)
layer = LayerScale(dim=4, init_value=1e-3)
x = torch.randn(2, 3, 4)
branch = torch.randn_like(x)
out = layer(x, branch)
print("Output shape:", out.shape)
print("Gamma values:", layer.gamma.detach())
print("Residual delta:", (out - x).abs().mean().item())""",
    'explanation': r"""### 核心思路
LayerScale 对子层输出执行逐通道缩放 $y=x+\gamma\odot f(x)$，其中 `gamma` 从很小的常数开始学习。`gamma` 的 `(D,)` 形状会广播到任意前导维。

### 代码步骤拆解
- **Step 1（参数初始化）**：构造长度为 `dim` 的 `nn.Parameter`。
- **Step 2（分支缩放）**：计算 `gamma * f_x`，保持与输入相同形状。
- **Step 3（残差合并）**：把缩放分支与 `x` 相加得到 `out`。

### 重点与易错点分析
- **注意点 1（参数注册）**：必须使用 `nn.Parameter`，普通张量不会被优化器收集。
- **注意点 2（广播约束）**：通道必须位于最后一维且等于 `dim`，否则缩放会失败或落在错误轴。
- **注意点 3（小值初始化）**：很小的 `init_value` 让深网训练初期接近恒等映射，之后再由梯度放大有效分支。

### 复杂度分析
- **时间复杂度**：$O(ND)$，逐元素缩放和相加。
- **空间复杂度**：$O(ND)$，训练时保存分支激活用于反向传播。""",
}

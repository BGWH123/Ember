"""Residual Connection task."""

TASK = {
    'title': 'Residual Connection',
    'title_zh': '残差连接（Residual Connection）',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'residual_connection',
    'starter_code': """def residual_connection(x, f_x):
    pass""",
    'hint': """Check if shapes match. If not, create a projection matrix W such that x @ W has the same shape as f(x). Initialize the entire rectangular projection with nn.init.eye_, including zeroing off-diagonal entries.""",
    'hint_zh': '检查形状是否匹配。如果不匹配，创建投影矩阵 W 使得 x @ W 与 f(x) 形状相同。 使用 nn.init.eye_ 初始化完整矩形投影，包括将对角线以外的位置置零。',
    'description_en': """Implement a residual connection: output = x + f(x), where f(x) is a sub-layer (e.g., linear transformation). If the input and output shapes differ, project x with a linear layer before adding. In this exercise, use a bias-free identity-like rectangular projection: copy overlapping channels and fill additional output channels with zero.""",
    'description_zh': """实现残差连接：output = x + f(x)，其中 f(x) 是子层（如线性变换）。如果输入和输出维度不同，先对 x 做线性投影再相加。 本题使用不含偏置的矩形单位投影：复制共有通道，新增输出通道填零。""",
    'theory_en': """Residual connections allow gradients to flow directly through the network via shortcut paths, enabling training of very deep networks (ResNet, Transformer).""",
    'theory_zh': '残差连接通过捷径路径让梯度直接回流，使得极深网络（ResNet、Transformer）可以训练。',
    "tests": [
        {
            "name": 'Equal shapes and gradients',
            "code": """x = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], requires_grad=True)
branch = torch.tensor([[2., -1., 4.], [3., 2., 1.]], requires_grad=True)
out = {fn}(x, branch)
assert torch.allclose(out, x + branch)
out.sum().backward()
assert torch.equal(x.grad, torch.ones_like(x))
assert torch.equal(branch.grad, torch.ones_like(branch))""",
        },
        {
            "name": 'Projection to narrower channels',
            "code": """x = torch.tensor([[1., 2., 3.], [-2., 4., 1.]])
branch = torch.tensor([[0.5, 1.], [2., -1.]])
# The documented demo initializes the square portion of the projection as identity.
out = {fn}(x, branch)
assert out.shape == (2, 2)
assert torch.allclose(out, x[:, :2] + branch)""",
        },
    ],
    'solution': '''def residual_connection(x, f_x):
    """残差连接（Residual Connection） (Residual Connection)。"""
    # Step 1: 通道数一致时直接构造恒等捷径 x + f(x)
    if x.shape == f_x.shape:
        return x + f_x
    else:
        # Step 2: 通道数不同时建立无偏置矩形单位投影以对齐残差支路
        in_features = x.shape[-1]
        out_features = f_x.shape[-1]
        proj = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            nn.init.eye_(proj.weight)
        # Step 3: 将投影后的重叠通道与分支输出逐元素相加
        return proj(x) + f_x''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 4)
branch = torch.randn(2, 3)
out = residual_connection(x, branch)
print("Output shape:", out.shape)
print("Sample values:", out[0])""",
    'explanation': r"""### 核心思路
残差块输出为 $y=P(x)+f(x)$。当两支形状一致时 $P=I$，梯度可沿恒等捷径直接传播；当最后一维不同，本题用矩形单位矩阵 $P$ 对齐通道：前 $\min(D_{in},D_{out})$ 个通道原样复制，扩宽时新增通道为零，缩窄时截去尾部通道。

### 代码步骤拆解
- **Step 1（选择恒等捷径）**：若 `x.shape == f_x.shape`，直接返回 `x+f_x`，不引入任何参数或矩阵乘法。
- **Step 2（构造矩形单位投影）**：由 `in_features=x.shape[-1]` 与 `out_features=f_x.shape[-1]` 建立无偏置 `nn.Linear`，`nn.init.eye_(proj.weight)` 将形状 `(D_out,D_in)` 的权重初始化为矩形单位阵。
- **Step 3（对齐后相加）**：`proj(x)` 把最后一维变为 `D_out`，与 `f_x` 逐元素相加；在缩窄示例中等价于 `x[..., :D_out]+f_x`。

### 重点与易错点分析
- **注意点 1（Linear 权重方向）**：`nn.Linear` 权重存为 `(D_out,D_in)`，前向内部使用 $xW^\top$；若手写 `(D_in,D_out)` 的投影矩阵，转置关系必须对应。
- **注意点 2（矩形 eye 的确切语义）**：扩宽不是复制或随机填充新增通道，而是补零；缩窄则保留输入的前部重叠通道，这使本题投影确定且无偏置。
- **注意点 3（工程中的投影参数）**：这里每次调用临时创建固定初值投影，适合函数题；真实 ResNet/Transformer 会在模块初始化时注册可训练投影，以便跨批次复用、迁移设备并被优化器管理。

### 复杂度分析
- **时间复杂度**：形状相同时逐元素相加为 $O(ND)$；需要投影时通用线性层为 $O(ND_{in}D_{out})$，其中 $N$ 是前导位置数。
- **空间复杂度**：矩形投影权重占 $O(D_{in}D_{out})$，投影结果占 $O(ND_{out})$；恒等分支仅产生 $O(ND)$ 输出。""",
}

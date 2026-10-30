"""Tanh Activation task."""

TASK = {
    'title': 'Tanh Activation',
    'title_zh': 'Tanh 激活函数',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'tanh',
    'starter_code': """def tanh(x):
    pass""",
    'hint': 'For large |x|, tanh(x) approaches ±1. You can also use the identity tanh(x) = 2*sigmoid(2x) - 1.',
    'hint_zh': '|x| 很大时 tanh(x) → ±1。也可以用恒等式 tanh(x) = 2*σ(2x) - 1。',
    'description_en': """Implement the hyperbolic tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). Output range is (-1, 1). Use numerically stable computation.""",
    'description_zh': '实现双曲正切激活函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))。输出范围为 (-1, 1)。使用数值稳定计算。',
    'theory_en': """Tanh is zero-centered (output mean ≈ 0), which helps gradients flow better than sigmoid in hidden layers. It still suffers from vanishing gradient for large |x|.""",
    'theory_zh': 'Tanh 是零中心化的（输出均值≈0），比 Sigmoid 更适合隐藏层。但在 |x| 很大时仍有梯度消失问题。',
    "tests": [
        {
            "name": 'Values across signs and dimensions',
            "code": """x = torch.tensor([[-3., -0.2, 0.], [0.5, 2., 5.]], dtype=torch.float64)
out = {fn}(x)
assert out.shape == x.shape
assert torch.allclose(out, torch.tanh(x), atol=1e-10)""",
        },
        {
            "name": 'Large magnitudes remain finite',
            "code": """x = torch.tensor([-1000., -100., 0., 100., 1000.])
out = {fn}(x)
assert torch.isfinite(out).all()
assert torch.allclose(out, torch.tanh(x), atol=1e-6)""",
        },
        {
            "name": 'Finite backward including extreme inputs',
            "code": """x = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64, requires_grad=True)
{fn}(x).sum().backward()
expected = torch.tanh(x.detach())
assert torch.isfinite(x.grad).all()
assert torch.allclose(x.grad, 1 - expected.square(), atol=1e-10)""",
        },
    ],
    'solution': '''def tanh(x):
    """实现Tanh 激活函数（Tanh Activation）。"""
    # Step 1: 将任意形状的输入逐元素送入避免显式指数溢出的稳定 tanh 内核
    out = torch.tanh(x)
    # Step 2: 保留输入的 dtype、device 与计算图，使导数可按 1 - tanh(x)^2 反传
    # Step 3: 返回压缩到开区间 (-1, 1) 且形状不变的激活张量
    return out''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 3)
out = tanh(x)
print("Output shape:", out.shape)
print("Sample values:", out[0])""",
    'explanation': r"""### 核心思路
双曲正切逐元素计算 $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2\sigma(2x)-1$，把实数映射到 $(-1,1)$。代码调用稳定内核而不直接构造 $e^x$ 与 $e^{-x}$，因此在 `x=±1000` 时仍返回有限的饱和值。

### 代码步骤拆解
- **Step 1（稳定逐元素求值）**：`torch.tanh(x)` 对 `x` 的每个元素独立求双曲正切，不改变任一维度。
- **Step 2（保留自动微分链路）**：`out` 直接连接到 `x`，反向传播使用 $\frac{\partial\tanh x}{\partial x}=1-\tanh^2x$。
- **Step 3（返回饱和激活）**：输出沿用 `x` 的 shape、dtype 和 device，正负大值分别趋近 $1$ 与 $-1$。

### 重点与易错点分析
- **注意点 1（指数溢出）**：按定义直接计算 `exp(x)` 会在大正数处溢出，甚至形成 `inf/inf`；稳定内核通过分段或等价形式规避该问题。
- **注意点 2（饱和区梯度）**：当 $|x|$ 很大时，`out` 接近 $\pm1$，导数接近 0；这是 tanh 固有的梯度消失，而不是数值错误。
- **注意点 3（与 Sigmoid 对比）**：tanh 零中心且值域宽度为 2，但两者都具有指数饱和；隐藏层常偏好 tanh，而二分类概率输出通常使用 sigmoid。

### 复杂度分析
- **时间复杂度**：设输入共有 $N$ 个元素，逐元素内核为 $O(N)$。
- **空间复杂度**：返回张量占 $O(N)$；除输出及反向所需激活外，每个元素仅需常数临时空间。""",
}

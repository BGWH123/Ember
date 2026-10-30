"""Sigmoid Activation task."""

TASK = {
    'title': 'Sigmoid Activation',
    'title_zh': 'Sigmoid 激活函数',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'sigmoid',
    'starter_code': """def sigmoid(x):
    pass""",
    'hint': """For stability with large negative x, use `z = torch.exp(-torch.abs(x))` pattern, or clamp inputs before exp.""",
    'hint_zh': '对大负数 x，先用 `torch.abs(x)` 取绝对值再做 exp，或对输入做 clamp。',
    'description_en': """Implement the sigmoid activation function: σ(x) = 1 / (1 + exp(-x)). Use numerically stable computation for large negative inputs (return values in (0, 1)).""",
    'description_zh': '实现 Sigmoid 激活函数：σ(x) = 1 / (1 + exp(-x))。对大负数输入使用数值稳定计算，返回值在 (0, 1) 区间。',
    'theory_en': """Sigmoid squashes any real number to (0, 1), making it useful for binary classification output layers. However, it suffers from vanishing gradient when |x| is large.""",
    'theory_zh': 'Sigmoid 将任意实数压缩到 (0, 1) 区间，适用于二分类输出层。但在 |x| 很大时会出现梯度消失问题。',
    "tests": [
        {
            "name": 'Values across signs and dimensions',
            "code": """x = torch.tensor([[-3., -0.2, 0.], [0.5, 2., 5.]], dtype=torch.float64)
out = {fn}(x)
assert out.shape == x.shape
assert torch.allclose(out, torch.sigmoid(x), atol=1e-10)""",
        },
        {
            "name": 'Large magnitudes remain finite',
            "code": """x = torch.tensor([-1000., -100., 0., 100., 1000.])
out = {fn}(x)
assert torch.isfinite(out).all()
assert torch.allclose(out, torch.sigmoid(x), atol=1e-6)""",
        },
        {
            "name": 'Finite backward including extreme inputs',
            "code": """x = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64, requires_grad=True)
{fn}(x).sum().backward()
expected = torch.sigmoid(x.detach())
assert torch.isfinite(x.grad).all()
assert torch.allclose(x.grad, expected * (1 - expected), atol=1e-10)""",
        },
    ],
    'solution': '''def sigmoid(x):
    """实现Sigmoid 激活函数（Sigmoid Activation）。"""
    # Step 1: 按符号拆分输入，选择不会产生正指数溢出的等价公式
    positive = x >= 0
    result = torch.empty_like(x)
    # Step 2: 正半轴计算 1/(1+exp(-x))，此时指数参数不大于零
    result[positive] = 1.0 / (1.0 + torch.exp(-x[positive]))
    # Step 3: 负半轴改写为 exp(x)/(1+exp(x)) 并合并两支结果
    exp_x = torch.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result''',
    'demo': """torch.manual_seed(42)
x = torch.tensor([-1000.0, -2.0, 0.0, 2.0, 1000.0])
out = sigmoid(x)
print("Inputs:", x)
print("Stable sigmoid:", out)
print("Matches torch.sigmoid:", torch.allclose(out, torch.sigmoid(x)))""",
    'explanation': r"""### 核心思路
Sigmoid 为 $\sigma(x)=1/(1+e^{-x})$，但该直式在 $x\ll0$ 时会计算巨大的 $e^{-x}$。利用代数等价式 $\sigma(x)=e^x/(1+e^x)$ 处理负半轴，使两条分支的指数参数都不大于 0，从根源上避免 `exp` 上溢。

### 代码步骤拆解
- **Step 1（按符号选择稳定分支）**：`positive = x >= 0` 生成与 `x` 同形状的布尔索引，并分配同 dtype、device 的 `result`。
- **Step 2（计算非负输入）**：对 `x[positive]` 使用 $1/(1+e^{-x})$；这里 $-x\le0$，指数值最多为 1。
- **Step 3（计算负输入并回填）**：令 `exp_x = exp(x[~positive])`，使用 $e^x/(1+e^x)$，再按原索引写回以保持输入布局。

### 重点与易错点分析
- **注意点 1（极值稳定性）**：统一计算 `exp(-x)` 会在 `x=-1000` 处溢出为 `inf`；分支公式只会下溢到 0，而 0 正是该极限的正确浮点近似。
- **注意点 2（零点与导数）**：$x=0$ 被放入正分支并精确得到 $0.5$；自动微分应给出 $\sigma(x)(1-\sigma(x))$，在饱和区趋近 0，体现梯度消失。
- **注意点 3（有限精度边界）**：数学值严格位于 $(0,1)$，但有限精度下极端输入会舍入为恰好 0 或 1；这是 sigmoid 饱和而非 NaN/Inf 数值故障。

### 复杂度分析
- **时间复杂度**：$O(N)$，每个输入元素只进入一个指数分支并执行常数次算术。
- **空间复杂度**：$O(N)$，输出、布尔掩码和分支临时值与输入元素数同阶。""",
}

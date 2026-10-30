"""GELU Activation task."""

TASK = {
    'title': 'GELU Activation',
    'title_zh': 'GELU 激活函数',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'my_gelu',
    'starter_code': """def my_gelu(x):
    pass""",
    'hint': """Exact: `0.5 * x * (1 + torch.erf(x / sqrt(2)))`
Approx: `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`""",
    'hint_zh': """精确版：`0.5 * x * (1 + torch.erf(x / sqrt(2)))`
近似版：`0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`""",
    'description_en': """Implement the GELU activation function.

GELU (Gaussian Error Linear Unit) smoothly gates inputs based on their value, used in transformers like BERT and GPT.

**Signature:** `my_gelu(x) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape

**Returns:** element-wise GELU activation, same shape as input

**Constraints:**
- Exact formula: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- Must match `F.gelu` within 1e-4
- `gelu(0) = 0`""",
    'description_zh': """实现 GELU 激活函数。

GELU（高斯误差线性单元）根据输入值平滑地进行门控，广泛用于 BERT 和 GPT 等 Transformer。

**签名:** `my_gelu(x) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量

**返回:** 逐元素 GELU 激活，形状与输入相同

**约束:**
- 精确公式：`x * 0.5 * (1 + erf(x / sqrt(2)))`
- 必须与 `F.gelu` 误差在 1e-4 以内
- `gelu(0) = 0`""",
    'theory_en': r"""GELU (Gaussian Error Linear Unit) smoothly gates inputs by their probability under a Gaussian distribution.

**Exact Formula:**
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Approximation (faster):**
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)$$

**Properties:**
- Smooth everywhere (unlike ReLU's kink at 0)
- Biased toward 0 for negative inputs (stochastic regularization effect)
- Used in BERT, GPT, ViT, and most modern Transformers""",
    'theory_zh': r"""GELU（高斯误差线性单元）根据输入在高斯分布下的概率平滑地进行门控。

**精确公式：**
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**近似公式（更快）：**
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)$$

**性质：**
- 处处光滑（不同于 ReLU 在 0 处的折点）
- 负输入偏向 0（随机正则化效果）
- 用于 BERT、GPT、ViT 和大多数现代 Transformer""",
    "tests": [
        {
            "name": 'Matches F.gelu',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
out = {fn}(x)
ref = torch.nn.functional.gelu(x)
assert torch.allclose(out, ref, atol=1e-4), 'Does not match F.gelu'""",
        },
        {
            "name": 'gelu(0) = 0',
            "code": """import torch
out = {fn}(torch.tensor([0.0]))
assert torch.allclose(out, torch.tensor([0.0]), atol=1e-7), f'gelu(0) = {out.item()}'""",
        },
        {
            "name": 'Shape preservation',
            "code": """import torch
x = torch.randn(2, 3, 4)
assert {fn}(x).shape == x.shape, 'Shape mismatch'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
x = torch.randn(4, 8, requires_grad=True)
{fn}(x).sum().backward()
assert x.grad is not None and x.grad.shape == x.shape, 'Gradient issue'""",
        },
    ],
    'solution': '''def my_gelu(x):
    """计算精确高斯误差线性单元 (GELU)。"""
    # Step 1: 将输入缩放到标准高斯误差函数的坐标
    scaled = x / math.sqrt(2.0)

    # Step 2: 计算高斯累积分布门控系数
    gate = 0.5 * (1.0 + torch.erf(scaled))

    # Step 3: 用平滑概率门控原输入
    return x * gate''',
    'demo': """torch.manual_seed(42)
x = torch.linspace(-2.0, 2.0, steps=5)
out = my_gelu(x)
reference = F.gelu(x)
print("Output shape:", out.shape)
print("Sample values:", out)
print("Max error:", (out - reference).abs().max().item())""",
    'explanation': r"""### 核心思路
精确 GELU 按 $\operatorname{GELU}(x)=\frac{x}{2}[1+\operatorname{erf}(x/\sqrt{2})]$ 平滑门控每个元素，输出形状保持不变。

### 代码步骤拆解
- **Step 1（输入缩放）**：`scaled` 计算 $x/\sqrt{2}$，与标准正态分布公式对齐。
- **Step 2（门控概率）**：`torch.erf` 生成范围在 0 到 1 的 `gate`。
- **Step 3（逐元素输出）**：`x * gate` 得到与输入同形状的激活值。

### 重点与易错点分析
- **注意点 1（精确公式）**：题目要求 erf 版本，不应替换为带 $0.044715x^3$ 的 tanh 近似。
- **注意点 2（零点性质）**：当 `x=0` 时乘法确保结果严格为 0。
- **注意点 3（自动微分）**：`torch.erf` 可微，能够保持输入梯度链路。

### 复杂度分析
- **时间复杂度**：$O(M)$，其中 $M$ 为输入元素总数。
- **空间复杂度**：$O(M)$，中间缩放值和门控系数与输入等大。""",
}

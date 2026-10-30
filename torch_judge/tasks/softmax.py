"""Implement Softmax task."""

TASK = {
    'title': 'Implement Softmax',
    'title_zh': '实现 Softmax',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'my_softmax',
    'starter_code': """def my_softmax(x, dim=-1):
    pass""",
    'hint': """1. `x_max = x.max(dim=dim, keepdim=True).values`
2. `e_x = exp(x - x_max)`
3. `return e_x / e_x.sum(dim=dim, keepdim=True)`""",
    'hint_zh': """1. `x_max = x.max(dim=dim, keepdim=True).values`
2. `e_x = exp(x - x_max)`
3. `return e_x / e_x.sum(dim=dim, keepdim=True)`""",
    'description_en': """Implement the softmax function.

Softmax converts raw logits into a probability distribution by exponentiating and normalizing, used in classification and attention.

**Signature:** `my_softmax(x, dim=-1) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape
- `dim` — dimension along which to compute softmax

**Returns:** probability tensor (sums to 1 along dim), same shape as input

**Constraints:**
- Subtract max for numerical stability before exp
- Must handle large values without NaN/Inf""",
    'description_zh': """实现 softmax 函数。

Softmax 通过指数化和归一化将原始 logits 转换为概率分布，用于分类和注意力机制。

**签名:** `my_softmax(x, dim=-1) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量
- `dim` — 计算 softmax 的维度

**返回:** 概率张量（沿 dim 求和为 1），形状与输入相同

**约束:**
- 在 exp 之前减去最大值以保证数值稳定
- 必须处理大值而不产生 NaN/Inf""",
    'theory_en': r"""Softmax converts a vector of real values into a probability distribution.

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^C e^{x_j}}$$

**Numerical Stability:**
Direct computation can overflow for large $x_i$. The stable version subtracts $\max(x)$ before exponentiation:
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^C e^{x_j - \max(x)}}$$

**Temperature Scaling:**
$$\text{softmax}_\tau(x_i) = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}$$
- $\tau \to 0$: sharp (argmax-like)
- $\tau \to \infty$: uniform distribution""",
    'theory_zh': r"""Softmax 将实数值向量转换为概率分布。

**公式：**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^C e^{x_j}}$$

**数值稳定性：**
大值 $x_i$ 的直接计算可能溢出。稳定版本在指数化前减去 $\max(x)$：
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^C e^{x_j - \max(x)}}$$

**温度缩放：**
$$\text{softmax}_\tau(x_i) = \frac{e^{x_i / \tau}}{\sum_j e^{x_j / \tau}}$$
- $\tau \to 0$：尖锐（接近 argmax）
- $\tau \to \infty$：均匀分布""",
    'diagram_en': """```mermaid
flowchart TD
    X[x] --> MAX[x_max = max(x)]
    MAX --> SUB[Subtract: x - x_max]
    SUB --> EXP[Exp: e^(x - x_max)]
    EXP --> SUM[Sum: sum(e^(x - x_max))]
    EXP --> DIV[Divide: exp / sum]
    SUM --> DIV
    DIV --> OUT[Softmax output]
```""",
    'diagram_zh': """```mermaid
flowchart TD
    X[x] --> MAX[x_max = max(x)]
    MAX --> SUB[减去: x - x_max]
    SUB --> EXP[指数: e^(x - x_max)]
    EXP --> SUM[求和: sum(e^(x - x_max))]
    EXP --> DIV[相除: exp / sum]
    SUM --> DIV
    DIV --> OUT[Softmax 输出]
```""",
    "tests": [
        {
            "name": 'Basic 1-D',
            "code": """import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'""",
        },
        {
            "name": '2-D along dim=-1',
            "code": """import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'""",
        },
        {
            "name": 'Numerical stability',
            "code": """import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'""",
        },
        {
            "name": 'dim=0 softmax',
            "code": """import torch
torch.manual_seed(3)
x = torch.randn(4, 3)
out = {fn}(x, dim=0)
exp_x = (x - x.max(dim=0, keepdim=True).values).exp()
expected = exp_x / exp_x.sum(dim=0, keepdim=True)
assert torch.allclose(out, expected, atol=1e-5), f'dim=0 softmax failed'
assert torch.allclose(out.sum(dim=0), torch.ones(3), atol=1e-5), 'columns should sum to 1'""",
        },
    ],
    'solution': '''def my_softmax(x, dim=-1):
    """实现实现 Softmax（Implement Softmax）。"""
    # Step 1: 沿归一化轴提取每个切片的最大 logit 作为稳定平移量
    x_max = x.max(dim=dim, keepdim=True).values
    # Step 2: 对平移后的非正 logits 求指数以避免上溢
    exp_x = torch.exp(x - x_max)
    # Step 3: 用同一轴上的指数和归一化为离散概率分布
    return exp_x / exp_x.sum(dim=dim, keepdim=True)''',
    'demo': r"""torch.manual_seed(42)
x = torch.tensor([[1000.0, 1001.0, 1002.0], [-2.0, 0.0, 2.0]])
probs = my_softmax(x, dim=-1)
print("Probabilities:\n", probs)
print("Row sums:", probs.sum(dim=-1))
print("All finite:", torch.isfinite(probs).all().item())""",
    'explanation': r"""### 核心思路
Softmax 在指定轴上把 logits 转成 $p_i=e^{x_i}/\sum_j e^{x_j}$。利用平移不变性 $\operatorname{softmax}(x+c)=\operatorname{softmax}(x)$，取 $m=\max_jx_j$ 后计算 $p_i=e^{x_i-m}/\sum_je^{x_j-m}$，所有指数参数均不大于 0，且至少一个分母项等于 1。

### 代码步骤拆解
- **Step 1（求切片最大值）**：`x.max(dim=dim, keepdim=True).values` 为每个待归一化切片保留一个 `x_max`，维数保持不变以便逐元素相减。
- **Step 2（稳定指数化）**：`x - x_max` 最大为 0，`exp_x` 因此位于 $(0,1]$，大正 logits 不再导致指数上溢。
- **Step 3（概率归一化）**：沿同一个 `dim` 求 `exp_x.sum(...)` 并逐项相除，使每个切片的概率和为 1，输出形状与 `x` 完全一致。

### 重点与易错点分析
- **注意点 1（max 与 sum 的轴）**：两次归约必须使用同一 `dim`；例如二维张量的 `dim=0` 是按列归一化，不能写死最后一维。
- **注意点 2（为什么分母安全）**：减最大值后至少存在 $e^0=1$，所以有限 logits 下分母不会下溢为 0；`keepdim=True` 则保证任意正负轴索引都能正确广播。
- **注意点 3（耦合梯度）**：softmax 不是逐元素独立激活，其 Jacobian 为 $\partial p_i/\partial x_j=p_i(\delta_{ij}-p_j)$；改变一个 logit 会影响同切片内全部概率。

### 复杂度分析
- **时间复杂度**：$O(N)$，对输入总元素数 $N$ 依次执行最大值归约、指数和求和归约。
- **空间复杂度**：$O(N)$，`exp_x` 与输出占主导，归约结果大小为 $N/C$，其中 $C$ 是 `dim` 的长度。""",
}

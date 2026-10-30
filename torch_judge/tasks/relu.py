"""Implement ReLU task."""

TASK = {
    'title': 'Implement ReLU',
    'title_zh': '实现 ReLU',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'relu',
    'starter_code': """def relu(x):
    pass""",
    'hint': '`relu(x) = max(0, x)` element-wise → `x * (x > 0)`',
    'hint_zh': '`relu(x) = max(0, x)` 逐元素 → `x * (x > 0)`',
    'description_en': """Implement the ReLU activation function.

ReLU (Rectified Linear Unit) outputs the input directly if positive, otherwise zero. It is the most widely used activation in deep learning.

**Signature:** `relu(x) -> Tensor`

**Parameters:**
- `x` — input tensor of any shape

**Returns:** element-wise ReLU activation, same shape as input

**Constraints:**
- `relu(x) = max(0, x)` element-wise
- Must support gradient flow
- Must be efficient on large tensors""",
    'description_zh': """实现 ReLU 激活函数。

ReLU（修正线性单元）在输入为正时直接输出，否则输出零，是深度学习中最广泛使用的激活函数。

**签名:** `relu(x) -> Tensor`

**参数:**
- `x` — 任意形状的输入张量

**返回:** 逐元素 ReLU 激活，形状与输入相同

**约束:**
- `relu(x) = max(0, x)` 逐元素
- 必须支持梯度流
- 在大张量上必须高效""",
    'theory_en': r"""ReLU (Rectified Linear Unit) is the most widely used activation function.

**Formula:**
$$\text{ReLU}(x) = \max(0, x)$$

**Properties:**
- Non-linear but piecewise linear
- Computationally cheap (no exponentials or divisions)
- Sparse activation: ~50% of neurons are inactive on average

**The Dying ReLU Problem:**
If a neuron's weights are updated such that it always outputs 0, gradients are zero and it never recovers. Solutions: LeakyReLU, PReLU, ELU.""",
    'theory_zh': r"""ReLU（修正线性单元）是最广泛使用的激活函数。

**公式：**
$$\text{ReLU}(x) = \max(0, x)$$

**性质：**
- 非线性但分段线性
- 计算廉价（无指数或除法）
- 稀疏激活：平均约 50% 神经元不活跃

**死亡 ReLU 问题：**
如果神经元的权重更新导致其始终输出 0，则梯度为 0 且永远无法恢复。解决方案：LeakyReLU、PReLU、ELU。""",
    "tests": [
        {
            "name": 'Basic values',
            "code": """import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = torch.tensor([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'""",
        },
        {
            "name": '2-D tensor',
            "code": """import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert torch.allclose(out, x.clamp(min=0)), 'Value mismatch on random input'""",
        },
        {
            "name": 'Gradient check',
            "code": """import torch
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'Gradient not computed'
assert x.grad[0] == 0., f'grad at x=-1 should be 0, got {x.grad[0]}'
assert x.grad[2] == 1., f'grad at x=1 should be 1, got {x.grad[2]}'
assert x.grad[3] == 1., f'grad at x=2 should be 1, got {x.grad[3]}'
assert x.grad[1] in (0., 1.), f'grad at x=0 should be 0 or 1, got {x.grad[1]}'""",
        },
        {
            "name": 'Performance',
            "code": """import torch, time
big = torch.randn(1024, 1024)
t0 = time.perf_counter()
for _ in range(100):
    {fn}(big)
elapsed = time.perf_counter() - t0
assert elapsed < 5.0, f'Too slow: {elapsed:.2f}s for 100 iterations'""",
        },
    ],
    'solution': '''def relu(x):
    """实现 ReLU (Implement ReLU)。"""
    # Step 1: 逐元素判断输入是否位于 ReLU 的正半轴
    positive_mask = (x > 0).to(dtype=x.dtype)
    # Step 2: 正值乘 1 保持线性，非正值乘 0 截断
    activated = x * positive_mask
    # Step 3: 返回与输入形状相同的分段线性激活
    return activated''',
    'demo': """torch.manual_seed(42)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print('Input: ', x)
print('Output:', relu(x))""",
    'explanation': r"""### 核心思路
ReLU 是逐元素分段线性函数 $\operatorname{ReLU}(x)=\max(0,x)=x\mathbf 1[x>0]$。正半轴保持恒等映射，负半轴截断为零，因此在不引入指数运算的情况下提供非线性并产生稀疏激活。

### 代码步骤拆解
- **Step 1（划分正负半轴）**：`x > 0` 为每个元素生成布尔掩码，并转成与 `x` 相同 dtype，避免固定转成 float32 破坏 double 或低精度输入。
- **Step 2（执行分段映射）**：`x * positive_mask` 让正元素乘 1 原样通过，让零和负元素乘 0。
- **Step 3（保持张量布局语义）**：所有运算逐元素进行，`activated` 的形状、设备和 dtype 与 `x` 一致，可直接接入后续层。

### 重点与易错点分析
- **注意点 1（零点次梯度）**：ReLU 在 $x=0$ 不可导；该比较掩码实现选择零梯度，测试通常允许框架在 0 处采用 0 或 1 的约定。
- **注意点 2（不要原地修改）**：使用新张量避免覆盖仍被其他计算分支需要的 `x`；原地 ReLU 可能触发 autograd 的版本计数错误。
- **注意点 3（死亡 ReLU）**：负区间导数恒为 0，若神经元长期落在负侧便无法靠该样本恢复；LeakyReLU 在负侧保留小斜率，而 GELU/SiLU 使用平滑门控。

### 复杂度分析
- **时间复杂度**：设输入共有 $N$ 个元素，比较与乘法各线性扫描一次，时间为 $O(N)$。
- **空间复杂度**：非原地输出与掩码均为 $O(N)$；融合 ReLU 内核可避免显式保存转换后的掩码，但反向仍需获知正值位置。""",
}

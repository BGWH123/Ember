"""Leaky ReLU task."""

TASK = {
    'title': 'Leaky ReLU',
    'title_zh': 'Leaky ReLU',
    'difficulty': 'Easy',
    'category': '激活函数',
    'function_name': 'leaky_relu',
    'starter_code': """def leaky_relu(x, alpha=0.01):
    pass""",
    'hint': 'Use `torch.where(x > 0, x, alpha * x)`.',
    'hint_zh': '使用 `torch.where(x > 0, x, alpha * x)`。',
    'description_en': """Implement Leaky ReLU: f(x) = x if x > 0, else f(x) = alpha * x, where alpha is a small negative slope (default 0.01). This addresses the 'dying ReLU' problem where neurons can become permanently inactive.""",
    'description_zh': """实现 Leaky ReLU：f(x) = x（若 x > 0），否则 f(x) = alpha * x，其中 alpha 是一个小的负斜率（默认 0.01）。这解决了「死亡 ReLU」问题，即神经元可能永久失活。""",
    'theory_en': """Leaky ReLU allows a small gradient for negative inputs, preventing neurons from dying. It is a simple and effective improvement over standard ReLU.""",
    'theory_zh': 'Leaky ReLU 允许负输入有小的梯度，防止神经元死亡。是对标准 ReLU 简单而有效的改进。',
    "tests": [
        {
            "name": 'Default and custom negative slope',
            "code": """x = torch.tensor([[-3., -0.5, 0.], [0.5, 2., 4.]])
assert torch.allclose({fn}(x), F.leaky_relu(x, 0.01))
assert torch.allclose({fn}(x, 0.2), F.leaky_relu(x, 0.2))
assert torch.allclose({fn}(x, 0.), torch.relu(x))""",
        },
        {
            "name": 'Gradient away from zero',
            "code": """x = torch.tensor([-3., -1., 2., 4.], requires_grad=True)
before = x.detach().clone()
{fn}(x, 0.15).sum().backward()
assert torch.allclose(x.grad, torch.tensor([0.15, 0.15, 1., 1.]))
assert torch.equal(x.detach(), before)""",
        },
    ],
    'solution': '''def leaky_relu(x, alpha=0.01):
    """实现带负半轴斜率的激活函数 (Leaky ReLU)。"""
    # Step 1: 构造正半轴选择条件
    positive_mask = x > 0

    # Step 2: 对负半轴应用 alpha 缩放
    negative_values = alpha * x

    # Step 3: 逐元素选择两条分支
    out = torch.where(positive_mask, x, negative_values)
    return out''',
    'demo': """torch.manual_seed(42)
x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
out = leaky_relu(x, alpha=0.1)
print("Input:", x)
print("Output:", out)
print("Matches PyTorch:", torch.allclose(out, F.leaky_relu(x, 0.1)))""",
    'explanation': r"""### 核心思路
Leaky ReLU 按 $f(x)=x$（$x>0$）和 $f(x)=\alpha x$（否则）逐元素选择，使负半轴仍保留非零梯度。输入与输出形状完全一致。

### 代码步骤拆解
- **Step 1（条件掩码）**：用 `x > 0` 标记正半轴元素。
- **Step 2（负分支）**：计算 `alpha * x` 作为非正元素候选值。
- **Step 3（逐元素选择）**：`torch.where` 合并两条可微分支。

### 重点与易错点分析
- **注意点 1（零点约定）**：题目使用严格 `x > 0`，零点走负分支，但函数值仍为零。
- **注意点 2（非原地实现）**：避免覆盖 `x`，从而保留叶子张量和自动求导所需值。
- **注意点 3（alpha 含义）**：`alpha=0` 退化为 ReLU；小正值可缓解死亡神经元。

### 复杂度分析
- **时间复杂度**：$O(N)$，每个元素执行一次比较和选择。
- **空间复杂度**：$O(N)$，保存布尔掩码及反向传播状态。""",
}

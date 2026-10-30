"""L2 Regularization (Weight Decay) task."""

TASK = {
    'title': 'L2 Regularization (Weight Decay)',
    'title_zh': 'L2 正则化（Weight Decay）',
    'difficulty': 'Easy',
    'category': '正则化',
    'function_name': 'l2_regularization',
    'starter_code': """def l2_regularization(w, lambda_reg):
    pass""",
    'hint': 'Use `torch.pow()` or `**2`, then `.sum()`. Multiply by 0.5 * lambda.',
    'hint_zh': '使用 `torch.pow()` 或 `**2`，然后 `.sum()`，再乘以 0.5 * λ。',
    'description_en': """Implement L2 regularization (weight decay). Given a weight tensor `w`, compute the L2 penalty as half the sum of squared values: L2 = 0.5 * λ * Σw_i². Returns the penalty (scalar). The 0.5 factor simplifies the gradient to λ*w.""",
    'description_zh': '实现 L2 正则化（Weight Decay）。给定权重张量 `w`，计算平方和的一半作为惩罚项：L2 = 0.5 * λ * Σw_i²。返回标量惩罚值。0.5 因子使梯度简化为 λ*w。',
    'theory_en': """L2 regularization penalizes large weights by adding the sum of squared weights to the loss. It shrinks all weights uniformly toward zero but rarely makes them exactly zero.""",
    'theory_zh': 'L2 正则化通过惩罚权重的平方和来抑制大权重。它均匀地将所有权重向零收缩，但很少精确为零。',
    "tests": [
        {
            "name": 'Signed multidimensional penalty',
            "code": """w = torch.tensor([[-2., 0., 3.], [4., -1., 2.]], dtype=torch.float64)
expected = 0.3 * 0.5 * w.abs().pow(2).sum()
actual = {fn}(w, 0.3)
assert actual.ndim == 0
assert torch.allclose(actual, expected, atol=1e-10)""",
        },
        {
            "name": 'Zero coefficient and zero weights',
            "code": """assert {fn}(torch.tensor([-3., 2.]), 0.).item() == 0.
assert {fn}(torch.zeros(2, 3), 0.7).item() == 0.""",
        },
        {
            "name": 'Gradient and input preservation',
            "code": """w = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)
before = w.detach().clone()
{fn}(w, 0.4).backward()
assert torch.equal(w.detach(), before)
assert torch.allclose(w.grad, 0.4 * w.detach(), atol=1e-10)""",
        },
    ],
    'solution': '''def l2_regularization(w, lambda_reg):
    """计算 L2 权重正则项 (L2 Regularization)。"""
    # Step 1: 逐元素计算权重平方
    squared_weights = w.pow(2)

    # Step 2: 汇总后应用二分之一系数与正则强度
    penalty = 0.5 * lambda_reg * squared_weights.sum()
    return penalty''',
    'demo': """torch.manual_seed(42)
w = torch.randn(3, 4)
penalty = l2_regularization(w, lambda_reg=0.1)
print("Weight shape:", w.shape)
print("L2 penalty:", penalty.item())
print("Zero coefficient:", l2_regularization(w, 0.0).item())""",
    'explanation': r"""### 核心思路
L2 正则化计算 $\frac{\lambda}{2}\sum_iw_i^2$，将任意形状的 `w` 平方并全局求和。系数 $1/2$ 使梯度恰为 $\lambda w$。

### 代码步骤拆解
- **Step 1（平方）**：通过 `w.pow(2)` 生成与输入同形状的平方张量。
- **Step 2（全局归约）**：对所有维度求和得到标量平方范数。
- **Step 3（正则缩放）**：乘以 `0.5 * lambda_reg` 返回最终 `penalty`。

### 重点与易错点分析
- **注意点 1（梯度系数）**：遗漏 $1/2$ 会使梯度变成 $2\lambda w$，与题目定义不符。
- **注意点 2（L2 与权重衰减）**：普通 SGD 下两者常等价，但 AdamW 的解耦权重衰减与把 L2 项加入损失并不完全相同。
- **注意点 3（数值范围）**：大权重平方可能溢出低精度类型，混合精度训练中通常在 FP32 中累计正则项。

### 复杂度分析
- **时间复杂度**：$O(N)$，主要开销是 $N$ 个权重的平方与求和。
- **空间复杂度**：$O(N)$，反向传播需保留输入；标量输出为 $O(1)$。""",
}

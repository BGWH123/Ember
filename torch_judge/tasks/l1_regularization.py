"""L1 Regularization (Lasso) task."""

TASK = {
    'title': 'L1 Regularization (Lasso)',
    'title_zh': 'L1 正则化（Lasso）',
    'difficulty': 'Easy',
    'category': '正则化',
    'function_name': 'l1_regularization',
    'starter_code': """def l1_regularization(w, lambda_reg):
    pass""",
    'hint': 'Use `torch.abs()` and `.sum()`.',
    'hint_zh': '使用 `torch.abs()` 和 `.sum()`。',
    'description_en': """Implement L1 regularization. Given a weight tensor `w`, compute the L1 penalty as the sum of absolute values: L1 = λ * Σ|w_i|. Returns the penalty (scalar).""",
    'description_zh': '实现 L1 正则化。给定权重张量 `w`，计算绝对值之和作为惩罚项：L1 = λ * Σ|w_i|。返回标量惩罚值。',
    'theory_en': """L1 regularization adds the sum of absolute values of weights to the loss. It promotes sparsity (many weights become exactly zero).""",
    'theory_zh': 'L1 正则化将权重绝对值之和加入损失函数，促进稀疏性（许多权重精确为0）。',
    "tests": [
        {
            "name": 'Signed multidimensional penalty',
            "code": """w = torch.tensor([[-2., 0., 3.], [4., -1., 2.]], dtype=torch.float64)
expected = 0.3 * 1.0 * w.abs().pow(1).sum()
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
assert torch.allclose(w.grad, 0.4 * w.detach().sign(), atol=1e-10)""",
        },
    ],
    'solution': '''def l1_regularization(w, lambda_reg):
    """计算 L1 稀疏正则项 (Lasso Regularization)。"""
    # Step 1: 逐元素计算权重绝对值
    absolute_weights = torch.abs(w)

    # Step 2: 汇总所有参数并乘以正则化系数
    penalty = lambda_reg * absolute_weights.sum()
    return penalty''',
    'demo': """torch.manual_seed(42)
w = torch.randn(3, 4)
penalty = l1_regularization(w, lambda_reg=0.1)
print("Weight shape:", w.shape)
print("L1 penalty:", penalty.item())
print("Zero coefficient:", l1_regularization(w, 0.0).item())""",
    'explanation': r"""### 核心思路
L1 正则化计算 $\lambda\sum_i|w_i|$，对任意形状的 `w` 逐元素取绝对值后求和，输出标量 `penalty`。它通过常幅值的次梯度促进参数稀疏。

### 代码步骤拆解
- **Step 1（绝对值）**：用 `torch.abs(w)` 得到与 `w` 同形状的非负张量。
- **Step 2（全局归约）**：调用 `.sum()` 汇总所有参数维度。
- **Step 3（系数缩放）**：乘以 `lambda_reg` 得到最终标量惩罚项。

### 重点与易错点分析
- **注意点 1（零点次梯度）**：PyTorch 在 $w_i=0$ 处采用零次梯度，这是绝对值函数合法的次梯度选择。
- **注意点 2（输入保留）**：使用非原地运算，避免修改待优化参数 `w`。
- **注意点 3（损失组合）**：该函数只返回正则项，训练时应由调用方将其加入任务损失，且不要 `.detach()`。

### 复杂度分析
- **时间复杂度**：$O(N)$，需要访问 $N$ 个权重元素。
- **空间复杂度**：$O(N)$，自动求导通常保存绝对值运算所需状态；输出本身为 $O(1)$。""",
}

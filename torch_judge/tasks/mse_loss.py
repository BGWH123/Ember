"""Mean Squared Error (MSE) task."""

TASK = {
    'title': 'Mean Squared Error (MSE)',
    'title_zh': '均方误差损失（MSE）',
    'difficulty': 'Easy',
    'category': '损失函数',
    'function_name': 'mse_loss',
    'starter_code': """def mse_loss(pred, target):
    pass""",
    'hint': 'Use `(pred - target) ** 2` then `.mean()`.',
    'hint_zh': '使用 `(pred - target) ** 2` 然后 `.mean()`。',
    'description_en': """Implement mean squared error loss: MSE = (1/n) * Σ(pred_i - target_i)². Compute element-wise squared difference, then mean over all elements.""",
    'description_zh': '实现均方误差损失：MSE = (1/n) * Σ(pred_i - target_i)²。先逐元素计算差的平方，再对所有元素求平均。',
    'theory_en': """MSE penalizes large errors quadratically. It is the standard loss for regression tasks. The gradient is linear in the error, making optimization stable.""",
    'theory_zh': 'MSE 对大误差的惩罚是二次增长。是回归任务的标准损失函数。梯度与误差线性相关，优化过程稳定。',
    "tests": [
        {
            "name": 'Mean over all tensor elements',
            "code": """p = torch.tensor([[1., -2., 4.], [0., 3., 8.]], dtype=torch.float64)
t = torch.tensor([[0., 2., 2.], [-1., 1., 5.]], dtype=torch.float64)
actual = {fn}(p, t)
assert actual.ndim == 0
assert torch.allclose(actual, F.mse_loss(p, t), atol=1e-10)""",
        },
        {
            "name": 'Perfect predictions and scalar',
            "code": """x = torch.tensor([1., 2., 3.])
assert {fn}(x, x).item() == 0.
assert torch.allclose({fn}(torch.tensor(4.), torch.tensor(1.)), torch.tensor(9.))""",
        },
        {
            "name": 'Backward gradient',
            "code": """p = torch.tensor([1., -2., 3.], dtype=torch.float64, requires_grad=True)
t = torch.tensor([0., 4., 1.], dtype=torch.float64)
{fn}(p, t).backward()
assert torch.allclose(p.grad, 2 * (p.detach() - t) / p.numel(), atol=1e-10)""",
        },
    ],
    'solution': '''def mse_loss(pred, target):
    """均方误差损失（MSE） (Mean Squared Error (MSE))。"""
    # Step 1: 逐元素计算预测值相对目标值的残差
    residual = pred - target
    # Step 2: 平方残差，使正负误差同向且放大较大的偏差
    squared_error = torch.pow(residual, 2)
    # Step 3: 对张量中的全部元素取算术平均得到标量 MSE
    return squared_error.mean()''',
    'demo': """torch.manual_seed(42)
pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
target = torch.tensor([3.0, -0.5, 2.0, 7.0])
loss = mse_loss(pred, target)
print('逐样本平方误差:', (pred - target).square().tolist())
print('均方误差:', f'{loss.item():.4f}')
print('最大误差样本:', int((pred - target).abs().argmax()))""",
    'explanation': r"""### 核心思路
均方误差将预测与目标的逐元素残差二次化并对全部 $N$ 个元素平均：$L=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2$。平方消除符号并让大误差获得更强惩罚；其梯度 $\frac{\partial L}{\partial\hat y_i}=\frac{2}{N}(\hat y_i-y_i)$ 与残差线性相关。

### 代码步骤拆解
- **Step 1（形成残差）**：`residual = pred - target` 保持输入形状，每个位置表示带符号预测偏差。
- **Step 2（二次惩罚）**：`torch.pow(residual, 2)` 逐元素平方，不在此阶段做任何维度归约。
- **Step 3（全元素平均）**：`.mean()` 同时归约所有轴，返回零维标量，并使梯度按总元素数缩放。

### 重点与易错点分析
- **注意点 1（归约语义）**：本题要求对张量所有元素取平均，不是先按特征求和再按 batch 平均；多维输入的除数是 `pred.numel()`。
- **注意点 2（广播可能掩盖形状错误）**：减法遵循 PyTorch 广播规则，`pred` 与 `target` 形状不同也可能运行成功；任务语义通常要求逐位置对应，应由调用方确保形状匹配。
- **注意点 3（与 MAE 的差异）**：MSE 对离群误差呈二次增长且梯度随残差增大，较 MAE 更敏感；完美预测处损失和梯度均为零。

### 复杂度分析
- **时间复杂度**：$O(N)$，N 为广播后张量元素数，减法、平方与求和各线性扫描一次。
- **空间复杂度**：前向显式保存 `residual` 和 `squared_error` 时为 $O(N)$；最终损失本身是 $O(1)$。""",
}

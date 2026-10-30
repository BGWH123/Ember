"""Binary Cross-Entropy Loss task."""

TASK = {
    'title': 'Binary Cross-Entropy Loss',
    'title_zh': '二元交叉熵损失',
    'difficulty': 'Easy',
    'category': '损失函数',
    'function_name': 'binary_cross_entropy',
    'starter_code': """def binary_cross_entropy(logits, targets):
    pass""",
    'hint': """Use `F.logsigmoid(z)` for log(σ(z)) and `F.logsigmoid(-z)` for log(1-σ(z)). Avoid computing sigmoid explicitly.""",
    'hint_zh': '使用 `F.logsigmoid(z)` 计算 log(σ(z))，`F.logsigmoid(-z)` 计算 log(1-σ(z))，避免显式计算 sigmoid。',
    'description_en': """Implement binary cross-entropy loss with logits: BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))], where z are logits and y are binary targets in {0,1}. Use numerically stable computation via `logsigmoid`.""",
    'description_zh': """实现带 logits 的二元交叉熵损失：BCE = -mean[y*log(σ(z)) + (1-y)*log(1-σ(z))]，其中 z 是 logits，y 是 {0,1} 二元目标。使用 `logsigmoid` 做数值稳定计算。""",
    'theory_en': """Binary cross-entropy measures the divergence between predicted probabilities and binary targets. Using logits directly (with logsigmoid) avoids numerical instability from computing exp of large numbers.""",
    'theory_zh': '二元交叉熵衡量预测概率与二元目标的差异。直接使用 logits（配合 logsigmoid）可避免大数 exp 的数值不稳定问题。',
    "tests": [
        {
            "name": 'Logits and mean reduction',
            "code": """z = torch.tensor([[0., 2., -3.], [1., -2., 0.5]], dtype=torch.float64)
y = torch.tensor([[0., 1., 1.], [1., 0., 1.]], dtype=torch.float64)
actual = {fn}(z, y)
assert actual.ndim == 0
assert torch.allclose(actual, F.binary_cross_entropy_with_logits(z, y), atol=1e-10)""",
        },
        {
            "name": 'Extreme logits stay finite',
            "code": """z = torch.tensor([-1000., 1000., -1000., 1000.])
y = torch.tensor([0., 1., 1., 0.])
out = {fn}(z, y)
assert torch.isfinite(out)
assert torch.allclose(out, torch.tensor(500.), atol=1e-4)""",
        },
        {
            "name": 'Backward gradient',
            "code": """x = torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True)
y = torch.tensor([1., 0., 1.], dtype=torch.float64)
{fn}(x, y).backward()
assert torch.allclose(x.grad, (torch.sigmoid(x.detach()) - y) / 3, atol=1e-10)""",
        },
    ],
    'solution': '''def binary_cross_entropy(logits, targets):
    """带 logits 的二元交叉熵 (Binary Cross-Entropy with Logits)。"""
    # Step 1: 稳定计算正类与负类的对数概率
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits)

    # Step 2: 按目标组合逐元素负对数似然
    element_loss = -(
        targets * log_positive + (1.0 - targets) * log_negative
    )

    # Step 3: 对全部元素取均值
    return element_loss.mean()''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(2, 4)
targets = torch.randint(0, 2, (2, 4)).float()
loss = binary_cross_entropy(logits, targets)
print("Output shape:", loss.shape)
print("Loss:", loss.item())""",
    'explanation': r"""### 核心思路
直接在 logits 域计算 $-[y\log\sigma(z)+(1-y)\log\sigma(-z)]$，避免先求 sigmoid 后取对数的溢出与下溢。

### 代码步骤拆解
- **Step 1（稳定对数概率）**：`F.logsigmoid(logits)` 与其相反数分别表示正负类对数概率。
- **Step 2（目标选择）**：`targets` 逐元素选择正类项，`1-targets` 选择负类项。
- **Step 3（均值归约）**：对 `element_loss` 的全部元素取均值得到标量。

### 重点与易错点分析
- **注意点 1（数值稳定）**：不要显式计算 `log(sigmoid(z))`，极端 logits 会产生 `log(0)`。
- **注意点 2（输入语义）**：`logits` 是未归一化实数，不能先当作概率截断。
- **注意点 3（归约范围）**：`.mean()` 覆盖所有维度，与 PyTorch 默认 reduction 一致。

### 复杂度分析
- **时间复杂度**：$O(N)$，逐元素计算并归约。
- **空间复杂度**：$O(N)$，保存对数概率与逐元素损失。""",
}

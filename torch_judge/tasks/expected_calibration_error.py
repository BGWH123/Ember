"""Expected Calibration Error (ECE) & Brier Score task."""

TASK = {
    'title': 'Expected Calibration Error (ECE) & Brier Score',
    'title_zh': '期望校准误差（ECE）与 Brier 分数',
    'difficulty': 'Medium',
    'category': '评测与基准',
    'function_name': 'calibration_metrics',
    'starter_code': """def calibration_metrics(probs, targets, n_bins=10):
    pass""",
    'hint': """Use top-1 confidence for ECE and one-hot targets for Brier score. Skip empty bins so their means are never evaluated.""",
    'hint_zh': 'ECE 使用 top-1 置信度，Brier 分数使用 one-hot 标签。跳过空桶，避免对其求均值。',
    'description_en': """Compute top-1 expected calibration error using equal-width confidence bins and the multiclass Brier score. Return both as scalar tensors.""",
    'description_zh': '使用等宽置信度分桶计算 top-1 期望校准误差，并计算多分类 Brier 分数。返回两个标量张量。',
    'theory_en': """ECE compares average confidence with empirical accuracy inside each bin and weights the gap by bin frequency. Brier score measures the squared distance from the full probability vector to the one-hot outcome.""",
    'theory_zh': 'ECE 比较每个桶内的平均置信度与经验准确率，并按桶频率加权差距。Brier 分数衡量完整概率向量与 one-hot 结果之间的平方距离。',
    'diagram_en': """```mermaid
flowchart LR
P[probabilities] --> T[top-1 confidence]
T --> B[confidence bins]
B --> E[weighted calibration gaps]
P --> R[squared one-hot error]
```""",
    'diagram_zh': """```mermaid
flowchart LR
P[概率] --> T[top-1 置信度]
T --> B[置信度分桶]
B --> E[加权校准差]
P --> R[one-hot 平方误差]
```""",
    "tests": [
        {
            "name": 'exact two-bin reference',
            "code": """import torch
probs = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
targets = torch.tensor([0, 1, 1, 0])
ece, brier = {fn}(probs, targets, n_bins=2)
expected_ece = 0.5 * abs(0.5 - 0.65) + 0.5 * abs(0.5 - 0.85)
one_hot = torch.nn.functional.one_hot(targets, 2).float()
expected_brier = ((probs - one_hot) ** 2).sum(1).mean()
assert ece.shape == () and brier.shape == ()
assert torch.allclose(ece, torch.tensor(expected_ece), atol=1e-7)
assert torch.allclose(brier, expected_brier)""",
        },
        {
            "name": 'perfect predictions',
            "code": """import torch
probs = torch.eye(3)
targets = torch.arange(3)
ece, brier = {fn}(probs, targets, n_bins=5)
assert ece.shape == () and brier.shape == ()
assert torch.equal(ece, torch.tensor(0.0))
assert torch.equal(brier, torch.tensor(0.0))""",
        },
        {
            "name": 'lower boundary belongs to first bin',
            "code": """import torch
probs = torch.zeros(2, 2)
targets = torch.tensor([0, 1])
ece, brier = {fn}(probs, targets, n_bins=2)
assert ece.shape == () and brier.shape == ()
assert torch.allclose(ece, torch.tensor(0.5))
assert torch.allclose(brier, torch.tensor(1.0))""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
raw = torch.randn(5, 3, requires_grad=True)
probs = raw.softmax(-1)
targets = torch.tensor([0, 1, 2, 1, 0])
ece, brier = {fn}(probs, targets, n_bins=4)
(ece + brier).backward()
assert ece.shape == () and brier.shape == ()
assert raw.grad is not None and raw.grad.shape == raw.shape
assert torch.isfinite(raw.grad).all()""",
        },
    ],
    'solution': '''def calibration_metrics(probs, targets, n_bins=10):
    """计算期望校准误差与 Brier 分数 (ECE and Brier Score)。"""
    # Step 1: 提取 top-1 置信度、预测类别与正确性
    confidence, predictions = probs.max(dim=-1)
    accuracy = (predictions == targets).to(probs.dtype)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)

    # Step 2: 对非空置信度桶累加频率加权校准差
    ece = probs.new_zeros(())
    for bin_idx in range(n_bins):
        lower = boundaries[bin_idx]
        upper = boundaries[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            calibration_gap = (accuracy[in_bin].mean() - confidence[in_bin].mean()).abs()
            ece = ece + in_bin.to(probs.dtype).mean() * calibration_gap

    # Step 3: 与 one-hot 目标计算多分类 Brier 分数
    num_classes = probs.shape[-1]
    one_hot = F.one_hot(targets, num_classes=num_classes).to(probs.dtype)
    brier = (probs - one_hot).square().sum(dim=-1).mean()
    return ece, brier''',
    'demo': """torch.manual_seed(42)
probs = torch.softmax(torch.randn(8, 4), dim=-1)
targets = torch.randint(0, 4, (8,))
ece, brier = calibration_metrics(probs, targets, n_bins=5)
print("ECE:", ece.item())
print("Brier score:", brier.item())""",
    'explanation': r"""### 核心思路
ECE 将 top-1 置信度分桶并累加 $\frac{|B_m|}{N}|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)|$；Brier 分数则度量完整概率向量到 one-hot 标签的平方距离。

### 代码步骤拆解
- **Step 1（置信度提取）**：从 `(B, C)` 概率取最大值与类别，比较 `targets` 得到 `(B,)` 正确性。
- **Step 2（分桶校准）**：逐桶筛选样本，对非空桶计算准确率—置信度差并按样本频率加权。
- **Step 3（概率误差）**：构造 `(B, C)` one-hot 标签，沿类别维求平方和后取批次均值。

### 重点与易错点分析
- **注意点 1（边界唯一归属）**：首桶使用闭区间包含 0，其余桶左开右闭，确保 0 和 1 均被覆盖且样本不重复。
- **注意点 2（空桶处理）**：必须跳过空桶，直接对空张量求均值会产生 NaN 并污染 ECE。
- **注意点 3（梯度特性）**：分桶选择本身不可微，但桶内 `confidence.mean()` 和 Brier 项仍可向概率输入传播梯度。

### 复杂度分析
- **时间复杂度**：$O(BK+BC)$，其中 $K$ 为桶数；逐桶扫描与 Brier 计算构成开销。
- **空间复杂度**：$O(BC)$，one-hot 标签是主要额外张量。""",
}

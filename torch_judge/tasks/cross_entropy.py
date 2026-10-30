"""Cross-Entropy Loss task."""

TASK = {
    'title': 'Cross-Entropy Loss',
    'title_zh': '交叉熵损失',
    'difficulty': 'Easy',
    'category': '损失函数',
    'function_name': 'cross_entropy_loss',
    'starter_code': """def cross_entropy_loss(logits, targets):
    pass""",
    'hint': """1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`
2. `return -log_probs[arange(B), targets].mean()`""",
    'hint_zh': """1. `log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)`
2. `return -log_probs[arange(B), targets].mean()`""",
    'description_en': """Implement cross-entropy loss for classification.

Cross-entropy measures the difference between predicted logits and true class labels. It is the standard loss for classification tasks.

**Signature:** `cross_entropy_loss(logits, targets) -> Tensor`

**Parameters:**
- `logits` — raw scores (B, C) where C is the number of classes
- `targets` — ground-truth class indices (B,)

**Returns:** scalar mean loss

**Constraints:**
- Must be numerically stable (handle large logits)
- Use log-sum-exp trick for stability""",
    'description_zh': """实现分类交叉熵损失。

交叉熵衡量预测 logits 与真实类别标签之间的差异，是分类任务的标准损失函数。

**签名:** `cross_entropy_loss(logits, targets) -> Tensor`

**参数:**
- `logits` — 原始分数 (B, C)，C 为类别数
- `targets` — 真实类别索引 (B,)

**返回:** 标量平均损失

**约束:**
- 必须数值稳定（处理大 logits）
- 使用 log-sum-exp 技巧保证稳定性""",
    'theory_en': r"""Cross-Entropy Loss measures the difference between predicted probability distribution and true labels.

**Formula:**
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$$

For hard labels (one-hot $y$), this simplifies to:
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i, t_i})$$
where $t_i$ is the true class index.

**Log-Sum-Exp Trick:**
To compute $\log \sum_j e^{x_j}$ stably:
$$\text{lse}(x) = \max(x) + \log \sum_j e^{x_j - \max(x)}$$
This prevents overflow when logits are large.""",
    'theory_zh': r"""交叉熵损失衡量预测概率分布与真实标签之间的差异。

**公式：**
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$$

对于硬标签（one-hot $y$），简化为：
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i, t_i})$$
其中 $t_i$ 是真实类别索引。

**Log-Sum-Exp 技巧：**
为稳定计算 $\log \sum_j e^{x_j}$：
$$\text{lse}(x) = \max(x) + \log \sum_j e^{x_j - \max(x)}$$
这防止了大 logits 时的溢出问题。""",
    "tests": [
        {
            "name": 'Matches F.cross_entropy',
            "code": """import torch
torch.manual_seed(0)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
out = {fn}(logits, targets)
ref = torch.nn.functional.cross_entropy(logits, targets)
assert torch.allclose(out, ref, atol=1e-5), f'Mismatch: {out.item():.4f} vs {ref.item():.4f}'""",
        },
        {
            "name": 'Numerical stability',
            "code": """import torch
logits = torch.tensor([[1000., 0., 0.], [0., 1000., 0.]])
targets = torch.tensor([0, 1])
out = {fn}(logits, targets)
assert not torch.isnan(out), 'NaN with large logits'
assert not torch.isinf(out), 'Inf with large logits'
assert out.item() < 0.01, 'Should be ~0 for confident correct predictions'""",
        },
        {
            "name": 'Scalar output',
            "code": """import torch
out = {fn}(torch.randn(8, 5), torch.randint(0, 5, (8,)))
assert out.dim() == 0, 'Loss must be a scalar'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
logits = torch.randn(8, 5, requires_grad=True)
targets = torch.randint(0, 5, (8,))
{fn}(logits, targets).backward()
assert logits.grad is not None, 'logits.grad is None'""",
        },
    ],
    'solution': '''def cross_entropy_loss(logits, targets):
    """计算数值稳定的交叉熵损失 (Cross-Entropy Loss)。"""
    B = logits.shape[0]

    # Step 1: 使用 log-sum-exp 得到每个样本的对数归一化常数
    log_normalizer = torch.logsumexp(logits, dim=-1)

    # Step 2: 收集真实类别分数并计算平均负对数似然
    target_logits = logits[torch.arange(B, device=logits.device), targets]
    loss = (log_normalizer - target_logits).mean()
    return loss''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
loss = cross_entropy_loss(logits, targets)
print("Loss:", loss.item())
print("Reference loss:", F.cross_entropy(logits, targets).item())""",
    'explanation': r"""### 核心思路
交叉熵等价于真实类别的负对数概率，即 $\frac{1}{B}\sum_i[\log\sum_c e^{z_{ic}}-z_{i,y_i}]$，直接在 logits 上计算可避免显式 softmax。

### 代码步骤拆解
- **Step 1（稳定归一化）**：沿类别维对 `(B, C)` 的 `logits` 使用 `torch.logsumexp`，得到 `(B,)` 的 `log_normalizer`。
- **Step 2（目标收集）**：使用批次索引与 `targets` 收集每行真实类别的 `target_logits`。
- **Step 3（批次归约）**：计算逐样本负对数似然并取均值，返回零维标量 `loss`。

### 重点与易错点分析
- **注意点 1（数值稳定性）**：`logsumexp` 内部执行最大值平移，避免大 logits 的指数溢出。
- **注意点 2（设备对齐）**：批次索引需创建在 `logits.device`，否则 GPU 输入会触发设备不匹配。
- **注意点 3（输入语义）**：`targets` 是类别索引而非 one-hot，且必须落在 $[0,C)$。

### 复杂度分析
- **时间复杂度**：$O(BC)$，主要开销是遍历所有类别计算 `logsumexp`。
- **空间复杂度**：$O(B)$，仅额外保存逐样本归一化常数和目标分数。""",
}

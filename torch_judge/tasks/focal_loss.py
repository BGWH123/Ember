"""Focal Loss task."""

TASK = {
    'title': 'Focal Loss',
    'title_zh': 'Focal Loss',
    'difficulty': 'Medium',
    'category': '损失函数',
    'function_name': 'focal_loss',
    'starter_code': """def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    pass""",
    'hint': """1. Stable softmax: `exp(logits - max)` → normalize → `probs`
2. `p_t = probs[arange(N), targets]`
3. `return (-alpha * (1-p_t)**gamma * log(p_t+1e-8)).mean()`""",
    'hint_zh': """1. 稳定 softmax：`exp(logits - max)` → 归一化 → `probs`
2. `p_t = probs[arange(N), targets]`
3. `return (-alpha * (1-p_t)**gamma * log(p_t+1e-8)).mean()`""",
    'description_en': """Implement Focal Loss for handling class imbalance in classification.

Focal Loss down-weights easy examples so the model focuses on hard ones. It was introduced in RetinaNet for dense object detection.

**Signature:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`

**Parameters:**
- `logits` — raw class scores, shape (N, C)
- `targets` — integer class indices, shape (N,)
- `alpha` — weighting factor (scalar)
- `gamma` — focusing parameter; higher values down-weight easy examples more

**Returns:** scalar mean loss

**Formula:**
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```
where `p_t = softmax(logits)[i, targets[i]]`

**Constraints:**
- Compute softmax manually (no `F.*`)
- Use numerically stable softmax (subtract max before exp)
- Return the mean over the batch""",
    'description_zh': """实现用于处理分类任务中类别不平衡问题的 Focal Loss。

Focal Loss 降低简单样本的权重，使模型专注于困难样本。它由 RetinaNet 在密集目标检测中提出。

**签名:** `focal_loss(logits, targets, alpha=0.25, gamma=2.0) -> Tensor`

**参数:**
- `logits` — 原始类别分数，形状 (N, C)
- `targets` — 整数类别索引，形状 (N,)
- `alpha` — 加权因子（标量）
- `gamma` — 聚焦参数；值越大，对简单样本的降权越强

**返回:** 标量均值损失

**公式:**
```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
```
其中 `p_t = softmax(logits)[i, targets[i]]`

**约束:**
- 手动计算 softmax（不得使用 `F.*`）
- 使用数值稳定的 softmax（exp 前减去最大值）
- 返回批次上的均值""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
torch.manual_seed(0)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
loss = {fn}(logits, targets)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'
assert loss.item() > 0, 'Loss should be positive'""",
        },
        {
            "name": 'gamma=0 reduces to weighted cross-entropy',
            "code": """import torch
torch.manual_seed(1)
logits = torch.randn(16, 4)
targets = torch.randint(0, 4, (16,))
alpha = 0.25
loss_focal = {fn}(logits, targets, alpha=alpha, gamma=0.0)
# reference: manual cross-entropy weighted by alpha
N = logits.shape[0]
shifted = logits - logits.max(dim=-1, keepdim=True).values
probs = torch.exp(shifted) / torch.exp(shifted).sum(dim=-1, keepdim=True)
p_t = probs[torch.arange(N), targets]
ce = (-alpha * torch.log(p_t + 1e-8)).mean()
assert torch.allclose(loss_focal, ce, atol=1e-5), f'gamma=0 should equal weighted CE. Got {loss_focal.item():.6f} vs {ce.item():.6f}'""",
        },
        {
            "name": 'Higher gamma down-weights easy examples',
            "code": """import torch
# easy example: p_t=0.9, hard example: p_t=0.3
# higher gamma should reduce the easy example's contribution more
torch.manual_seed(5)
logits = torch.randn(32, 4)
targets = torch.randint(0, 4, (32,))
loss_g0 = {fn}(logits, targets, alpha=1.0, gamma=0.0)
loss_g2 = {fn}(logits, targets, alpha=1.0, gamma=2.0)
loss_g5 = {fn}(logits, targets, alpha=1.0, gamma=5.0)
# higher gamma always reduces loss (down-weights easy examples)
assert loss_g5 <= loss_g2, f'gamma=5 ({loss_g5:.4f}) should be <= gamma=2 ({loss_g2:.4f})'
assert loss_g2 <= loss_g0, f'gamma=2 ({loss_g2:.4f}) should be <= gamma=0 ({loss_g0:.4f})'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
logits = torch.randn(4, 3, requires_grad=True)
targets = torch.randint(0, 3, (4,))
loss = {fn}(logits, targets)
loss.backward()
assert logits.grad is not None, 'Gradient did not flow back to logits'
assert logits.grad.shape == logits.shape""",
        },
    ],
    'solution': '''def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """计算用于类别不平衡的焦点损失 (Focal Loss)。"""
    B, C = logits.shape

    # Step 1: 最大值平移后手动计算稳定 softmax
    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(shifted_logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

    # Step 2: 收集每个样本真实类别的预测概率
    batch_indices = torch.arange(B, device=logits.device)
    target_probs = probs[batch_indices, targets]

    # Step 3: 应用焦点调制因子并取批次均值
    log_target_probs = torch.log(target_probs + 1e-8)
    loss = -alpha * (1.0 - target_probs) ** gamma * log_target_probs
    return loss.mean()''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(8, 4)
targets = torch.randint(0, 4, (8,))
loss_gamma_0 = focal_loss(logits, targets, gamma=0.0)
loss_gamma_2 = focal_loss(logits, targets, gamma=2.0)
print("Gamma=0 loss:", loss_gamma_0.item())
print("Gamma=2 loss:", loss_gamma_2.item())""",
    'explanation': r"""### 核心思路
Focal Loss 在交叉熵前乘 $(1-p_t)^\gamma$，即 $-\alpha(1-p_t)^\gamma\log p_t$，从而压低高置信度简单样本并聚焦困难样本。

### 代码步骤拆解
- **Step 1（稳定 softmax）**：每行 `logits` 减最大值后指数归一化，得到 `(B, C)` 概率。
- **Step 2（真实类概率）**：按 `targets` 收集 `(B,)` 的 `target_probs`。
- **Step 3（焦点调制）**：计算对数概率与调制因子，逐样本损失取均值得到标量。

### 重点与易错点分析
- **注意点 1（数值稳定性）**：指数前最大值平移避免溢出，`log` 前加 $10^{-8}$ 避免极小概率产生负无穷。
- **注意点 2（设备对齐）**：`torch.arange` 必须位于 `logits.device`，否则 GPU 索引会报错。
- **注意点 3（退化关系）**：当 `gamma=0` 时调制因子恒为 1，损失退化为 `alpha` 加权交叉熵。

### 复杂度分析
- **时间复杂度**：$O(BC)$，softmax 需要遍历全部类别。
- **空间复杂度**：$O(BC)$，平移 logits、指数和概率张量占主要空间。""",
}

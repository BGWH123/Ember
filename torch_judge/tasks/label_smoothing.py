"""Label Smoothing Loss task."""

TASK = {
    'title': 'Label Smoothing Loss',
    'title_zh': '标签平滑损失',
    'difficulty': 'Easy',
    'category': '正则化',
    'function_name': 'label_smoothing',
    'starter_code': """def label_smoothing(logits, targets, smoothing=0.1):
    pass""",
    'hint': """1. `soft = full((N,C), ε/(C-1))`, then `soft.scatter_(1, targets, 1-ε)`
2. `log_probs` via stable log-softmax
3. `return -(soft * log_probs).sum(dim=-1).mean()`""",
    'hint_zh': """1. `soft = full((N,C), ε/(C-1))`，再 `soft.scatter_(1, targets, 1-ε)`
2. 用稳定 log-softmax 得到 `log_probs`
3. `return -(soft * log_probs).sum(dim=-1).mean()`""",
    'description_en': """Implement label smoothing cross-entropy loss.

Label smoothing prevents overconfidence by mixing the one-hot target distribution with a uniform distribution. It is widely used in Transformer training.

**Signature:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`

**Parameters:**
- `logits` — raw model outputs, shape `(N, C)`
- `targets` — integer class indices, shape `(N,)`
- `smoothing` — smoothing factor ε ∈ [0, 1)

**Returns:** scalar loss

**Formula:** soft target for correct class = `1 - ε`, for others = `ε / (C - 1)`""",
    'description_zh': """实现标签平滑交叉熵损失。

标签平滑通过将 one-hot 目标分布与均匀分布混合来防止过度自信，在 Transformer 训练中被广泛使用。

**签名:** `label_smoothing(logits, targets, smoothing=0.1) -> Tensor`

**参数:**
- `logits` — 模型原始输出，形状 `(N, C)`
- `targets` — 整数类别索引，形状 `(N,)`
- `smoothing` — 平滑系数 ε ∈ [0, 1)

**返回:** 标量损失

**公式:** 正确类别的软目标 = `1 - ε`，其他类别 = `ε / (C - 1)`""",
    "tests": [
        {
            "name": 'Returns scalar',
            "code": """import torch
logits = torch.randn(4, 10)
targets = torch.tensor([0, 3, 7, 9])
loss = {fn}(logits, targets)
assert loss.shape == (), f'Expected scalar, got shape {loss.shape}'""",
        },
        {
            "name": 'smoothing=0 matches standard cross-entropy',
            "code": """import torch
torch.manual_seed(42)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))
loss_smooth = {fn}(logits, targets, smoothing=0.0)
loss_ce = torch.nn.functional.cross_entropy(logits, targets)
assert torch.allclose(loss_smooth, loss_ce, atol=1e-5), f'With smoothing=0, should match CE: {loss_smooth} vs {loss_ce}'""",
        },
        {
            "name": 'smoothing reduces confidence',
            "code": """import torch
torch.manual_seed(0)
logits = torch.randn(16, 8)
targets = torch.randint(0, 8, (16,))
loss_0 = {fn}(logits, targets, smoothing=0.0)
loss_s = {fn}(logits, targets, smoothing=0.1)
# Label smoothing increases loss when model is confident
assert loss_s.item() != loss_0.item(), 'Smoothed loss should differ from standard CE'
assert loss_s.item() > 0, 'Loss must be positive'""",
        },
        {
            "name": 'Gradient flows',
            "code": """import torch
logits = torch.randn(4, 6, requires_grad=True)
targets = torch.tensor([0, 1, 2, 3])
loss = {fn}(logits, targets, smoothing=0.1)
loss.backward()
assert logits.grad is not None, 'Gradient not computed'""",
        },
        {
            "name": 'Exact smoothed loss value',
            "code": """import torch
torch.manual_seed(0)
B, C = 4, 5
logits = torch.randn(B, C)
targets = torch.randint(0, C, (B,))
smoothing = 0.1
# Reference using solution formula: non-target = smoothing/(C-1), target = 1-smoothing
logits_max = logits.max(dim=-1, keepdim=True).values
shifted = logits - logits_max
log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
smooth_labels = torch.full((B, C), smoothing / (C - 1))
smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
expected = -(smooth_labels * log_probs).sum(dim=-1).mean()
out = {fn}(logits, targets, smoothing)
assert torch.allclose(out, expected, atol=1e-5), f'Expected {expected.item():.6f}, got {out.item():.6f}'""",
        },
    ],
    'solution': '''def label_smoothing(logits, targets, smoothing=0.1):
    """计算标签平滑交叉熵 (Label Smoothing Cross-Entropy)。"""
    _, C = logits.shape

    # Step 1: 用最大值平移计算稳定的对数概率
    row_max = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - row_max
    log_probs = shifted_logits - torch.log(
        torch.exp(shifted_logits).sum(dim=-1, keepdim=True)
    )

    # Step 2: 构造目标类与非目标类的平滑分布
    soft_targets = torch.full_like(log_probs, smoothing / (C - 1))
    soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    # Step 3: 计算逐样本交叉熵并取批次均值
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    return loss''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
smoothed_loss = label_smoothing(logits, targets, smoothing=0.1)
plain_loss = label_smoothing(logits, targets, smoothing=0.0)
print("Logits shape:", logits.shape)
print("Smoothed loss:", smoothed_loss.item())
print("Plain loss:", plain_loss.item())""",
    'explanation': r"""### 核心思路
标签平滑把 one-hot 目标改为正确类概率 $1-\varepsilon$、其余类概率 $\varepsilon/(C-1)$，再与稳定的 `log_probs` 计算交叉熵。数据流由 `(N, C)` 的 `logits` 和 `(N,)` 的 `targets` 汇聚为标量损失。

### 代码步骤拆解
- **Step 1（稳定归一化）**：每行减去 `row_max` 后计算 `log_probs`，保持形状 `(N, C)`。
- **Step 2（软标签）**：用 `torch.full_like` 填充非目标概率，再通过 `scatter_` 写入目标类概率。
- **Step 3（交叉熵）**：沿类别维求和得到 `(N,)` 损失，最后对批次取均值。

### 重点与易错点分析
- **注意点 1（类别数边界）**：公式含 $C-1$，因此至少需要两个类别；单类别输入会发生除零。
- **注意点 2（最大值平移）**：`keepdim=True` 保留 `(N, 1)`，使减法和后续对数归一化能正确广播且避免指数上溢。
- **注意点 3（定义差异）**：本题把平滑质量只分给非目标类；这与把 $\varepsilon$ 均匀分给全部 $C$ 类的另一常见定义不同。

### 复杂度分析
- **时间复杂度**：$O(NC)$，主要计算是 log-softmax 与软标签交叉熵。
- **空间复杂度**：$O(NC)$，需要保存 `log_probs` 和 `soft_targets`。""",
}

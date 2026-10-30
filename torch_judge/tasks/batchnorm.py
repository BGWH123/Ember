"""Implement BatchNorm task."""

TASK = {
    'title': 'Implement BatchNorm',
    'title_zh': '实现 BatchNorm',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'my_batch_norm',
    'starter_code': """def my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    pass""",
    'hint': """Train: `mean/var = x.mean/var(dim=0)`, update running stats with `momentum`
Eval: use `running_mean/running_var` directly
Both: `gamma * (x - mean) / sqrt(var + eps) + beta`""",
    'hint_zh': """训练：`mean/var = x.mean/var(dim=0)`，用 `momentum` 更新运行统计量
推理：直接使用 `running_mean/running_var`
两者：`gamma * (x - mean) / sqrt(var + eps) + beta`""",
    'description_en': """Implement Batch Normalization with train/eval modes.

BatchNorm normalizes activations per feature using batch statistics during training and running statistics during inference, stabilizing deep network training.

**Signature:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`

**Parameters:**
- `x` — input tensor (N, D)
- `gamma`, `beta` — learnable affine parameters (D,)
- `running_mean`, `running_var` — running statistics (D,), updated in-place during training

**Returns:** normalized and affine-transformed tensor, same shape as x

**Constraints:**
- Training: use batch stats, update running stats with momentum
- Inference: use running stats only
- Use `unbiased=False` for batch variance""",
    'description_zh': """实现带训练/推理模式的批归一化。

BatchNorm 在训练时使用批统计量、推理时使用运行统计量对每个特征进行归一化，从而稳定深度网络训练。

**签名:** `my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True) -> Tensor`

**参数:**
- `x` — 输入张量 (N, D)
- `gamma`, `beta` — 可学习的仿射参数 (D,)
- `running_mean`, `running_var` — 运行统计量 (D,)，训练时原地更新

**返回:** 归一化并仿射变换后的张量，形状与 x 相同

**约束:**
- 训练模式：使用批统计量，用 momentum 更新运行统计量
- 推理模式：仅使用运行统计量
- 批方差使用 `unbiased=False`""",
    "tests": [
        {
            "name": 'Training mode — zero mean per feature',
            "code": """import torch
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
col_means = out.mean(dim=0)
assert torch.allclose(col_means, torch.zeros(4), atol=1e-5), f'Column means not zero: {col_means}'""",
        },
        {
            "name": 'Training mode — numerical correctness and running stats update',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(16, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
momentum = 0.1
out = {fn}(x, gamma, beta, running_mean, running_var, momentum=momentum, training=True)

# Reference using batch stats
mean = x.mean(dim=0)
var = x.var(dim=0, unbiased=False)
ref = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch'

# Running stats should have moved toward batch stats
expected_mean = (1 - momentum) * torch.zeros_like(mean) + momentum * mean
expected_var = (1 - momentum) * torch.ones_like(var) + momentum * var
assert torch.allclose(running_mean, expected_mean, atol=1e-6), 'running_mean not updated correctly'
assert torch.allclose(running_var, expected_var, atol=1e-6), 'running_var not updated correctly'""",
        },
        {
            "name": 'Inference mode — uses running statistics',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(4, 8)
gamma = torch.randn(8)
beta = torch.randn(8)

# Pretend these came from previous training
running_mean = torch.randn(8)
running_var = torch.rand(8) + 0.5  # positive

out = {fn}(x, gamma, beta, running_mean.clone(), running_var.clone(), training=False)
ref = gamma * (x - running_mean) / torch.sqrt(running_var + 1e-5) + beta
assert torch.allclose(out, ref, atol=1e-4), 'Inference should use running stats'""",
        },
        {
            "name": 'Gradient flow w.r.t inputs and affine params',
            "code": """import torch
x = torch.randn(4, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
running_mean = torch.zeros(8)
running_var = torch.ones(8)
out = {fn}(x, gamma, beta, running_mean, running_var, training=True)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'
assert beta.grad is not None, 'beta.grad is None'""",
        },
    ],
    'solution': '''def my_batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    """批归一化 (Batch Normalization)。"""
    # Step 1: 训练时计算批统计量并更新运行状态
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        running_mean.mul_(1.0 - momentum).add_(batch_mean.detach(), alpha=momentum)
        running_var.mul_(1.0 - momentum).add_(batch_var.detach(), alpha=momentum)
        mean, var = batch_mean, batch_var
    else:
        mean, var = running_mean, running_var

    # Step 2: 使用选定统计量标准化输入
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # Step 3: 应用可学习仿射变换
    return gamma * x_norm + beta''',
    'demo': """torch.manual_seed(42)
x = torch.randn(8, 4)
gamma, beta = torch.ones(4), torch.zeros(4)
running_mean, running_var = torch.zeros(4), torch.ones(4)
out = my_batch_norm(x, gamma, beta, running_mean, running_var)
print("Output shape:", out.shape)
print("Column means:", out.mean(dim=0))
print("Running mean:", running_mean)""",
    'explanation': r"""### 核心思路
训练阶段用批均值和总体方差归一化，并维护指数移动平均；推理阶段改用固定运行统计量，最后执行 $\gamma\hat{x}+\beta$。

### 代码步骤拆解
- **Step 1（统计量选择）**：训练时计算 `batch_mean/batch_var` 并原地更新运行状态，推理时直接读取状态。
- **Step 2（标准化）**：`mean` 和 `var` 沿批次维广播到 `x`。
- **Step 3（仿射变换）**：`gamma` 与 `beta` 在特征维逐元素缩放和偏移。

### 重点与易错点分析
- **注意点 1（方差约定）**：本题运行方差按 `unbiased=False` 更新，必须与测试公式一致。
- **注意点 2（停止梯度）**：写入运行状态前使用 `.detach()`，避免状态更新连入计算图。
- **注意点 3（eps 位置）**：`eps` 加在方差上再开方，防止零方差除零。

### 复杂度分析
- **时间复杂度**：$O(BD)$，统计和归一化均线性扫描输入。
- **空间复杂度**：$O(D)$，除输出外只需保存特征统计量。""",
}

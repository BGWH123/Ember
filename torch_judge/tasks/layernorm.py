"""Implement LayerNorm task."""

TASK = {
    'title': 'Implement LayerNorm',
    'title_zh': '实现 LayerNorm',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'my_layer_norm',
    'starter_code': """def my_layer_norm(x, gamma, beta, eps=1e-5):
    pass""",
    'hint': """1. `mean = x.mean(dim=-1, keepdim=True)`
2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`
3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`""",
    'hint_zh': """1. `mean = x.mean(dim=-1, keepdim=True)`
2. `var = x.var(dim=-1, keepdim=True, unbiased=False)`
3. `x_norm = (x - mean) / sqrt(var + eps)` → `gamma * x_norm + beta`""",
    'description_en': """Implement Layer Normalization.

LayerNorm normalizes each sample across the feature dimension, stabilizing training without dependence on batch size.

**Signature:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`

**Parameters:**
- `x` — input tensor (..., D)
- `gamma` — scale parameter (D,)
- `beta` — shift parameter (D,)
- `eps` — epsilon for numerical stability

**Returns:** normalized tensor, same shape as x

**Constraints:**
- Normalize over the last dimension
- Use `unbiased=False` for variance
- Must match `F.layer_norm`""",
    'description_zh': """实现层归一化。

LayerNorm 对每个样本沿特征维度进行归一化，不依赖批大小即可稳定训练。

**签名:** `my_layer_norm(x, gamma, beta, eps=1e-5) -> Tensor`

**参数:**
- `x` — 输入张量 (..., D)
- `gamma` — 缩放参数 (D,)
- `beta` — 偏移参数 (D,)
- `eps` — 数值稳定性的 epsilon

**返回:** 归一化后的张量，形状与 x 相同

**约束:**
- 沿最后一个维度归一化
- 方差使用 `unbiased=False`
- 必须与 `F.layer_norm` 一致""",
    'theory_en': r"""Layer Normalization normalizes each sample independently across the feature dimension, unlike BatchNorm which normalizes across the batch.

**Formula:**
$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
where $\mu = \frac{1}{D} \sum_{i=1}^D x_i$ and $\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$

**Why LayerNorm over BatchNorm in Transformers?**
- Sequence lengths vary; batch statistics become unreliable
- Independent per-sample normalization avoids batch-size dependencies
- Pre-norm (LN before attention/FFN) is more stable than post-norm for deep networks""",
    'theory_zh': r"""层归一化对每个样本沿特征维度独立归一化，与 BatchNorm 沿批次维度归一化不同。

**公式：**
$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
其中 $\mu = \frac{1}{D} \sum_{i=1}^D x_i$，$\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$

**为什么 Transformer 中使用 LayerNorm 而非 BatchNorm？**
- 序列长度变化；批次统计量不可靠
- 逐样本独立归一化避免依赖批大小
- Pre-norm（在注意力/FFN 之前做 LN）比 post-norm 对深层网络更稳定""",
    'diagram_en': """```mermaid
flowchart TD
    X[x] --> MEAN[Compute mean<br/>mu = mean(x, dim=-1)]
    X --> VAR[Compute variance<br/>sigma^2 = var(x, dim=-1)]
    MEAN --> NORM[Normalize<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]
    VAR --> NORM
    GAMMA[gamma] --> SCALE[Scale & Shift<br/>gamma * x_norm + beta]
    BETA[beta] --> SCALE
    NORM --> SCALE
```""",
    'diagram_zh': """```mermaid
flowchart TD
    X[x] --> MEAN[计算均值<br/>mu = mean(x, dim=-1)]
    X --> VAR[计算方差<br/>sigma^2 = var(x, dim=-1)]
    MEAN --> NORM[归一化<br/>x_norm = (x - mu) / sqrt(sigma^2 + eps)]
    VAR --> NORM
    GAMMA[gamma] --> SCALE[缩放与偏移<br/>gamma * x_norm + beta]
    BETA[beta] --> SCALE
    NORM --> SCALE
```""",
    "tests": [
        {
            "name": 'Shape and basic behavior',
            "code": """import torch
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = {fn}(x, gamma, beta)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch vs F.layer_norm'""",
        },
        {
            "name": 'With learned parameters',
            "code": """import torch
x = torch.randn(4, 16)
gamma = torch.randn(16)
beta = torch.randn(16)
out = {fn}(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [16], gamma, beta)
assert torch.allclose(out, ref, atol=1e-4), 'Value mismatch with non-trivial gamma/beta'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
x = torch.randn(2, 8, requires_grad=True)
gamma = torch.ones(8, requires_grad=True)
beta = torch.zeros(8, requires_grad=True)
out = {fn}(x, gamma, beta)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gamma.grad is not None, 'gamma.grad is None'""",
        },
    ],
    'solution': '''def my_layer_norm(x, gamma, beta, eps=1e-5):
    """实现逐样本层归一化 (Layer Normalization)。"""
    # Step 1: 沿最后一维计算均值与有偏方差
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)

    # Step 2: 使用方差内的 eps 完成稳定标准化
    normalized = (x - mean) / torch.sqrt(variance + eps)

    # Step 3: 应用可学习的逐特征仿射变换
    out = gamma * normalized + beta
    return out''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 3, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
print("Output shape:", out.shape)
print("Mean sample:", out[0, 0].mean().item())
print("Matches PyTorch:", torch.allclose(out, F.layer_norm(x, [8], gamma, beta), atol=1e-4))""",
    'explanation': r"""### 核心思路
LayerNorm 对每个样本的最后一维独立计算 $\mu$ 与 $\sigma^2$，再执行 `(x - mean) / sqrt(variance + eps)` 并用 `gamma`、`beta` 仿射变换。输入输出形状保持不变。

### 代码步骤拆解
- **Step 1（统计量）**：`mean` 和 `variance` 均以 `keepdim=True` 得到末维为 1 的张量。
- **Step 2（标准化）**：从 `x` 减均值并除以稳定标准差。
- **Step 3（仿射恢复）**：广播 `(D,)` 的 `gamma`、`beta` 到所有前导维。

### 重点与易错点分析
- **注意点 1（方差定义）**：必须设置 `unbiased=False`，使用总体方差才能与 `F.layer_norm` 一致。
- **注意点 2（eps 位置）**：`eps` 应加在方差内部再开方；加在开方后会改变数值与梯度。
- **注意点 3（广播维度）**：保留归约维可避免 `(B, S)` 与 `(B, S, D)` 错位广播。

### 复杂度分析
- **时间复杂度**：$O(ND)$，每个特征参与统计和归一化。
- **空间复杂度**：$O(ND)$，反向传播保存归一化结果和统计量。""",
}

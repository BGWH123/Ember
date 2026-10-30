"""Implement RMSNorm task."""

TASK = {
    'title': 'Implement RMSNorm',
    'title_zh': '实现 RMSNorm',
    'difficulty': 'Medium',
    'category': '归一化',
    'function_name': 'rms_norm',
    'starter_code': """def rms_norm(x, weight, eps=1e-06):
    pass""",
    'hint': """`rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`
`return x / rms * weight`  ← no mean subtraction""",
    'hint_zh': """`rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`
`return x / rms * weight`  ← 无需减均值""",
    'description_en': """Implement RMSNorm (Root Mean Square Layer Normalization).

RMSNorm is a simpler alternative to LayerNorm that skips mean subtraction, normalizing only by the root mean square of activations.

**Signature:** `rms_norm(x, weight, eps=1e-6) -> Tensor`

**Parameters:**
- `x` — input tensor (..., D)
- `weight` — learnable scale parameter (D,)
- `eps` — epsilon for numerical stability

**Returns:** normalized tensor, same shape as x

**Constraints:**
- `RMS(x) = sqrt(mean(x^2) + eps)` over last dim
- Output: `x / RMS(x) * weight`""",
    'description_zh': """实现 RMSNorm（均方根层归一化）。

RMSNorm 是 LayerNorm 的简化替代，跳过均值减法，仅通过激活值的均方根进行归一化。

**签名:** `rms_norm(x, weight, eps=1e-6) -> Tensor`

**参数:**
- `x` — 输入张量 (..., D)
- `weight` — 可学习的缩放参数 (D,)
- `eps` — 数值稳定性的 epsilon

**返回:** 归一化后的张量，形状与 x 相同

**约束:**
- `RMS(x) = sqrt(mean(x^2) + eps)` 沿最后一维
- 输出：`x / RMS(x) * weight`""",
    'theory_en': r"""RMSNorm simplifies LayerNorm by removing mean centering, using only root-mean-square scaling.

**Formula:**
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}$$
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \odot \gamma$$

**Comparison with LayerNorm:**
- ~30% fewer FLOPs (no mean computation)
- Empirically works as well as LayerNorm for most tasks
- Used in LLaMA, Mistral, and other modern LLMs

The key insight: mean subtraction may be unnecessary if the next layer (e.g., attention) is shift-invariant.""",
    'theory_zh': r"""RMSNorm 简化了 LayerNorm，去除了均值中心化，仅使用均方根缩放。

**公式：**
$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}$$
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \odot \gamma$$

**与 LayerNorm 对比：**
- 计算量减少约 30%（无需计算均值）
-  empirically 在大多数任务上与 LayerNorm 表现相当
- 用于 LLaMA、Mistral 等现代 LLM

核心洞见：如果下一层（如注意力）对平移不变，则均值减法可能是不必要的。""",
    'diagram_en': """```mermaid
flowchart TD
    X[x] --> RMS[Compute RMS<br/>rms = sqrt(mean(x^2))]
    RMS --> NORM[Normalize<br/>x / rms]
    W[weight] --> SCALE[Scale<br/>x / rms * weight]
    NORM --> SCALE
    style RMS fill:#e1f5e1
    note right of RMS
        No mean subtraction!<br/>vs LayerNorm
    end
```""",
    'diagram_zh': """```mermaid
flowchart TD
    X[x] --> RMS[计算 RMS<br/>rms = sqrt(mean(x^2))]
    RMS --> NORM[归一化<br/>x / rms]
    W[weight] --> SCALE[缩放<br/>x / rms * weight]
    NORM --> SCALE
    style RMS fill:#e1f5e1
    note right of RMS
        不减均值！<br/>与 LayerNorm 的区别
    end
```""",
    "tests": [
        {
            "name": 'Basic behavior',
            "code": """import torch
x = torch.randn(2, 8)
weight = torch.ones(8)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch'""",
        },
        {
            "name": 'With learned weight',
            "code": """import torch
x = torch.randn(4, 16)
weight = torch.randn(16)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch with non-trivial weight'""",
        },
        {
            "name": '3-D input',
            "code": """import torch
x = torch.randn(2, 4, 32)
weight = torch.ones(32)
out = {fn}(x, weight)
assert out.shape == x.shape, f'Shape mismatch on 3-D: {out.shape}'
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Value mismatch on 3-D'""",
        },
        {
            "name": 'Differs from LayerNorm (no mean subtraction)',
            "code": """import torch
torch.manual_seed(0)
x = torch.ones(2, 8) * 5.0 + torch.randn(2, 8) * 0.1
weight = torch.ones(8)
out = {fn}(x, weight)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
ref = x / rms * weight
assert torch.allclose(out, ref, atol=1e-5), 'Should match RMS formula (no mean subtraction)'
# LayerNorm would subtract mean first, giving very different results
ln_out = torch.nn.functional.layer_norm(x, [8], weight, torch.zeros(8))
assert not torch.allclose(out, ln_out, atol=0.1), 'Output should differ from LayerNorm'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
x = torch.randn(2, 8, requires_grad=True)
weight = torch.ones(8, requires_grad=True)
out = {fn}(x, weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert weight.grad is not None, 'weight.grad is None'""",
        },
    ],
    'solution': '''def rms_norm(x, weight, eps=1e-06):
    """实现 RMSNorm (Implement RMSNorm)。"""
    # Step 1: 沿最后一维计算 RMS(x)=sqrt(mean(x²)+eps)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    # Step 2: 用均方根缩放输入，但不执行 LayerNorm 的均值中心化
    normalized = x / rms
    # Step 3: 逐特征乘可学习参数 weight，恢复模型所需的通道尺度
    return normalized * weight''',
    'demo': """torch.manual_seed(42)
x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())
print('Input means:', x.mean(dim=-1))
print('Output means are not forced to zero:', out.mean(dim=-1))""",
    'explanation': r"""### 核心思路
RMSNorm 对每个最后维向量计算 $\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{i=1}^D x_i^2+\epsilon}$，输出 $y_i=x_i/\operatorname{RMS}(x)\cdot w_i$。它只消除整体幅值，不减去均值，因此保留输入方向与偏移信息，比 LayerNorm 少一次均值中心化。

### 代码步骤拆解
- **Step 1（计算均方根）**：`x.pow(2).mean(dim=-1,keepdim=True)` 将任意前导形状 `(...,D)` 归约为 `(...,1)`，加 `eps` 后开平方得到 `rms`。
- **Step 2（统一向量尺度）**：`normalized=x/rms` 依靠末维单例轴广播，使每个 token 或样本独立归一化；没有计算或减去 `x.mean(-1)`。
- **Step 3（学习逐通道增益）**：形状 `(D,)` 的 `weight` 广播到所有前导维，逐特征恢复或重分配归一化后的幅值。

### 重点与易错点分析
- **注意点 1（epsilon 的位置）**：本题定义为 $\sqrt{\operatorname{mean}(x^2)+\epsilon}$；把 `eps` 加在平方根外会产生不同结果，尤其是接近零的向量。
- **注意点 2（与 LayerNorm 区分）**：RMSNorm 输出均值通常不为零，也没有 bias 参数；若先减均值就变成了 LayerNorm 的核心计算，违背算法定义。
- **注意点 3（混合精度实现）**：平方可能在 fp16 大值处溢出；生产模型常临时用 fp32 累积平方均值再转回输入 dtype。本题直接遵循输入 dtype 与给定公式。

### 复杂度分析
- **时间复杂度**：设有 $N$ 个向量、隐藏维为 $D$，平方归约、缩放和乘权重均为线性操作，总计 $O(ND)$。
- **空间复杂度**：输出与 `normalized` 为 $O(ND)$，每个向量的 `rms` 为 $O(N)$；融合实现可减少中间张量但渐近复杂度不变。""",
}

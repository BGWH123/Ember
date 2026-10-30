"""Softmax with Temperature task."""

TASK = {
    'title': 'Softmax with Temperature',
    'title_zh': '带温度的 Softmax',
    'difficulty': 'Easy',
    'category': '采样与解码',
    'function_name': 'softmax_temperature',
    'starter_code': """def softmax_temperature(logits, temperature):
    pass""",
    'hint': """Divide logits by T before computing softmax. Use the standard max-subtraction trick for numerical stability.""",
    'hint_zh': '计算 softmax 前先将 logits 除以 T。使用标准的 max 减法技巧保证数值稳定。',
    'description_en': """Implement softmax with temperature scaling: p_i = exp(z_i / T) / Σ_j exp(z_j / T), where T > 0 is the temperature. Higher T makes distribution more uniform (less confident), lower T makes it sharper (more confident).""",
    'description_zh': """实现带温度缩放的 Softmax：p_i = exp(z_i / T) / Σ_j exp(z_j / T)，其中 T > 0 是温度。T 越大分布越均匀（置信度低），T 越小分布越尖锐（置信度高）。""",
    'theory_en': """Temperature scaling controls the sharpness of a probability distribution. It is used in text generation (sampling diversity) and knowledge distillation (softening teacher labels).""",
    'theory_zh': '温度缩放控制概率分布的尖锐程度。用于文本生成（控制采样多样性）和知识蒸馏（软化教师标签）。',
    "tests": [
        {
            "name": 'Temperature and final dimension',
            "code": """x = torch.tensor([[1., 3., -2.], [2., 1., 0.]], dtype=torch.float64)
for temperature in [0.25, 1., 4.]:
    out = {fn}(x, temperature)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.softmax(x / temperature, dim=-1), atol=1e-10)""",
        },
        {
            "name": 'Large logits and shift invariance',
            "code": """x = torch.tensor([[10000., 10001., 9999.], [-10000., -10000., -10000.]])
out = {fn}(x, 0.5)
assert torch.isfinite(out).all()
assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-6)
assert torch.allclose(out, {fn}(x - 10000., 0.5), atol=1e-6)""",
        },
        {
            "name": 'Backward with nonconstant upstream gradient',
            "code": """x = torch.tensor([0.2, -0.5, 1.], dtype=torch.float64, requires_grad=True)
reference = x.detach().clone().requires_grad_()
weights = torch.tensor([1., -2., 3.], dtype=torch.float64)
({fn}(x, 0.7) * weights).sum().backward()
(torch.softmax(reference / 0.7, dim=-1) * weights).sum().backward()
assert torch.allclose(x.grad, reference.grad, atol=1e-10)""",
        },
    ],
    'solution': '''def softmax_temperature(logits, temperature):
    """实现带温度的 Softmax（Softmax with Temperature）。"""
    # Step 1: 用正温度重缩放 logits 以调节类别间距
    scaled = logits / temperature
    # Step 2: 减去每行最大缩放 logit 后执行稳定指数化
    max_val = scaled.max(dim=-1, keepdim=True).values
    exp_scaled = torch.exp(scaled - max_val)
    # Step 3: 沿类别维归一化得到温度控制的概率分布
    probs = exp_scaled / exp_scaled.sum(dim=-1, keepdim=True)
    return probs''',
    'demo': """torch.manual_seed(42)
logits = torch.randn(2, 4)
cold_probs = softmax_temperature(logits, temperature=0.3)
hot_probs = softmax_temperature(logits, temperature=3.0)
print("Cold distribution:", cold_probs[0])
print("Hot distribution:", hot_probs[0])
print("Peak probabilities (cold, hot):", cold_probs[0].max().item(), hot_probs[0].max().item())""",
    'explanation': r"""### 核心思路
温度 softmax 定义为 $p_i(T)=\exp(z_i/T)/\sum_j\exp(z_j/T)$。$T<1$ 放大 logit 差距，使概率趋向 argmax；$T>1$ 压缩差距，使分布趋近均匀。实现仍采用减最大值形式，以稳定处理温度缩放后可能非常大的数值。

### 代码步骤拆解
- **Step 1（温度缩放 logits）**：`scaled = logits / temperature` 保持输入形状 $(B,V)$，直接改变任意两类之间的差值 $(z_i-z_j)/T$。
- **Step 2（稳定指数化）**：沿词表维从 `scaled` 减去 `max_val`，使最大指数参数为 0，再计算 `exp_scaled`。
- **Step 3（类别概率归一化）**：沿最后一维除以指数和，每行得到总和为 1 的 `probs`。

### 重点与易错点分析
- **注意点 1（温度定义域）**：$T$ 必须严格大于 0；$T=0$ 会除零，负温度会反转类别偏好，都不符合常规采样温度定义。
- **注意点 2（低温极限与稳定性）**：很小的 $T$ 会放大 logits，必须在除温度后再减该行最大值；否则即使原 logits 温和，`exp(logits / T)` 也可能溢出。
- **注意点 3（梯度与用途）**：对 logits 的 Jacobian 额外带 $1/T$ 因子，低温不仅更尖锐也会放大局部梯度尺度；生成时用它控制多样性，蒸馏时常用高温暴露非最大类别关系。

### 复杂度分析
- **时间复杂度**：$O(BV)$，对每个批次的 $V$ 个类别执行缩放、归约、指数与归一化。
- **空间复杂度**：$O(BV)$，`scaled`、`exp_scaled` 和输出概率是主要张量。""",
}

"""Gradient Norm Clipping task."""

TASK = {
    'title': 'Gradient Norm Clipping',
    'title_zh': '梯度范数裁剪',
    'difficulty': 'Easy',
    'category': '优化器与学习率',
    'function_name': 'clip_grad_norm',
    'starter_code': """def clip_grad_norm(parameters, max_norm):
    pass""",
    'hint': """1. total_norm = sqrt(Σ p.grad.norm()²)
2. clip_coef = max_norm / total_norm
3. if total_norm > max_norm: scale all grads by clip_coef
4. return original total_norm (float)""",
    'hint_zh': """1. total_norm = sqrt(Σ p.grad.norm()²)
2. clip_coef = max_norm / total_norm
3. 若 total_norm > max_norm：所有梯度乘以 clip_coef
4. 返回原始 total_norm（float）""",
    'description_en': """Implement gradient norm clipping.

Gradient clipping rescales all parameter gradients when their combined L2 norm exceeds a threshold, preventing exploding gradients.

**Signature:** `clip_grad_norm(parameters, max_norm) -> float`

**Parameters:**
- `parameters` — list of tensors with `.grad` attributes
- `max_norm` — maximum allowed gradient norm

**Returns:** original total gradient norm (float)

**Constraints:**
- Total norm = `sqrt(sum(p.grad.norm()^2))`
- Only clip if total norm > max_norm
- Preserve gradient direction""",
    'description_zh': """实现梯度范数裁剪。

梯度裁剪在所有参数梯度的 L2 范数超过阈值时进行缩放，防止梯度爆炸。

**签名:** `clip_grad_norm(parameters, max_norm) -> float`

**参数:**
- `parameters` — 带 `.grad` 属性的张量列表
- `max_norm` — 允许的最大梯度范数

**返回:** 原始总梯度范数（浮点数）

**约束:**
- 总范数 = `sqrt(sum(p.grad.norm()^2))`
- 仅在总范数 > max_norm 时裁剪
- 保持梯度方向不变""",
    "tests": [
        {
            "name": 'Clips to max_norm',
            "code": """import torch
p1 = torch.randn(10, requires_grad=True)
p2 = torch.randn(10, requires_grad=True)
(p1 * 10).sum().backward()
(p2 * 10).sum().backward()
{fn}([p1, p2], max_norm=1.0)
new_norm = torch.sqrt(p1.grad.norm()**2 + p2.grad.norm()**2).item()
assert new_norm <= 1.0 + 1e-5, f'Clipped norm {new_norm:.4f} > 1.0'""",
        },
        {
            "name": 'Returns original norm',
            "code": """import torch
p = torch.randn(10, requires_grad=True)
(p * 3).sum().backward()
expected = p.grad.norm().item()
returned = {fn}([p], max_norm=100.0)
assert abs(returned - expected) < 1e-4, f'Returned {returned:.4f}, expected {expected:.4f}'""",
        },
        {
            "name": 'No change when norm < max_norm',
            "code": """import torch
p = torch.randn(4, requires_grad=True)
(p * 0.001).sum().backward()
grad_before = p.grad.clone()
{fn}([p], max_norm=100.0)
assert torch.equal(p.grad, grad_before), 'Should not change when norm < max_norm'""",
        },
        {
            "name": 'Preserves direction',
            "code": """import torch
torch.manual_seed(0)
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
dir_before = p.grad / p.grad.norm()
{fn}([p], max_norm=1.0)
dir_after = p.grad / p.grad.norm()
assert torch.allclose(dir_before, dir_after, atol=1e-5), 'Should preserve direction'""",
        },
    ],
    'solution': '''def clip_grad_norm(parameters, max_norm):
    """按全局 L2 范数裁剪梯度 (Gradient Norm Clipping)。"""
    # Step 1: 过滤无梯度参数并计算原始全局范数
    parameters = [param for param in parameters if param.grad is not None]
    if not parameters:
        return 0.0
    squared_norms = torch.stack([param.grad.norm().square() for param in parameters])
    total_norm = torch.sqrt(squared_norms.sum())

    # Step 2: 仅在超阈值时按统一比例缩放
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for param in parameters:
            param.grad.mul_(scale)

    # Step 3: 返回裁剪前范数
    return total_norm.item()''',
    'demo': """torch.manual_seed(42)
param = torch.randn(100, requires_grad=True)
(param * 10.0).sum().backward()
before = param.grad.norm().item()
original_norm = clip_grad_norm([param], max_norm=1.0)
print("Original norm:", original_norm)
print("Clipped norm:", param.grad.norm().item())
print("Scale applied:", param.grad.norm().item() / before)""",
    'explanation': r"""### 核心思路
先计算所有梯度拼接后的全局范数 $\sqrt{\sum_i\|g_i\|_2^2}$，若超过 `max_norm`，则统一乘以比例以保持方向。

### 代码步骤拆解
- **Step 1（全局范数）**：过滤 `grad is None` 的参数并汇总每个梯度范数平方。
- **Step 2（统一缩放）**：所有梯度原地乘相同 `scale`，裁剪后总范数不超过阈值。
- **Step 3（结果返回）**：返回裁剪前 `total_norm` 的 Python 浮点值。

### 重点与易错点分析
- **注意点 1（方向保持）**：必须对全部梯度使用同一比例，逐张量独立裁剪会改变全局方向。
- **注意点 2（数值稳定）**：分母加 `1e-6` 防止极小范数除法问题，同时只在超阈值时缩放。
- **注意点 3（缺失梯度）**：无梯度参数应跳过；全部为空时返回 0 避免 `stack` 失败。

### 复杂度分析
- **时间复杂度**：$O(P)$，遍历所有有效梯度元素。
- **空间复杂度**：$O(K)$，其中 $K$ 为有梯度的参数张量数。""",
}

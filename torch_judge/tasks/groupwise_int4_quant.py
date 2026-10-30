"""Group-Wise Asymmetric INT4 Quantization task."""

TASK = {
    'title': 'Group-Wise Asymmetric INT4 Quantization',
    'title_zh': '分组非对称 INT4 量化',
    'difficulty': 'Medium',
    'category': '量化',
    'function_name': 'groupwise_int4_quantize',
    'starter_code': """def groupwise_int4_quantize(W, group_size=64, eps=1e-8):
    pass""",
    'hint': """View the input as `(out_features, num_groups, group_size)`, compute one min, max, scale, and zero point per group, then reshape q and its dequantization back.""",
    'hint_zh': '将输入视为 `(out_features, num_groups, group_size)`，逐组计算最小值、最大值、尺度和零点，再将量化值与反量化值恢复原形状。',
    'description_en': 'Implement per-group asymmetric 4-bit quantization with integer levels from 0 through 15.',
    'description_zh': '实现逐组非对称 4 位量化，整数表示范围为 0 到 15。',
    'theory_en': """Small groups adapt their scale and zero point to local weight ranges, reducing error compared with one tensor-wide quantizer.""",
    'theory_zh': '小分组可使尺度与零点适应局部权重范围，相比整张量共用量化器可减小误差。',
    'diagram_en': """```text
W -> groups -> min/max -> scale & zero point -> round/clamp [0,15] -> dequantize
```""",
    'diagram_zh': """```text
W -> 分组 -> 最小/最大值 -> 尺度与零点 -> 舍入/截断到 [0,15] -> 反量化
```""",
    "tests": [
        {
            "name": 'shapes and dtypes',
            "code": """import torch
W = torch.randn(3, 8)
q, scale, zp, deq = {fn}(W, group_size=4)
assert q.shape == deq.shape == W.shape
assert scale.shape == zp.shape == (3, 2)
assert q.dtype == torch.int32 and deq.dtype == torch.float32""",
        },
        {
            "name": 'exact asymmetric reference',
            "code": """import torch
W = torch.tensor([[-2., -1., 0., 4., 1., 3., 5., 7.]])
q, scale, zp, deq = {fn}(W, group_size=4)
g = W.reshape(1, 2, 4)
lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
rs = (hi - lo).clamp(min=1e-8) / 15.0
rz = torch.clamp(torch.round(-lo / rs), 0., 15.)
rq = torch.clamp(torch.round(g / rs + rz), 0., 15.).to(torch.int32)
rd = ((rq.float() - rz) * rs).reshape_as(W)
assert torch.equal(q, rq.reshape_as(W))
assert torch.equal(scale, rs.squeeze(-1)) and torch.equal(zp, rz.squeeze(-1))
assert torch.equal(deq, rd)""",
        },
        {
            "name": 'constant group edge case',
            "code": """import torch
W = torch.full((2, 4), 3.0)
q, scale, zp, deq = {fn}(W, group_size=2, eps=1e-6)
assert q.min() >= 0 and q.max() <= 15
assert torch.allclose(scale, torch.full((2, 2), 1e-6 / 15.0))
assert torch.isfinite(deq).all()""",
        },
        {
            "name": 'gradient through dequantization scale',
            "code": """import torch
W = torch.tensor([[-2., -0.5, 1., 4.], [-3., 0., 2., 5.]], requires_grad=True)
q, scale, zp, deq = {fn}(W, group_size=4)
deq.square().sum().backward()
assert q.shape == W.shape and scale.requires_grad
assert W.grad is not None and W.grad.shape == W.shape
assert torch.isfinite(W.grad).all()""",
        },
    ],
    'solution': '''def groupwise_int4_quantize(W, group_size=64, eps=1e-8):
    """执行分组非对称 INT4 量化 (Group-Wise INT4 Quantization)。"""
    out_features, in_features = W.shape
    num_groups = in_features // group_size

    # Step 1: 分组并计算局部仿射量化参数
    grouped = W.reshape(out_features, num_groups, group_size)
    group_min = grouped.amin(dim=-1, keepdim=True)
    group_max = grouped.amax(dim=-1, keepdim=True)
    scale = (group_max - group_min).clamp(min=eps) / 15.0
    zero_point = torch.clamp(torch.round(-group_min / scale), 0.0, 15.0)

    # Step 2: 舍入并截断到四位整数范围
    quantized = torch.clamp(
        torch.round(grouped / scale + zero_point), 0.0, 15.0
    ).to(torch.int32)

    # Step 3: 反量化并恢复原权重形状
    dequantized = ((quantized.float() - zero_point) * scale).reshape_as(W)
    return quantized.reshape_as(W), scale.squeeze(-1), zero_point.squeeze(-1), dequantized''',
    'demo': """torch.manual_seed(42)
W = torch.randn(8, 16)
quantized, scale, zero_point, dequantized = groupwise_int4_quantize(W, group_size=4)
print("Output shape:", quantized.shape)
print("Scale shape:", scale.shape)
print("Mean error:", (W - dequantized).abs().mean().item())""",
    'explanation': """### 核心思路
每组用局部最小值和最大值构造非对称仿射量化器，将权重映射到整数 $[0,15]$，再通过相同 `scale` 与 `zero_point` 反量化。

### 代码步骤拆解
- **Step 1（分组标定）**：`grouped` 为 `(out_features, num_groups, group_size)`，逐组得到尺度与零点。
- **Step 2（整数映射）**：舍入后截断到 16 个等级并转为 `int32`。
- **Step 3（反量化）**：按 $(q-z)s$ 恢复浮点近似并重塑为原形状。

### 重点与易错点分析
- **注意点 1（常数组）**：范围用 `clamp(min=eps)`，避免零尺度除法。
- **注意点 2（无符号等级）**：INT4 等级是 0 到 15，而不是对称的 -8 到 7。
- **注意点 3（形状约束）**：`in_features` 必须被 `group_size` 整除，否则无法直接重塑。

### 复杂度分析
- **时间复杂度**：$O(OI)$，遍历全部权重完成标定和量化。
- **空间复杂度**：$O(OI)$，量化值与反量化值和原权重同规模。""",
}

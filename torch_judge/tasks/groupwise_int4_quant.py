"""Group-wise asymmetric INT4 quantization task."""

TASK = {
    "title": "Group-Wise Asymmetric INT4 Quantization",
    "title_zh": "分组非对称 INT4 量化",
    "difficulty": "Medium",
    "category": "量化",
    "function_name": "groupwise_int4_quantize",
    "starter_code": '''def groupwise_int4_quantize(W, group_size=64, eps=1e-8):
    pass
''',
    "hint": "View the input as `(out_features, num_groups, group_size)`, compute one min, max, scale, and zero point per group, then reshape q and its dequantization back.",
    "hint_zh": "将输入视为 `(out_features, num_groups, group_size)`，逐组计算最小值、最大值、尺度和零点，再将量化值与反量化值恢复原形状。",
    "description_en": "Implement per-group asymmetric 4-bit quantization with integer levels from 0 through 15.",
    "description_zh": "实现逐组非对称 4 位量化，整数表示范围为 0 到 15。",
    "theory_en": "Small groups adapt their scale and zero point to local weight ranges, reducing error compared with one tensor-wide quantizer.",
    "theory_zh": "小分组可使尺度与零点适应局部权重范围，相比整张量共用量化器可减小误差。",
    "diagram_en": "```text\nW -> groups -> min/max -> scale & zero point -> round/clamp [0,15] -> dequantize\n```",
    "diagram_zh": "```text\nW -> 分组 -> 最小/最大值 -> 尺度与零点 -> 舍入/截断到 [0,15] -> 反量化\n```",
    "tests": [
        {
            "name": "shapes and dtypes",
            "code": '''import torch
W = torch.randn(3, 8)
q, scale, zp, deq = {fn}(W, group_size=4)
assert q.shape == deq.shape == W.shape
assert scale.shape == zp.shape == (3, 2)
assert q.dtype == torch.int32 and deq.dtype == torch.float32
''',
        },
        {
            "name": "exact asymmetric reference",
            "code": '''import torch
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
assert torch.equal(deq, rd)
''',
        },
        {
            "name": "constant group edge case",
            "code": '''import torch
W = torch.full((2, 4), 3.0)
q, scale, zp, deq = {fn}(W, group_size=2, eps=1e-6)
assert q.min() >= 0 and q.max() <= 15
assert torch.allclose(scale, torch.full((2, 2), 1e-6 / 15.0))
assert torch.isfinite(deq).all()
''',
        },
        {
            "name": "gradient through dequantization scale",
            "code": '''import torch
W = torch.tensor([[-2., -0.5, 1., 4.], [-3., 0., 2., 5.]], requires_grad=True)
q, scale, zp, deq = {fn}(W, group_size=4)
deq.square().sum().backward()
assert q.shape == W.shape and scale.requires_grad
assert W.grad is not None and W.grad.shape == W.shape
assert torch.isfinite(W.grad).all()
''',
        },
    ],
    "solution": '''import torch

def groupwise_int4_quantize(W, group_size=64, eps=1e-8):
    out_features, in_features = W.shape
    num_groups = in_features // group_size
    W_g = W.reshape(out_features, num_groups, group_size)
    w_min = W_g.amin(dim=-1, keepdim=True)
    w_max = W_g.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min).clamp(min=eps) / 15.0
    zero_point = torch.clamp(torch.round(-w_min / scale), 0.0, 15.0)
    q = torch.clamp(torch.round(W_g / scale + zero_point), 0.0, 15.0).to(torch.int32)
    W_dequant = ((q.float() - zero_point) * scale).reshape(W.shape)
    return q.reshape(W.shape), scale.squeeze(-1), zero_point.squeeze(-1), W_dequant
''',
    "demo": '''W = torch.randn(128, 256)
q, scale, zero_point, W_dequant = groupwise_int4_quantize(W, group_size=64)
print(q.dtype, scale.shape, (W - W_dequant).abs().mean().item())
''',
    "explanation": "Each output-row group receives its own affine quantizer. The integer tensor stores 4-bit levels in int32, while scale and zero point retain one value per group.",
}

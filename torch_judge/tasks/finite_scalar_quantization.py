"""Finite Scalar Quantization task."""

TASK = {
    "title": "Finite Scalar Quantization (FSQ)",
    "title_zh": "有限标量量化（FSQ）",
    "difficulty": "Medium",
    "category": "扩散与流模型",
    "function_name": "fsq_quantize",
    "starter_code": '''def fsq_quantize(z, levels):
    pass
''',
    "hint": "Process each final-axis coordinate with tanh, straight-through rounding, and normalization. Accumulate integer digits using a basis multiplied by each preceding level.",
    "hint_zh": "逐个处理最后一维坐标：tanh、有直通梯度的舍入、归一化。使用由先前各级别累乘得到的基数累加整数位。",
    "description_en": "Implement FSQ for z shaped (...,D) and D integer levels, each at least 2. Bound and straight-through round each scalar, normalize it to [-1,1], and return (z_hat,code), where code is a mixed-radix integer.",
    "description_zh": "为形状 (...,D) 的 z 和 D 个不小于 2 的整数级别实现 FSQ。对各标量限幅并直通舍入，再归一化到 [-1,1]，返回 (z_hat,code)，其中 code 是混合进制整数。",
    "theory_en": "FSQ replaces vector-codebook lookup with independent scalar grids. Mixed-radix encoding maps the D quantized digits bijectively to integers from zero to product(levels)-1.",
    "theory_zh": "FSQ 用独立标量网格替代向量码本查找。混合进制编码将 D 个量化数字双射到 0 至各级别乘积减一的整数。",
    "diagram_en": "```text\nz_i -> tanh bound -> STE round -> normalize -> z_hat_i\n                         \\-> integer digits -> mixed-radix code\n```",
    "diagram_zh": "```text\nz_i -> tanh 限幅 -> STE 舍入 -> 归一化 -> z_hat_i\n                         \\-> 整数位 -> 混合进制编码\n```",
    "tests": [
        {
            "name": "output shapes and code range",
            "code": '''import torch
z=torch.randn(2,3,3)
q,code={fn}(z,[3,4,5])
assert q.shape==z.shape and code.shape==(2,3) and code.dtype==torch.long
assert q.min()>=-1 and q.max()<=1 and code.min()>=0 and code.max()<60
''',
        },
        {
            "name": "exact scalar reference",
            "code": '''import torch
z=torch.tensor([[0.,.5,-.7],[2.,-2.,.1]])
levels=[3,4,5]
q,code={fn}(z,levels)
vals=[]; ref_code=torch.zeros(2,dtype=torch.long); basis=1
for i,L in enumerate(levels):
    h=(L-1)/2.; b=h*(torch.tanh(z[:,i])+1); r=torch.round(b)
    vals.append(r/h-1); ref_code += r.long()*basis; basis*=L
assert torch.equal(q,torch.stack(vals,-1)) and torch.equal(code,ref_code)
''',
        },
        {
            "name": "extreme-value edge codes",
            "code": '''import torch
z=torch.tensor([[-100.,-100.],[100.,100.]])
q,code={fn}(z,(2,3))
assert torch.equal(code,torch.tensor([0,5]))
assert torch.equal(q,torch.tensor([[-1.,-1.],[1.,1.]]))
''',
        },
        {
            "name": "straight-through gradient flow",
            "code": '''import torch
z=torch.randn(4,3,requires_grad=True)
q,code={fn}(z,[3,5,7])
q.sum().backward()
expected=torch.stack([1-torch.tanh(z.detach()[:,i]).square() for i in range(3)],-1)
assert z.grad is not None and torch.allclose(z.grad,expected,atol=1e-6)
assert code.requires_grad is False
''',
        },
    ],
    "solution": '''import torch

def fsq_quantize(z, levels):
    quantized = []
    code = torch.zeros_like(z[..., 0], dtype=torch.long)
    basis = 1
    for i in range(len(levels)):
        level = levels[i]
        half_width = (level - 1) / 2.0
        bounded = half_width * (torch.tanh(z[..., i]) + 1.0)
        rounded = bounded + (torch.round(bounded) - bounded).detach()
        quantized.append(rounded / half_width - 1.0)
        code = code + torch.round(bounded).long() * basis
        basis = basis * level
    z_hat = torch.stack(quantized, dim=-1)
    return z_hat, code
''',
    "demo": '''z = torch.randn(2, 4, 3, requires_grad=True)
z_hat, code = fsq_quantize(z, [3, 5, 7])
print(z_hat.shape, code.shape, code.min().item(), code.max().item())
''',
    "explanation": "The first coordinate is the least-significant mixed-radix digit. STE rounding is discrete forward but differentiates as the identity with respect to each bounded scalar.",
}

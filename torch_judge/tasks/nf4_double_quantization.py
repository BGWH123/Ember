"""QLoRA NF4 and double-quantization task."""

TASK = {
    "title": "QLoRA NF4 Codebook & Double Quantization",
    "title_zh": "QLoRA NF4 分位数码本与双重量化",
    "difficulty": "Hard",
    "category": "量化",
    "function_name": "nf4_double_quantize",
    "starter_code": '''import torch

def nf4_double_quantize(W, codebook, block_size=64, eps=1e-8):
    pass
''',
    "hint": "Normalize each flattened block by its absmax and use a broadcasted distance to all 16 codebook entries. Then quantize the vector of absmax values with one global 8-bit scale.",
    "hint_zh": "用各展平块的绝对值最大值归一化，并广播计算到 16 个码本值的距离；随后用一个全局 8 位尺度量化 absmax 向量。",
    "description_en": "Quantize weights blockwise to a 16-entry NF4 codebook, then double-quantize the block scales and reconstruct the dequantized tensor.",
    "description_zh": "按块将权重量化到 16 项 NF4 码本，再双重量化块尺度并重建反量化张量。",
    "theory_en": "NF4 allocates its levels to a distribution suited to normally distributed weights. Double quantization reduces metadata cost by quantizing the per-block absmax values themselves.",
    "theory_zh": "NF4 将量化级分配给适合近似正态权重的分布。双重量化进一步量化逐块 absmax，从而降低元数据开销。",
    "diagram_en": "```text\nW -> flat blocks -> /absmax -> nearest NF4 index\nabsmax -> /c2 -> int8 levels -> dequantized absmax -> reconstruct W\n```",
    "diagram_zh": "```text\nW -> 展平分块 -> /absmax -> 最近 NF4 索引\nabsmax -> /c2 -> int8 级别 -> 反量化 absmax -> 重建 W\n```",
    "tests": [
        {
            "name": "returned tensor shapes",
            "code": '''import torch
cb = torch.linspace(-1,1,16)
W = torch.randn(4,8)
idx, qa, c2, deq = {fn}(W,cb,block_size=8)
assert idx.shape == deq.shape == W.shape
assert qa.shape == (4,) and c2.shape == torch.Size([])
assert idx.dtype == torch.long
''',
        },
        {
            "name": "exact two-level reference",
            "code": '''import torch
cb = torch.tensor([-1.,-.8,-.6,-.45,-.3,-.2,-.1,0.,.08,.16,.25,.4,.55,.7,.85,1.],dtype=torch.double)
W = torch.tensor([[1.,-.7,.2,0.],[-2.,.9,-.4,.1]],dtype=torch.double)
idx, qa, c2, deq = {fn}(W,cb,block_size=4,eps=1e-12)
b = W.reshape(-1,4); a = b.abs().amax(-1).clamp(min=1e-12)
ri = torch.argmin((b.div(a[:,None]).unsqueeze(-1)-cb).abs(),-1)
rc = a.abs().max().clamp(min=1e-12)/127.; rq = torch.clamp(torch.round(a/rc),-127.,127.)
rd = (cb[ri]*(rq*rc)[:,None]).reshape_as(W)
assert torch.equal(idx,ri.reshape_as(W)) and torch.equal(qa,rq)
assert torch.equal(c2,rc) and torch.equal(deq,rd)
''',
        },
        {
            "name": "zero-block edge case",
            "code": '''import torch
cb = torch.linspace(-1,1,16)
W = torch.zeros(16)
idx, qa, c2, deq = {fn}(W,cb,block_size=8,eps=1e-6)
assert qa.shape == (2,) and torch.equal(qa,torch.full_like(qa,127.))
assert torch.allclose(c2,torch.tensor(1e-6/127.))
assert torch.all(deq.abs() <= 1e-6) and torch.isfinite(deq).all()
''',
        },
        {
            "name": "gradient through reconstruction scales",
            "code": '''import torch
cb = torch.linspace(-1,1,16,requires_grad=True)
W = torch.tensor([1.,-.4,.2,.1,-2.,.3,.6,-.2],requires_grad=True)
idx, qa, c2, deq = {fn}(W,cb,block_size=4)
deq.square().sum().backward()
assert deq.requires_grad and idx.shape == W.shape
assert W.grad is not None and cb.grad is not None
assert torch.isfinite(W.grad).all() and torch.isfinite(cb.grad).all()
''',
        },
    ],
    "solution": '''import torch

def nf4_double_quantize(W, codebook, block_size=64, eps=1e-8):
    W_blocks = W.reshape(-1, block_size)
    absmax = W_blocks.abs().amax(dim=-1).clamp(min=eps)
    W_norm = W_blocks / absmax.unsqueeze(-1)
    indices = torch.argmin((W_norm.unsqueeze(-1) - codebook).abs(), dim=-1)
    c2 = absmax.abs().max().clamp(min=eps) / 127.0
    q_absmax = torch.clamp(torch.round(absmax / c2), -127.0, 127.0)
    absmax_deq = q_absmax * c2
    W_deq = (codebook[indices] * absmax_deq.unsqueeze(-1)).reshape(W.shape)
    return indices.reshape(W.shape), q_absmax, c2, W_deq
''',
    "demo": '''codebook = torch.linspace(-1,1,16)
W = torch.randn(128,128)
indices, q_scales, c2, W_deq = nf4_double_quantize(W,codebook,block_size=64)
print(indices.shape, q_scales.shape, c2.item(), (W-W_deq).abs().mean().item())
''',
    "explanation": "Indices encode one of 16 normalized values per weight. Instead of retaining every block absmax in full precision, the function stores 8-bit-like levels plus the scalar second-level scale c2.",
}

"""QLoRA NF4 Codebook & Double Quantization task."""

TASK = {
    'title': 'QLoRA NF4 Codebook & Double Quantization',
    'title_zh': 'QLoRA NF4 分位数码本与双重量化',
    'difficulty': 'Hard',
    'category': '量化',
    'function_name': 'nf4_double_quantize',
    'starter_code': """def nf4_double_quantize(W, codebook, block_size=64, eps=1e-08):
    pass""",
    'hint': """Normalize each flattened block by its absmax and use a broadcasted distance to all 16 codebook entries. Then quantize the vector of absmax values with one global 8-bit scale.""",
    'hint_zh': '用各展平块的绝对值最大值归一化，并广播计算到 16 个码本值的距离；随后用一个全局 8 位尺度量化 absmax 向量。',
    'description_en': """Quantize weights blockwise to a 16-entry NF4 codebook, then double-quantize the block scales and reconstruct the dequantized tensor.""",
    'description_zh': '按块将权重量化到 16 项 NF4 码本，再双重量化块尺度并重建反量化张量。',
    'theory_en': """NF4 allocates its levels to a distribution suited to normally distributed weights. Double quantization reduces metadata cost by quantizing the per-block absmax values themselves.""",
    'theory_zh': 'NF4 将量化级分配给适合近似正态权重的分布。双重量化进一步量化逐块 absmax，从而降低元数据开销。',
    'diagram_en': """```text
W -> flat blocks -> /absmax -> nearest NF4 index
absmax -> /c2 -> int8 levels -> dequantized absmax -> reconstruct W
```""",
    'diagram_zh': """```text
W -> 展平分块 -> /absmax -> 最近 NF4 索引
absmax -> /c2 -> int8 级别 -> 反量化 absmax -> 重建 W
```""",
    "tests": [
        {
            "name": 'returned tensor shapes',
            "code": """import torch
cb = torch.linspace(-1,1,16)
W = torch.randn(4,8)
idx, qa, c2, deq = {fn}(W,cb,block_size=8)
assert idx.shape == deq.shape == W.shape
assert qa.shape == (4,) and c2.shape == torch.Size([])
assert idx.dtype == torch.long""",
        },
        {
            "name": 'exact two-level reference',
            "code": """import torch
cb = torch.tensor([-1.,-.8,-.6,-.45,-.3,-.2,-.1,0.,.08,.16,.25,.4,.55,.7,.85,1.],dtype=torch.double)
W = torch.tensor([[1.,-.7,.2,0.],[-2.,.9,-.4,.1]],dtype=torch.double)
idx, qa, c2, deq = {fn}(W,cb,block_size=4,eps=1e-12)
b = W.reshape(-1,4); a = b.abs().amax(-1).clamp(min=1e-12)
ri = torch.argmin((b.div(a[:,None]).unsqueeze(-1)-cb).abs(),-1)
rc = a.abs().max().clamp(min=1e-12)/127.; rq = torch.clamp(torch.round(a/rc),-127.,127.)
rd = (cb[ri]*(rq*rc)[:,None]).reshape_as(W)
assert torch.equal(idx,ri.reshape_as(W)) and torch.equal(qa,rq)
assert torch.equal(c2,rc) and torch.equal(deq,rd)""",
        },
        {
            "name": 'zero-block edge case',
            "code": """import torch
cb = torch.linspace(-1,1,16)
W = torch.zeros(16)
idx, qa, c2, deq = {fn}(W,cb,block_size=8,eps=1e-6)
assert qa.shape == (2,) and torch.equal(qa,torch.full_like(qa,127.))
assert torch.allclose(c2,torch.tensor(1e-6/127.))
assert torch.all(deq.abs() <= 1e-6) and torch.isfinite(deq).all()""",
        },
        {
            "name": 'gradient through reconstruction scales',
            "code": """import torch
cb = torch.linspace(-1,1,16,requires_grad=True)
W = torch.tensor([1.,-.4,.2,.1,-2.,.3,.6,-.2],requires_grad=True)
idx, qa, c2, deq = {fn}(W,cb,block_size=4)
deq.square().sum().backward()
assert deq.requires_grad and idx.shape == W.shape
assert W.grad is not None and cb.grad is not None
assert torch.isfinite(W.grad).all() and torch.isfinite(cb.grad).all()""",
        },
    ],
    'solution': '''def nf4_double_quantize(W, codebook, block_size=64, eps=1e-08):
    """QLoRA NF4 分位数码本与双重量化 (QLoRA NF4 Codebook & Double Quantization)。"""
    # Step 1: 展平权重并按 block_size 分块，用逐块绝对最大值归一化到 NF4 码本范围
    weight_blocks = W.reshape(-1, block_size)
    block_scales = weight_blocks.abs().amax(dim=-1).clamp(min=eps)
    normalized_blocks = weight_blocks / block_scales.unsqueeze(-1)
    # Step 2: 为每个归一化权重选择距离最近的 16 级 NF4 码字索引
    codebook_indices = torch.argmin((normalized_blocks.unsqueeze(-1) - codebook).abs(), dim=-1)
    # Step 3: 用全局 8 位尺度再次量化块尺度，并结合码字重建反量化权重
    second_scale = block_scales.abs().max().clamp(min=eps) / 127.0
    quantized_scales = torch.clamp(torch.round(block_scales / second_scale), -127.0, 127.0)
    dequantized_scales = quantized_scales * second_scale
    dequantized_weight = (codebook[codebook_indices] * dequantized_scales.unsqueeze(-1)).reshape(W.shape)
    return (codebook_indices.reshape(W.shape), quantized_scales, second_scale, dequantized_weight)''',
    'demo': """torch.manual_seed(42)
codebook = torch.linspace(-1, 1, 16)
W = torch.randn(128, 128)
indices, q_scales, c2, W_deq = nf4_double_quantize(W, codebook, block_size=64)
print('权重/索引形状:', tuple(W.shape), tuple(indices.shape))
print('块数与二级尺度:', q_scales.numel(), f'{c2.item():.6f}')
print('平均绝对重建误差:', f'{(W - W_deq).abs().mean().item():.4f}')""",
    'explanation': r"""### 核心思路
NF4 先为每个 B 元素块取 $a_b=\max_j|W_{b,j}|$，归一化 $u_{b,j}=W_{b,j}/a_b$ 后选择 16 项非均匀码本中最近的值 $q(u_{b,j})$。双重量化再用全局尺度 $c_2=\max_b a_b/127$ 将所有块尺度量化为 8 位级别 $q_b=\operatorname{round}(a_b/c_2)$，最终重建 $\hat W_{b,j}=\mathrm{codebook}[i_{b,j}]\,(q_bc_2)$。

### 代码步骤拆解
- **Step 1（逐块 absmax 归一化）**：`W.reshape(-1,block_size)` 得到 `(N/B,B)`，`block_scales` 为每块一个标量，除法把每块动态范围映射到约 $[-1,1]$。
- **Step 2（NF4 最近邻编码）**：广播差值形成 `(N/B,B,16)`，`argmin` 返回每个权重的码本索引，索引最终恢复为 `W.shape`。
- **Step 3（尺度的二次量化）**：`second_scale` 把最大块尺度映射到 127；整数级 `quantized_scales` 反量化后与查表码字相乘得到 `dequantized_weight`。

### 重点与易错点分析
- **注意点 1（零块的尺度约定）**：`clamp(min=eps)` 令全零块尺度成为 eps；若全部块都为零，则 $c_2=\varepsilon/127$ 且量化尺度为 127，避免除零并符合测试约定。
- **注意点 2（分块前提）**：直接 `reshape(-1,block_size)` 要求 `W.numel()` 能被块大小整除；生产实现通常还需对尾块填充并在重建后裁剪。
- **注意点 3（离散路径的梯度）**：`argmin` 和 `round` 本身不可微，但选中的 `codebook` 值与由 `W` 计算的连续尺度仍参与重建梯度；这不是完整量化感知训练中的 STE。

### 复杂度分析
- **时间复杂度**：设权重数为 N、码本大小为 C=16，最近邻距离计算为 $O(NC)$，尺度量化和重建为 $O(N)$。
- **空间复杂度**：广播距离张量占 $O(NC)$ 并主导峰值；索引与反量化权重各占 $O(N)$，块尺度占 $O(N/B)$。""",
}

"""Finite Scalar Quantization (FSQ) task."""

TASK = {
    'title': 'Finite Scalar Quantization (FSQ)',
    'title_zh': '有限标量量化（FSQ）',
    'difficulty': 'Medium',
    'category': '扩散与流模型',
    'function_name': 'fsq_quantize',
    'starter_code': """def fsq_quantize(z, levels):
    pass""",
    'hint': """Process each final-axis coordinate with tanh, straight-through rounding, and normalization. Accumulate integer digits using a basis multiplied by each preceding level.""",
    'hint_zh': '逐个处理最后一维坐标：tanh、有直通梯度的舍入、归一化。使用由先前各级别累乘得到的基数累加整数位。',
    'description_en': """Implement FSQ for z shaped (...,D) and D integer levels, each at least 2. Bound and straight-through round each scalar, normalize it to [-1,1], and return (z_hat,code), where code is a mixed-radix integer.""",
    'description_zh': '为形状 (...,D) 的 z 和 D 个不小于 2 的整数级别实现 FSQ。对各标量限幅并直通舍入，再归一化到 [-1,1]，返回 (z_hat,code)，其中 code 是混合进制整数。',
    'theory_en': """FSQ replaces vector-codebook lookup with independent scalar grids. Mixed-radix encoding maps the D quantized digits bijectively to integers from zero to product(levels)-1.""",
    'theory_zh': 'FSQ 用独立标量网格替代向量码本查找。混合进制编码将 D 个量化数字双射到 0 至各级别乘积减一的整数。',
    'diagram_en': r"""```text
z_i -> tanh bound -> STE round -> normalize -> z_hat_i
                         \-> integer digits -> mixed-radix code
```""",
    'diagram_zh': r"""```text
z_i -> tanh 限幅 -> STE 舍入 -> 归一化 -> z_hat_i
                         \-> 整数位 -> 混合进制编码
```""",
    "tests": [
        {
            "name": 'output shapes and code range',
            "code": """import torch
z=torch.randn(2,3,3)
q,code={fn}(z,[3,4,5])
assert q.shape==z.shape and code.shape==(2,3) and code.dtype==torch.long
assert q.min()>=-1 and q.max()<=1 and code.min()>=0 and code.max()<60""",
        },
        {
            "name": 'exact scalar reference',
            "code": """import torch
z=torch.tensor([[0.,.5,-.7],[2.,-2.,.1]])
levels=[3,4,5]
q,code={fn}(z,levels)
vals=[]; ref_code=torch.zeros(2,dtype=torch.long); basis=1
for i,L in enumerate(levels):
    h=(L-1)/2.; b=h*(torch.tanh(z[:,i])+1); r=torch.round(b)
    vals.append(r/h-1); ref_code += r.long()*basis; basis*=L
assert torch.equal(q,torch.stack(vals,-1)) and torch.equal(code,ref_code)""",
        },
        {
            "name": 'extreme-value edge codes',
            "code": """import torch
z=torch.tensor([[-100.,-100.],[100.,100.]])
q,code={fn}(z,(2,3))
assert torch.equal(code,torch.tensor([0,5]))
assert torch.equal(q,torch.tensor([[-1.,-1.],[1.,1.]]))""",
        },
        {
            "name": 'straight-through gradient flow',
            "code": """import torch
z=torch.randn(4,3,requires_grad=True)
q,code={fn}(z,[3,5,7])
q.sum().backward()
expected=torch.stack([1-torch.tanh(z.detach()[:,i]).square() for i in range(3)],-1)
assert z.grad is not None and torch.allclose(z.grad,expected,atol=1e-6)
assert code.requires_grad is False""",
        },
    ],
    'solution': '''def fsq_quantize(z, levels):
    """执行有限标量量化与混合进制编码 (Finite Scalar Quantization)。"""
    quantized_coords = []
    code = torch.zeros_like(z[..., 0], dtype=torch.long)
    basis = 1

    # Step 1: 逐坐标限幅，并用直通估计器执行离散舍入
    for coord_idx, level in enumerate(levels):
        half_width = (level - 1) / 2.0
        bounded = half_width * (torch.tanh(z[..., coord_idx]) + 1.0)
        discrete = torch.round(bounded)
        rounded_ste = bounded + (discrete - bounded).detach()
        quantized_coords.append(rounded_ste / half_width - 1.0)

        # Step 2: 按混合进制基数累加离散码
        code = code + discrete.long() * basis
        basis *= level

    # Step 3: 将各标量量化结果重新堆叠到特征维
    z_hat = torch.stack(quantized_coords, dim=-1)
    return z_hat, code''',
    'demo': """torch.manual_seed(42)
z = torch.randn(2, 4, 3, requires_grad=True)
z_hat, code = fsq_quantize(z, [3, 5, 7])
print("Quantized shape:", z_hat.shape)
print("Code shape:", code.shape)
print("Code range:", code.min().item(), code.max().item())""",
    'explanation': r"""### 核心思路
FSQ 对每个坐标独立执行 tanh 限幅与离散舍入，并用 STE 保留梯度；离散数字再按混合进制 $c=\sum_i d_i\prod_{j<i}L_j$ 编成唯一整数。

### 代码步骤拆解
- **Step 1（标量量化）**：每个坐标映射到 $[0,L_i-1]$，舍入后再归一化回 $[-1,1]$。
- **Step 2（混合进制）**：`basis` 从 1 开始按各 `level` 累乘，使第一个坐标成为最低有效位。
- **Step 3（结果堆叠）**：所有量化坐标沿最后一维堆叠，恢复与 `z` 相同的形状。

### 重点与易错点分析
- **注意点 1（直通梯度）**：`bounded + (discrete - bounded).detach()` 前向等于舍入值，反向对 `bounded` 的导数为 1。
- **注意点 2（编码离散性）**：`code` 使用普通 `discrete.long()`，不应参与梯度；其范围为 $[0,\prod_iL_i-1]$。
- **注意点 3（级别约束）**：每个 `level` 至少为 2，否则 `half_width` 为零并导致归一化除零。

### 复杂度分析
- **时间复杂度**：$O(ND)$，逐坐标处理全部输入元素。
- **空间复杂度**：$O(ND)$，主要保存各坐标的量化结果。""",
}

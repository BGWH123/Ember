"""NTK-aware RoPE Scaling task."""

TASK = {
    'title': 'NTK-aware RoPE Scaling',
    'title_zh': 'NTK-aware RoPE 缩放',
    'difficulty': 'Easy',
    'category': '位置编码与嵌入',
    'function_name': 'ntk_rope',
    'starter_code': """def ntk_rope(q, k, scale):
    pass""",
    'hint': '`new_base = 10000 * scale**(D/(D-2))` → apply standard RoPE with `new_base` instead of `10000`.',
    'hint_zh': '`new_base = 10000 * scale**(D/(D-2))` → 用 `new_base` 替换 `10000` 应用标准 RoPE。',
    'description_en': """Implement NTK-aware RoPE scaling for long-context extrapolation.

Standard RoPE degrades on sequences longer than the training context. NTK-aware scaling adjusts the base frequency so that high-frequency dimensions are preserved while low-frequency ones are stretched, enabling extrapolation without fine-tuning.

**Signature:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`

**Parameters:**
- `q, k` — tensors of shape `(B, S, D)`
- `scale` — context length ratio (new_len / train_len), e.g. 4.0 for 4× context

**Returns:** rotated `(q, k)` with same shapes

**NTK base:** `base_new = 10000 * scale^(D/(D-2))`

Then apply standard RoPE with the new base.""",
    'description_zh': """实现 NTK-aware RoPE 缩放，用于长上下文外推。

标准 RoPE 在超过训练上下文长度的序列上性能下降。NTK-aware 缩放调整基础频率，保留高频维度同时拉伸低频维度，无需微调即可外推。

**签名:** `ntk_rope(q, k, scale) -> (Tensor, Tensor)`

**参数:**
- `q, k` — 形状 `(B, S, D)` 的张量
- `scale` — 上下文长度比例（新长度/训练长度），如 4× 上下文传 4.0

**返回:** 旋转后的 `(q, k)`，形状不变

**NTK 基底:** `base_new = 10000 * scale^(D/(D-2))`

然后用新基底应用标准 RoPE。""",
    "tests": [
        {
            "name": 'Output shapes',
            "code": """import torch
torch.manual_seed(0)
q = torch.randn(2, 16, 32)
k = torch.randn(2, 16, 32)
q_rot, k_rot = {fn}(q, k, scale=4.0)
assert q_rot.shape == q.shape, f'Q shape mismatch: {q_rot.shape}'
assert k_rot.shape == k.shape, f'K shape mismatch: {k_rot.shape}'""",
        },
        {
            "name": 'scale=1 matches standard RoPE',
            "code": """import torch
torch.manual_seed(1)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q_ntk, k_ntk = {fn}(q, k, scale=1.0)
# Standard RoPE with base=10000
B, S, D = q.shape
pos = torch.arange(S).float().unsqueeze(1)
dim = torch.arange(0, D, 2).float()
freqs = 1.0 / (10000.0 ** (dim / D))
angles = pos * freqs
cos_a, sin_a = torch.cos(angles), torch.sin(angles)
def rotate(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos_a - x2*sin_a, x1*sin_a + x2*cos_a], dim=-1).flatten(-2)
q_std, k_std = rotate(q), rotate(k)
assert torch.allclose(q_ntk, q_std, atol=1e-5), 'scale=1 should match standard RoPE'""",
        },
        {
            "name": 'Preserves norms',
            "code": """import torch
torch.manual_seed(2)
q = torch.randn(1, 12, 32)
k = torch.randn(1, 12, 32)
q_rot, k_rot = {fn}(q, k, scale=8.0)
assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4), 'RoPE must preserve norms'""",
        },
        {
            "name": 'scale>1 changes frequencies',
            "code": """import torch
torch.manual_seed(3)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q1, _ = {fn}(q, k, scale=1.0)
q4, _ = {fn}(q, k, scale=4.0)
assert not torch.allclose(q1, q4, atol=1e-3), 'Different scales should produce different rotations'""",
        },
        {
            "name": 'NTK base formula correctness',
            "code": """import torch
torch.manual_seed(0)
B, S, D = 1, 4, 8
scale = 4.0
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)
q_rot, k_rot = {fn}(q, k, scale)
# Reference with exact NTK base
new_base = 10000.0 * (scale ** (D / (D - 2)))
pos = torch.arange(S).float().unsqueeze(1)
dim = torch.arange(0, D, 2).float()
freqs = 1.0 / (new_base ** (dim / D))
angles = pos * freqs
cos_a, sin_a = torch.cos(angles), torch.sin(angles)
def rotate(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos_a - x2*sin_a, x1*sin_a + x2*cos_a], dim=-1).flatten(-2)
assert torch.allclose(q_rot, rotate(q), atol=1e-5), 'NTK base formula mismatch'""",
        },
    ],
    'solution': '''def ntk_rope(q, k, scale):
    """NTK-aware RoPE 缩放 (NTK-aware RoPE Scaling)。"""
    # Step 1: 按上下文扩展倍率修正 RoPE 基底，使低频通道获得更长波长
    B, S, D = q.shape
    new_base = 10000.0 * scale ** (D / (D - 2))
    # Step 2: 用新基底生成每个序列位置与偶奇通道对的旋转相位
    positions = torch.arange(S, device=q.device).float().unsqueeze(1)
    pair_dims = torch.arange(0, D, 2, device=q.device).float()
    frequencies = 1.0 / new_base ** (pair_dims / D)
    angles = positions * frequencies
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    def rotate(tensor):
        even_values, odd_values = (tensor[..., 0::2], tensor[..., 1::2])
        return torch.stack([even_values * cos_angles - odd_values * sin_angles, even_values * sin_angles + odd_values * cos_angles], dim=-1).flatten(-2)
    # Step 3: 对 q、k 应用相同的二维正交旋转并恢复交错通道布局
    return (rotate(q), rotate(k))''',
    'demo': """torch.manual_seed(42)
B, S, D = 1, 8, 16
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)
q_rot, k_rot = ntk_rope(q, k, scale=4.0)
new_base = 10000.0 * 4.0 ** (D / (D - 2))
print('NTK 调整后基底:', f'{new_base:.2f}')
print('旋转后 q/k 形状:', tuple(q_rot.shape), tuple(k_rot.shape))
print('q 范数最大偏差:', f'{(q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item():.2e}')""",
    'explanation': r"""### 核心思路
NTK-aware 缩放不直接压缩位置，而是把 RoPE 基底改为 $b'=10000\,s^{D/(D-2)}$，再使用频率 $\omega_i=(b')^{-2i/D}$。当上下文倍率 $s>1$ 时，低频维度的角速度显著降低、周期拉长，而 $i=0$ 的最高频率仍为 1；随后 q、k 的相邻通道对按位置相位 $p\omega_i$ 同步旋转。

### 代码步骤拆解
- **Step 1（NTK 基底外推）**：从 `(B,S,D)` 读取 D，用 `scale ** (D/(D-2))` 计算 `new_base`；`scale=1` 时精确退化为标准 RoPE 基底 10000。
- **Step 2（位置—频率外积）**：`positions` 为 `(S,1)`，`frequencies` 为 `(D/2,)`，广播产生 `(S,D/2)` 的 `angles`，供所有 batch 共享。
- **Step 3（同步旋转 q/k）**：拆出偶、奇通道后执行 $(x_e\cos-x_o\sin,\ x_e\sin+x_o\cos)$，stack/flatten 恢复 `(B,S,D)`。

### 重点与易错点分析
- **注意点 1（指数必须是 D/(D-2)）**：漏掉该维度相关指数会变成简单基底缩放，无法满足 NTK 频谱保持关系；公式也要求偶数 D 且 $D>2$。
- **注意点 2（频率索引的含义）**：`pair_dims=0,2,...` 后除以 D，等价于 $2i/D$；若改成连续 `0..D/2-1` 却仍除 D，会得到错误的频率衰减。
- **注意点 3（旋转保持范数与相对相位）**：每个 2D 块是正交矩阵，所以 q/k 的逐 token L2 范数不变；两者必须使用同一角度，注意力点积才表达位置差而非绝对相位错配。

### 复杂度分析
- **时间复杂度**：$O(BSD)$，q 和 k 的每个通道各参与常数次三角系数乘加。
- **空间复杂度**：共享的角度、正弦和余弦表占 $O(SD)$；返回的两组旋转张量占 $O(BSD)$。""",
}

"""Rotary Position Embedding (RoPE) task."""

TASK = {
    'title': 'Rotary Position Embedding (RoPE)',
    'title_zh': '旋转位置编码（RoPE）',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'apply_rope',
    'starter_code': """def apply_rope(q, k):
    pass""",
    'hint': """1. `freqs = 1 / 10000^(2i/D)` for i in `range(D//2)`
2. `angles = pos[:, None] * freqs` → `cos_a`, `sin_a`
3. Split: `x_e = x[..., 0::2]`, `x_o = x[..., 1::2]`
4. Rotate: `[x_e*cos - x_o*sin, x_e*sin + x_o*cos]` → stack → flatten""",
    'hint_zh': """1. `freqs = 1 / 10000^(2i/D)`，i 取 `range(D//2)`
2. `angles = pos[:, None] * freqs` → `cos_a`、`sin_a`
3. 拆分：`x_e = x[..., 0::2]`，`x_o = x[..., 1::2]`
4. 旋转：`[x_e*cos - x_o*sin, x_e*sin + x_o*cos]` → stack → flatten""",
    'description_en': """Implement Rotary Position Embedding (RoPE).

RoPE encodes position information by rotating query and key vectors in pairs, enabling relative position awareness through dot-product properties.

**Signature:** `apply_rope(q, k) -> (Tensor, Tensor)`

**Parameters:**
- `q` — query tensor (B, S, D)
- `k` — key tensor (B, S, D)

**Returns:** tuple of rotated (q, k) with same shapes

**Constraints:**
- Split into even/odd pairs, apply rotation with `angles = pos * 1/(10000^(2i/D))`
- Must preserve vector norms
- Dot products should depend only on relative position""",
    'description_zh': """实现旋转位置编码（RoPE）。

RoPE 通过成对旋转查询和键向量来编码位置信息，利用点积性质实现相对位置感知。

**签名:** `apply_rope(q, k) -> (Tensor, Tensor)`

**参数:**
- `q` — 查询张量 (B, S, D)
- `k` — 键张量 (B, S, D)

**返回:** 旋转后的 (q, k) 元组，形状不变

**约束:**
- 按奇偶对分割，使用 `angles = pos * 1/(10000^(2i/D))` 旋转
- 必须保持向量范数不变
- 点积应仅依赖于相对位置""",
    "tests": [
        {
            "name": 'Output shapes',
            "code": """import torch
q = torch.randn(2, 8, 64)
k = torch.randn(2, 8, 64)
q_rot, k_rot = {fn}(q, k)
assert q_rot.shape == q.shape, f'Q shape: {q_rot.shape}'
assert k_rot.shape == k.shape, f'K shape: {k_rot.shape}'""",
        },
        {
            "name": 'Preserves norm',
            "code": """import torch
torch.manual_seed(0)
q = torch.randn(1, 16, 32)
k = torch.randn(1, 16, 32)
q_rot, k_rot = {fn}(q, k)
assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4), 'RoPE should preserve norms'""",
        },
        {
            "name": 'Relative position property',
            "code": """import torch
torch.manual_seed(0)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q_rot, k_rot = {fn}(q, k)
q2 = torch.cat([torch.zeros(1, 3, 16), q], dim=1)
k2 = torch.cat([torch.zeros(1, 3, 16), k], dim=1)
q2_rot, k2_rot = {fn}(q2, k2)
# Same relative distance of 2: pos 0 vs pos 2, and pos 3 vs pos 5
dot1 = (q_rot[:, 0] * k_rot[:, 2]).sum(dim=-1)
dot2 = (q2_rot[:, 3] * k2_rot[:, 5]).sum(dim=-1)
assert torch.allclose(dot1, dot2, atol=1e-4), 'Dot product should depend on relative position only'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
q = torch.randn(1, 4, 8, requires_grad=True)
k = torch.randn(1, 4, 8, requires_grad=True)
qr, kr = {fn}(q, k)
(qr.sum() + kr.sum()).backward()
assert q.grad is not None and k.grad is not None, 'Missing gradients'""",
        },
        {
            "name": 'Numerical correctness',
            "code": """import torch
torch.manual_seed(0)
B, S, D = 1, 2, 4
q = torch.randn(B, S, D)
k = torch.randn(B, S, D)
q_rot, k_rot = {fn}(q, k)
# Reference: for each position p and pair i, angle = p / 10000^(2i/D)
pos = torch.arange(S).float()
freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (S, D//2)
cos_a = torch.cos(angles)
sin_a = torch.sin(angles)
def rotate_ref(x):
    x_e = x[..., 0::2]
    x_o = x[..., 1::2]
    r_e = x_e * cos_a - x_o * sin_a
    r_o = x_e * sin_a + x_o * cos_a
    return torch.stack([r_e, r_o], dim=-1).flatten(-2)
expected_q = rotate_ref(q)
expected_k = rotate_ref(k)
assert torch.allclose(q_rot, expected_q, atol=1e-5), f'Q rotation mismatch: max diff {(q_rot - expected_q).abs().max()}'
assert torch.allclose(k_rot, expected_k, atol=1e-5), f'K rotation mismatch: max diff {(k_rot - expected_k).abs().max()}'""",
        },
    ],
    'solution': '''def apply_rope(q, k):
    """实现旋转位置编码（RoPE）（Rotary Position Embedding (RoPE)）。"""
    # Step 1: 为每个位置和偶数维通道生成几何递减的旋转角
    B, S, D = q.shape
    positions = torch.arange(S, device=q.device).unsqueeze(1).float()
    pair_dims = torch.arange(0, D, 2, device=q.device).float()
    inv_freqs = 1.0 / 10000.0 ** (pair_dims / D)
    angles = positions * inv_freqs
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Step 2: 将每对奇偶通道视为二维向量并施加同位置正交旋转
    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos_angles - odd * sin_angles
        rotated_odd = even * sin_angles + odd * cos_angles
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    # Step 3: 对查询和键复用同一组角度以保留相对位置点积结构
    return (rotate(q), rotate(k))''',
    'demo': """torch.manual_seed(42)
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
q_rot, k_rot = apply_rope(q, k)
print("Rotated shapes:", q_rot.shape, k_rot.shape)
print("Maximum Q norm error:", (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item())
print("Position-0 unchanged:", torch.allclose(q[:, 0], q_rot[:, 0], atol=1e-6))""",
    'explanation': r"""### 核心思路
RoPE 把最后一维相邻的 $(x_{2i},x_{2i+1})$ 看作二维向量，在位置 $p$ 处旋转 $\theta_{p,i}=p\cdot10000^{-2i/D}$。旋转矩阵 $R(\theta)$ 满足 $R(\theta)^\top R(\theta)=I$，所以范数不变；同时 $(R_pq)^\top(R_sk)=q^\top R_{s-p}k$，查询与键的点积只显式依赖相对位移 $s-p$。

### 代码步骤拆解
- **Step 1（构造多尺度角频率）**：由 `q.shape` 得到 $B,S,D$，`positions` 为 $(S,1)$，`inv_freqs` 为 $(D/2,)$，相乘生成每个位置、每个通道对的 `angles`。
- **Step 2（成对旋转奇偶通道）**：`even` 与 `odd` 均为 $(B,S,D/2)$；用 $\left(x_e\cos\theta-x_o\sin\theta,\ x_e\sin\theta+x_o\cos\theta\right)$ 完成二维旋转。
- **Step 3（交错还原查询与键）**：`stack` 把每对旋转结果放回相邻槽位，`flatten(-2)` 恢复 $(B,S,D)$，且 `q`、`k` 使用完全相同的角度表。

### 重点与易错点分析
- **注意点 1（频率指数）**：`pair_dims` 必须取 $0,2,\ldots,D-2$，指数是 `pair_dims / D`；若误用 $i/D$，各通道旋转频率会偏慢一倍。
- **注意点 2（二维旋转方向）**：第二个分量必须是 $x_e\sin\theta+x_o\cos\theta$。符号写反虽仍可能保持范数，却会破坏与标准 RoPE 一致的相对位置相位。
- **注意点 3（维度与缓存）**：$D$ 必须为偶数，否则奇偶切片长度不同；实际长序列模型通常缓存 $(S,D/2)$ 的正余弦表，避免每层重复计算。

### 复杂度分析
- **时间复杂度**：$O(BSD)$，查询和键的逐元素成对旋转占主导。
- **空间复杂度**：除两个 $(B,S,D)$ 输出外，角度与正余弦表需要 $O(SD)$ 临时空间。""",
}

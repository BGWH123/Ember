"""YaRN RoPE (Frequency Ramp & Temperature Scaling) task."""

TASK = {
    'title': 'YaRN RoPE (Frequency Ramp & Temperature Scaling)',
    'title_zh': 'YaRN 旋转位置编码',
    'difficulty': 'Hard',
    'category': '位置编码与嵌入',
    'function_name': 'yarn_rope',
    'starter_code': """def yarn_rope(q, k, scale, orig_max_pos=4096, alpha=1.0, beta=32.0, base=10000.0):
    pass""",
    'hint': """Build pair frequencies, blend interpolated and original frequencies with the clamped wavelength ramp, then rotate even/odd features.""",
    'hint_zh': '构造每对维度的频率，用截断的波长斜坡混合插值频率和原频率，再旋转偶数/奇数特征。',
    'description_en': """Implement YaRN RoPE with a frequency-dependent interpolation ramp and attention-temperature magnitude scaling.""",
    'description_zh': '实现带频率插值斜坡和注意力温度幅值缩放的 YaRN 旋转位置编码。',
    'theory_en': """YaRN interpolates low-wavelength frequencies while retaining frequencies that already fit many wavelengths in the training context. A logarithmic magnitude factor adjusts attention temperature for extended contexts.""",
    'theory_zh': 'YaRN 对低波长数频率进行插值，同时保留在训练上下文中已有足够波长数的频率；对数幅值因子用于调整长上下文注意力温度。',
    'diagram_en': """```text
base frequencies -> wavelength ramp -> blended frequencies -> pairwise rotation -> mscale
```""",
    'diagram_zh': """```text
基础频率 -> 波长斜坡 -> 混合频率 -> 成对旋转 -> mscale
```""",
    "tests": [
        {
            "name": 'output shapes',
            "code": """import torch
q = torch.randn(2, 9, 8)
k = torch.randn(2, 9, 8)
qr, kr = {fn}(q, k, 4.0)
assert qr.shape == q.shape and kr.shape == k.shape""",
        },
        {
            "name": 'exact formula',
            "code": """import torch, math
torch.manual_seed(0)
q, k = torch.randn(1, 5, 6), torch.randn(1, 5, 6)
scale, orig, alpha, beta, base = 3.0, 32, 1.0, 8.0, 100.0
qr, kr = {fn}(q, k, scale, orig, alpha, beta, base)
i = torch.arange(3, dtype=q.dtype)
theta = 1.0 / (base ** (2.0 * i / 6))
gamma = torch.clamp((orig * theta / (2.0 * math.pi) - alpha) / (beta - alpha), 0.0, 1.0)
freq = (1.0 - gamma) * theta / scale + gamma * theta
ang = torch.arange(5, dtype=q.dtype)[:, None] * freq[None, :]
c, s = torch.cos(ang), torch.sin(ang)
ref = torch.stack((q[..., 0::2] * c - q[..., 1::2] * s, q[..., 0::2] * s + q[..., 1::2] * c), -1).flatten(-2)
ref = ref * (0.1 * math.log(scale) + 1.0)
assert torch.allclose(qr, ref, atol=1e-6)""",
        },
        {
            "name": 'scale one edge case',
            "code": """import torch
q = torch.randn(1, 4, 8)
k = torch.randn(1, 4, 8)
qr, _ = {fn}(q, k, 1.0)
i = torch.arange(4, dtype=q.dtype)
a = torch.arange(4, dtype=q.dtype)[:, None] * (1.0 / (10000.0 ** (2.0 * i / 8)))[None]
ref = torch.stack((q[..., 0::2] * a.cos() - q[..., 1::2] * a.sin(), q[..., 0::2] * a.sin() + q[..., 1::2] * a.cos()), -1).flatten(-2)
assert torch.allclose(qr, ref, atol=1e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
q = torch.randn(2, 5, 8, requires_grad=True)
k = torch.randn(2, 5, 8, requires_grad=True)
qr, kr = {fn}(q, k, 2.0)
(qr.square().mean() + kr.square().mean()).backward()
assert q.grad is not None and k.grad is not None""",
        },
    ],
    'solution': '''def yarn_rope(q, k, scale, orig_max_pos=4096, alpha=1.0, beta=32.0, base=10000.0):
    """实现YaRN 旋转位置编码（YaRN RoPE (Frequency Ramp & Temperature Scaling)）。"""
    # Step 1: 按训练窗口内波长数构造 ramp，混合原始与 1/scale 插值频率
    D = q.size(-1)
    i = torch.arange(D // 2, device=q.device, dtype=q.dtype)
    theta = 1.0 / base ** (2.0 * i / D)
    wavelengths = orig_max_pos * theta / (2.0 * math.pi)
    gamma = torch.clamp((wavelengths - alpha) / (beta - alpha), 0.0, 1.0)
    theta_yarn = (1.0 - gamma) * (theta / scale) + gamma * theta
    # Step 2: 将序列位置与每对通道频率外积，并计算长上下文 attention 温度尺度
    pos = torch.arange(q.size(1), device=q.device, dtype=q.dtype)
    angles = pos[:, None] * theta_yarn[None, :]
    cos_a, sin_a = (torch.cos(angles), torch.sin(angles))
    mscale = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0

    def rotate(x):
        even, odd = (x[..., 0::2], x[..., 1::2])
        return torch.stack((even * cos_a - odd * sin_a, even * sin_a + odd * cos_a), dim=-1).flatten(-2) * mscale
    # Step 3: 对 q、k 的偶奇通道执行同角二维旋转，再乘 mscale 保持注意力幅值
    return (rotate(q), rotate(k))''',
    'demo': """torch.manual_seed(42)
q = torch.randn(1, 8192, 64)
k = torch.randn(1, 8192, 64)
q_rot, k_rot = yarn_rope(q, k, scale=2.0)
print("Rotated shapes:", q_rot.shape, k_rot.shape)
print("Sample values:", q_rot[0, 0, :4])""",
    'explanation': r"""### 核心思路
基础 RoPE 每对通道频率为 $\omega_i=\text{base}^{-2i/D}$。YaRN 根据训练窗口内周期数 $r_i=L_0\omega_i/(2\pi)$ 构造 $\gamma_i=\operatorname{clip}((r_i-\alpha)/(\beta-\alpha),0,1)$，并混合插值频率与原频率：$\omega_i'=(1-\gamma_i)\omega_i/s+\gamma_i\omega_i$。位置 $p$ 再以角度 $p\omega_i'$ 旋转 Q/K。

### 代码步骤拆解
- **Step 1（频率相关插值）**：从 `D/2` 个偶奇维对生成 `theta`，用 `orig_max_pos` 下的周期数计算 ramp；低 `gamma` 频率按 `scale` 压缩，高 `gamma` 频率保留。
- **Step 2（角度与温度尺度）**：位置 `(S,)` 和频率 `(D/2,)` 外积成 `angles`；长上下文取 `mscale=1+0.1*log(scale)`。
- **Step 3（成对复旋转）**：每对 `(even,odd)` 映射为 $(x_e\cos a-x_o\sin a,\ x_e\sin a+x_o\cos a)$，Q/K 使用同角度并恢复 `(B,S,D)`。

### 重点与易错点分析
- **注意点 1（ramp 方向）**：公式中的 `gamma=1` 保留原频率，`gamma=0` 使用插值频率；把两项权重写反会在错误频段压缩位置相位。
- **注意点 2（偶数头维与广播）**：最后一维必须为偶数才能配对；`angles(S,D/2)` 会跨 batch 广播，且 Q/K 的序列长度和通道维需一致。
- **注意点 3（旋转与幅值）**：纯 RoPE 是范数保持的正交旋转；这里 Q、K 都乘 `mscale`，其点积整体乘 $mscale^2$，等价于调节长上下文注意力温度。

### 复杂度分析
- **时间复杂度**：生成角度为 $O(SD)$，旋转批量 Q/K 为 $O(BSD)$。
- **空间复杂度**：`angles`、正弦和余弦各占 $O(SD)$，输出 Q/K 占 $O(BSD)$。""",
}

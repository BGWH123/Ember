"""YaRN rotary position encoding task."""

TASK = {
    "title": "YaRN RoPE (Frequency Ramp & Temperature Scaling)",
    "title_zh": "YaRN 旋转位置编码",
    "difficulty": "Hard",
    "category": "位置编码与嵌入",
    "function_name": "yarn_rope",
    "starter_code": '''def yarn_rope(q, k, scale, orig_max_pos=4096, alpha=1.0, beta=32.0, base=10000.0):
    pass
''',
    "hint": "Build pair frequencies, blend interpolated and original frequencies with the clamped wavelength ramp, then rotate even/odd features.",
    "hint_zh": "构造每对维度的频率，用截断的波长斜坡混合插值频率和原频率，再旋转偶数/奇数特征。",
    "description_en": "Implement YaRN RoPE with a frequency-dependent interpolation ramp and attention-temperature magnitude scaling.",
    "description_zh": "实现带频率插值斜坡和注意力温度幅值缩放的 YaRN 旋转位置编码。",
    "theory_en": "YaRN interpolates low-wavelength frequencies while retaining frequencies that already fit many wavelengths in the training context. A logarithmic magnitude factor adjusts attention temperature for extended contexts.",
    "theory_zh": "YaRN 对低波长数频率进行插值，同时保留在训练上下文中已有足够波长数的频率；对数幅值因子用于调整长上下文注意力温度。",
    "diagram_en": "```text\nbase frequencies -> wavelength ramp -> blended frequencies -> pairwise rotation -> mscale\n```",
    "diagram_zh": "```text\n基础频率 -> 波长斜坡 -> 混合频率 -> 成对旋转 -> mscale\n```",
    "tests": [
        {
            "name": "output shapes",
            "code": '''import torch
q = torch.randn(2, 9, 8)
k = torch.randn(2, 9, 8)
qr, kr = {fn}(q, k, 4.0)
assert qr.shape == q.shape and kr.shape == k.shape
''',
        },
        {
            "name": "exact formula",
            "code": '''import torch, math
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
assert torch.allclose(qr, ref, atol=1e-6)
''',
        },
        {
            "name": "scale one edge case",
            "code": '''import torch
q = torch.randn(1, 4, 8)
k = torch.randn(1, 4, 8)
qr, _ = {fn}(q, k, 1.0)
i = torch.arange(4, dtype=q.dtype)
a = torch.arange(4, dtype=q.dtype)[:, None] * (1.0 / (10000.0 ** (2.0 * i / 8)))[None]
ref = torch.stack((q[..., 0::2] * a.cos() - q[..., 1::2] * a.sin(), q[..., 0::2] * a.sin() + q[..., 1::2] * a.cos()), -1).flatten(-2)
assert torch.allclose(qr, ref, atol=1e-6)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
q = torch.randn(2, 5, 8, requires_grad=True)
k = torch.randn(2, 5, 8, requires_grad=True)
qr, kr = {fn}(q, k, 2.0)
(qr.square().mean() + kr.square().mean()).backward()
assert q.grad is not None and k.grad is not None
''',
        },
    ],
    "solution": '''import torch
import math

def yarn_rope(q, k, scale, orig_max_pos=4096, alpha=1.0, beta=32.0, base=10000.0):
    D = q.size(-1)
    i = torch.arange(D // 2, device=q.device, dtype=q.dtype)
    theta = 1.0 / (base ** (2.0 * i / D))
    wavelengths = orig_max_pos * theta / (2.0 * math.pi)
    gamma = torch.clamp((wavelengths - alpha) / (beta - alpha), 0.0, 1.0)
    theta_yarn = (1.0 - gamma) * (theta / scale) + gamma * theta
    pos = torch.arange(q.size(1), device=q.device, dtype=q.dtype)
    angles = pos[:, None] * theta_yarn[None, :]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    mscale = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0

    def rotate(x):
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos_a - odd * sin_a,
                            even * sin_a + odd * cos_a), dim=-1).flatten(-2) * mscale

    return rotate(q), rotate(k)
''',
    "demo": '''q = torch.randn(1, 8192, 64)
k = torch.randn(1, 8192, 64)
qr, kr = yarn_rope(q, k, scale=2.0)
print(qr.shape, kr.shape)
''',
    "explanation": "Each adjacent feature pair is treated as a 2D vector. The YaRN ramp selects between linearly interpolated and original angular frequencies, and `mscale` multiplies both rotated tensors.",
}

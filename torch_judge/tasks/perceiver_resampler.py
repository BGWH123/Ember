"""VLM pixel-unshuffle visual token compression task."""

TASK = {
    "title": "VLM Pixel Unshuffle Visual Token Compression",
    "title_zh": "VLM Pixel Unshuffle 视觉 Token 压缩",
    "difficulty": "Easy",
    "category": "多模态",
    "function_name": "pixel_unshuffle_tokens",
    "starter_code": '''def pixel_unshuffle_tokens(x, scale_factor=2, proj_weight=None):
    pass
''',
    "hint": "Split each spatial axis into a coarse coordinate and an r-sized local coordinate, move both local coordinates next to the channel axis, then flatten.",
    "hint_zh": "将每个空间轴拆为粗粒度坐标和大小为 r 的局部坐标，把两个局部坐标移到通道轴旁再展平。",
    "description_en": "Compress a channels-last visual patch grid into fewer, wider tokens using pixel unshuffle, with an optional linear projection.",
    "description_zh": "使用 pixel unshuffle 将通道后置的视觉 patch 网格压缩为数量更少、维度更宽的 token，并支持可选线性投影。",
    "theory_en": "An r by r neighborhood is rearranged into channels, reducing token count by r squared while preserving all values. A projection can then map the expanded feature dimension.",
    "theory_zh": "每个 r×r 邻域被重排到通道维，使 token 数量缩小 r² 倍且不丢失数值，随后可用投影映射扩展后的特征维。",
    "diagram_en": "```text\n(B,H,W,D) -> split H,W by r -> move local axes -> (B,H/r*W/r,r*r*D) -> optional @W\n```",
    "diagram_zh": "```text\n(B,H,W,D) -> 按 r 拆分 H,W -> 移动局部轴 -> (B,H/r*W/r,r*r*D) -> 可选 @W\n```",
    "tests": [
        {"name": "compressed shape", "code": '''import torch
x = torch.randn(2, 6, 4, 3)
out = {fn}(x, 2)
assert out.shape == (2, 6, 12)
'''},
        {"name": "exact ordering", "code": '''import torch
x = torch.arange(16.).reshape(1, 4, 4, 1)
out = {fn}(x, 2)
ref = x.reshape(1, 2, 2, 2, 2, 1).permute(0, 1, 3, 2, 4, 5).reshape(1, 4, 4)
assert torch.equal(out, ref)
'''},
        {"name": "scale one edge case", "code": '''import torch
x = torch.randn(1, 2, 3, 4)
out = {fn}(x, scale_factor=1)
assert out.shape == (1, 6, 4) and torch.equal(out, x.reshape(1, 6, 4))
'''},
        {"name": "projection gradient flow", "code": '''import torch
x = torch.randn(2, 4, 4, 3, requires_grad=True)
w = torch.randn(12, 5, requires_grad=True)
out = {fn}(x, 2, w)
out.sum().backward()
assert out.shape == (2, 4, 5) and x.grad is not None and w.grad is not None
'''},
    ],
    "solution": '''import torch

def pixel_unshuffle_tokens(x, scale_factor=2, proj_weight=None):
    B, H, W, D = x.shape
    r = scale_factor
    tokens = x.reshape(B, H // r, r, W // r, r, D)
    tokens = tokens.permute(0, 1, 3, 2, 4, 5)
    tokens = tokens.reshape(B, (H // r) * (W // r), r * r * D)
    if proj_weight is not None:
        tokens = tokens @ proj_weight
    return tokens
''',
    "demo": '''grid = torch.randn(2, 24, 24, 128)
print(pixel_unshuffle_tokens(grid, scale_factor=2).shape)
''',
    "explanation": "The permutation groups each coarse-grid location with its row-major r by r neighborhood. Reshape and permute are differentiable rearrangements.",
}

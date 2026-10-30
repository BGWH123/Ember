"""2D Axial Self-Attention task."""

TASK = {
    "title": "2D Axial Self-Attention",
    "title_zh": "二维轴向自注意力（Axial Attention）",
    "difficulty": "Medium",
    "category": "注意力机制",
    "function_name": "axial_attention",
    "starter_code": '''def axial_attention(x, W_q, W_k, W_v, axis="height"):
    # Fold the other spatial axis into the batch, attend, then restore BCHW.
    pass
''',
    "hint": "Permute BCHW to `(B*W,H,C)` for height or `(B*H,W,C)` for width, then use scaled dot-product attention.",
    "hint_zh": "高度注意力将 BCHW 排列为 `(B*W,H,C)`，宽度注意力排列为 `(B*H,W,C)`，再执行缩放点积注意力。",
    "description_en": "Implement axial self-attention on a BCHW tensor. Attention is computed along exactly one spatial axis while the other spatial axis is folded into the batch.",
    "description_zh": "在 BCHW 张量上实现轴向自注意力。仅沿一个空间轴计算注意力，另一个空间轴折叠进批次维。",
    "theory_en": "Axial attention replaces full HW-by-HW attention with separate one-dimensional attention. For one axis, project Q, K, V and compute `softmax(QK^T/sqrt(C))V`.",
    "theory_zh": "轴向注意力用一维注意力替代完整的 HW×HW 注意力。沿指定轴投影 Q、K、V，并计算 `softmax(QK^T/sqrt(C))V`。",
    "diagram_en": "```text\nB,C,H,W -> fold other axis -> batch,axis,C -> QK attention -> B,C,H,W\n```",
    "diagram_zh": "```text\nB,C,H,W -> 折叠另一空间轴 -> 批次,轴,C -> QK 注意力 -> B,C,H,W\n```",
    "tests": [
        {
            "name": "height shape",
            "code": '''import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 4, 5)
w = torch.randn(3, 3)
out = {fn}(x, w, w, w, "height")
assert out.shape == x.shape
''',
        },
        {
            "name": "width numerical reference",
            "code": '''import torch
torch.manual_seed(1)
x = torch.randn(2, 3, 2, 4)
wq, wk, wv = [torch.randn(3, 3) for _ in range(3)]
out = {fn}(x, wq, wk, wv, "width")
seq = x.permute(0, 2, 3, 1).reshape(4, 4, 3)
ref = torch.softmax((seq @ wq) @ (seq @ wk).transpose(-2, -1) / (3 ** 0.5), -1) @ (seq @ wv)
ref = ref.reshape(2, 2, 4, 3).permute(0, 3, 1, 2)
assert torch.allclose(out, ref, atol=1e-6)
''',
        },
        {
            "name": "singleton attended axis",
            "code": '''import torch
x = torch.randn(1, 2, 1, 3)
eye = torch.eye(2)
out = {fn}(x, eye, eye, eye, "height")
assert torch.allclose(out, x, atol=1e-6)
''',
        },
        {
            "name": "gradient flow",
            "code": '''import torch
x = torch.randn(1, 2, 3, 4, requires_grad=True)
wq = torch.randn(2, 2, requires_grad=True)
wk = torch.randn(2, 2, requires_grad=True)
wv = torch.randn(2, 2, requires_grad=True)
out = {fn}(x, wq, wk, wv, "height")
out.square().sum().backward()
assert all(t.grad is not None for t in (x, wq, wk, wv))
''',
        },
    ],
    "solution": '''import torch

def axial_attention(x, W_q, W_k, W_v, axis="height"):
    B, C, H, W = x.shape
    if axis == "height":
        seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
    elif axis == "width":
        seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
    else:
        raise ValueError("axis must be 'height' or 'width'")
    q, k, v = seq @ W_q, seq @ W_k, seq @ W_v
    out = torch.softmax((q @ k.transpose(-2, -1)) / (C ** 0.5), dim=-1) @ v
    if axis == "height":
        return out.reshape(B, W, H, C).permute(0, 3, 2, 1)
    return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
''',
    "demo": '''x = torch.randn(2, 8, 6, 5)
w = torch.randn(8, 8)
print(axial_attention(x, w, w, w, "height").shape)
''',
    "explanation": "The permutation makes each row or column an independent sequence. Attention is standard scaled dot-product attention over that sequence, followed by the inverse reshape and permutation.",
}

"""2D Axial Self-Attention task."""

TASK = {
    'title': '2D Axial Self-Attention',
    'title_zh': '二维轴向自注意力（Axial Attention）',
    'difficulty': 'Medium',
    'category': '注意力机制',
    'function_name': 'axial_attention',
    'starter_code': """def axial_attention(x, W_q, W_k, W_v, axis="height"):
    pass""",
    'hint': """Permute BCHW to `(B*W,H,C)` for height or `(B*H,W,C)` for width, then use scaled dot-product attention.""",
    'hint_zh': '高度注意力将 BCHW 排列为 `(B*W,H,C)`，宽度注意力排列为 `(B*H,W,C)`，再执行缩放点积注意力。',
    'description_en': """Implement axial self-attention on a BCHW tensor. Attention is computed along exactly one spatial axis while the other spatial axis is folded into the batch.""",
    'description_zh': '在 BCHW 张量上实现轴向自注意力。仅沿一个空间轴计算注意力，另一个空间轴折叠进批次维。',
    'theory_en': """Axial attention replaces full HW-by-HW attention with separate one-dimensional attention. For one axis, project Q, K, V and compute `softmax(QK^T/sqrt(C))V`.""",
    'theory_zh': '轴向注意力用一维注意力替代完整的 HW×HW 注意力。沿指定轴投影 Q、K、V，并计算 `softmax(QK^T/sqrt(C))V`。',
    'diagram_en': """```text
B,C,H,W -> fold other axis -> batch,axis,C -> QK attention -> B,C,H,W
```""",
    'diagram_zh': """```text
B,C,H,W -> 折叠另一空间轴 -> 批次,轴,C -> QK 注意力 -> B,C,H,W
```""",
    "tests": [
        {
            "name": 'height shape',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 4, 5)
w = torch.randn(3, 3)
out = {fn}(x, w, w, w, "height")
assert out.shape == x.shape""",
        },
        {
            "name": 'width numerical reference',
            "code": """import torch
torch.manual_seed(1)
x = torch.randn(2, 3, 2, 4)
wq, wk, wv = [torch.randn(3, 3) for _ in range(3)]
out = {fn}(x, wq, wk, wv, "width")
seq = x.permute(0, 2, 3, 1).reshape(4, 4, 3)
ref = torch.softmax((seq @ wq) @ (seq @ wk).transpose(-2, -1) / (3 ** 0.5), -1) @ (seq @ wv)
ref = ref.reshape(2, 2, 4, 3).permute(0, 3, 1, 2)
assert torch.allclose(out, ref, atol=1e-6)""",
        },
        {
            "name": 'singleton attended axis',
            "code": """import torch
x = torch.randn(1, 2, 1, 3)
eye = torch.eye(2)
out = {fn}(x, eye, eye, eye, "height")
assert torch.allclose(out, x, atol=1e-6)""",
        },
        {
            "name": 'gradient flow',
            "code": """import torch
x = torch.randn(1, 2, 3, 4, requires_grad=True)
wq = torch.randn(2, 2, requires_grad=True)
wk = torch.randn(2, 2, requires_grad=True)
wv = torch.randn(2, 2, requires_grad=True)
out = {fn}(x, wq, wk, wv, "height")
out.square().sum().backward()
assert all(t.grad is not None for t in (x, wq, wk, wv))""",
        },
    ],
    'solution': '''def axial_attention(x, W_q, W_k, W_v, axis="height"):
    """二维轴向自注意力 (Axial Attention)。"""
    B, C, H, W = x.shape

    # Step 1: 将非注意力空间轴折叠进批次维
    if axis == "height":
        sequence = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
    elif axis == "width":
        sequence = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
    else:
        raise ValueError("axis must be 'height' or 'width'")

    # Step 2: 沿目标轴执行缩放点积注意力
    q = sequence @ W_q
    k = sequence @ W_k
    v = sequence @ W_v
    scores = q @ k.transpose(-2, -1) / (C ** 0.5)
    context = torch.softmax(scores, dim=-1) @ v

    # Step 3: 恢复 BCHW 布局
    if axis == "height":
        return context.reshape(B, W, H, C).permute(0, 3, 2, 1)
    return context.reshape(B, H, W, C).permute(0, 3, 1, 2)''',
    'demo': """torch.manual_seed(42)
B, C, H, W = 2, 8, 6, 5
x = torch.randn(B, C, H, W)
weight = torch.randn(C, C)
out = axial_attention(x, weight, weight, weight, "height")
print("Output shape:", out.shape)
print("Sample values:", out[0, :3, 0, 0])""",
    'explanation': """### 核心思路
将二维特征的一条空间轴视为序列，另一条轴并入批次，从而用一维注意力替代完整的 $(HW)^2$ 注意力。

### 代码步骤拆解
- **Step 1（轴重排）**：高度模式得到 `(B*W, H, C)`，宽度模式得到 `(B*H, W, C)`。
- **Step 2（轴内注意力）**：投影 `q/k/v`，沿目标空间轴计算注意力。
- **Step 3（布局恢复）**：按模式逆向 reshape 和 permute，返回 `(B, C, H, W)`。

### 重点与易错点分析
- **注意点 1（轴语义）**：折叠的是另一空间轴，不能让不同列或行之间意外互相注意。
- **注意点 2（连续性）**：`permute` 后使用 `reshape`，避免直接 `view` 遇到非连续内存。
- **注意点 3（非法参数）**：对未知 `axis` 显式报错，防止静默执行错误布局。

### 复杂度分析
- **时间复杂度**：高度模式为 $O(BWH^2C)$，宽度模式为 $O(BHW^2C)$。
- **空间复杂度**：分别为 $O(BWH^2)$ 或 $O(BHW^2)$，主要是注意力矩阵。""",
}

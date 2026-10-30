"""VLM Pixel Unshuffle Visual Token Compression task."""

TASK = {
    'title': 'VLM Pixel Unshuffle Visual Token Compression',
    'title_zh': 'VLM Pixel Unshuffle 视觉 Token 压缩',
    'difficulty': 'Easy',
    'category': '多模态',
    'function_name': 'pixel_unshuffle_tokens',
    'starter_code': """def pixel_unshuffle_tokens(x, scale_factor=2, proj_weight=None):
    pass""",
    'hint': """Split each spatial axis into a coarse coordinate and an r-sized local coordinate, move both local coordinates next to the channel axis, then flatten.""",
    'hint_zh': '将每个空间轴拆为粗粒度坐标和大小为 r 的局部坐标，把两个局部坐标移到通道轴旁再展平。',
    'description_en': """Compress a channels-last visual patch grid into fewer, wider tokens using pixel unshuffle, with an optional linear projection.""",
    'description_zh': '使用 pixel unshuffle 将通道后置的视觉 patch 网格压缩为数量更少、维度更宽的 token，并支持可选线性投影。',
    'theory_en': """An r by r neighborhood is rearranged into channels, reducing token count by r squared while preserving all values. A projection can then map the expanded feature dimension.""",
    'theory_zh': '每个 r×r 邻域被重排到通道维，使 token 数量缩小 r² 倍且不丢失数值，随后可用投影映射扩展后的特征维。',
    'diagram_en': """```text
(B,H,W,D) -> split H,W by r -> move local axes -> (B,H/r*W/r,r*r*D) -> optional @W
```""",
    'diagram_zh': """```text
(B,H,W,D) -> 按 r 拆分 H,W -> 移动局部轴 -> (B,H/r*W/r,r*r*D) -> 可选 @W
```""",
    "tests": [
        {
            "name": 'compressed shape',
            "code": """import torch
x = torch.randn(2, 6, 4, 3)
out = {fn}(x, 2)
assert out.shape == (2, 6, 12)""",
        },
        {
            "name": 'exact ordering',
            "code": """import torch
x = torch.arange(16.).reshape(1, 4, 4, 1)
out = {fn}(x, 2)
ref = x.reshape(1, 2, 2, 2, 2, 1).permute(0, 1, 3, 2, 4, 5).reshape(1, 4, 4)
assert torch.equal(out, ref)""",
        },
        {
            "name": 'scale one edge case',
            "code": """import torch
x = torch.randn(1, 2, 3, 4)
out = {fn}(x, scale_factor=1)
assert out.shape == (1, 6, 4) and torch.equal(out, x.reshape(1, 6, 4))""",
        },
        {
            "name": 'projection gradient flow',
            "code": """import torch
x = torch.randn(2, 4, 4, 3, requires_grad=True)
w = torch.randn(12, 5, requires_grad=True)
out = {fn}(x, 2, w)
out.sum().backward()
assert out.shape == (2, 4, 5) and x.grad is not None and w.grad is not None""",
        },
    ],
    'solution': '''def pixel_unshuffle_tokens(x, scale_factor=2, proj_weight=None):
    """VLM Pixel Unshuffle 视觉 Token 压缩 (VLM Pixel Unshuffle Visual Token Compression)。"""
    # Step 1: 将高宽分别拆成粗网格坐标与 r×r 邻域坐标
    B, H, W, D = x.shape
    scale = scale_factor
    tokens = x.reshape(B, H // scale, scale, W // scale, scale, D)
    # Step 2: 把两个邻域轴移到通道旁并展平为 r²D 维视觉 token
    tokens = tokens.permute(0, 1, 3, 2, 4, 5)
    tokens = tokens.reshape(B, H // scale * (W // scale), scale * scale * D)
    # Step 3: 可选地将扩展后的 token 特征投影到目标模型维度
    if proj_weight is not None:
        tokens = tokens @ proj_weight
    return tokens''',
    'demo': """torch.manual_seed(42)
grid = torch.arange(16.0).reshape(1, 4, 4, 1)
proj_weight = torch.randn(4, 3)
tokens = pixel_unshuffle_tokens(grid, scale_factor=2)
projected = pixel_unshuffle_tokens(grid, scale_factor=2, proj_weight=proj_weight)
print('Compressed shape / projected shape:', tokens.shape, projected.shape)
print('First 2x2 neighborhood token:', tokens[0, 0])
print('Value count preserved:', tokens.numel() == grid.numel())""",
    'explanation': r"""### 核心思路
Pixel unshuffle 是无损重排：把每个 $r\times r$ 空间邻域的值搬到特征维，使 $(B,H,W,D)$ 变成 $(B,HW/r^2,r^2D)$。token 数减少 $r^2$ 倍而总元素数不变；若提供 `proj_weight`，再用矩阵乘法把宽 token 映射到语言模型所需维度。

### 代码步骤拆解
- **Step 1（拆分空间坐标）**：`reshape(B,H//r,r,W//r,r,D)` 将行列坐标各表示为“粗网格索引 + 邻域内偏移”，要求 `H`、`W` 都能被 `scale` 整除。
- **Step 2（邻域并入通道）**：`permute(0,1,3,2,4,5)` 排成 `(B,H/r,W/r,r,r,D)`，再展平后三个轴，得到按粗网格行优先排列的 `(B,HW/r^2,r^2D)`。
- **Step 3（可选特征投影）**：当 `proj_weight` 形状为 `(r^2D,D_out)` 时，`tokens @ proj_weight` 输出 `(B,HW/r^2,D_out)`；不提供权重则返回原始无损重排。

### 重点与易错点分析
- **注意点 1（排列次序）**：两个局部偏移轴必须同时移到 `D` 前；错误的 `permute` 虽有相同形状，却会打乱每个 token 内像素与通道的固定对应关系。
- **注意点 2（整除约束）**：实现没有裁剪或填充，`H % r` 或 `W % r` 非零时无法合法 reshape；模型预处理应事先把视觉网格调整到可整除尺寸。
- **注意点 3（与池化的区别）**：unshuffle 不做平均或丢弃信息，只交换轴；压缩 token 数后通道增宽，因此无投影时元素量和激活存储量仍是 $BHWD$。

### 复杂度分析
- **时间复杂度**：重排涉及 $O(BHWD)$ 个元素；若投影到 $D_{out}$，矩阵乘法额外为 $O(BHW\,r^0D D_{out})=O(BHWDD_{out})$。
- **空间复杂度**：输出 token 为 $O(BHWD)$；若 `permute` 后的 reshape 触发连续化，还会产生同量级临时副本，投影输出另占 $O(BHW D_{out}/r^2)$。""",
}

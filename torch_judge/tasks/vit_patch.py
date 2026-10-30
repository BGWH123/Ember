"""ViT Patch Embedding task."""

TASK = {
    'title': 'ViT Patch Embedding',
    'title_zh': 'ViT Patch Embedding',
    'difficulty': 'Medium',
    'category': 'Transformer组件',
    'function_name': 'PatchEmbedding',
    'starter_code': """class PatchEmbedding(nn.Module):
    pass""",
    'hint': """`(B,C,H,W)` → reshape to `(B, n_h, P, n_w, P, C)` → permute → `(B, num_patches, C*P*P)` → `nn.Linear(C*P*P, embed_dim)`.""",
    'hint_zh': """`(B,C,H,W)` → reshape 为 `(B, n_h, P, n_w, P, C)` → permute → `(B, num_patches, C*P*P)` → `nn.Linear(C*P*P, embed_dim)`。""",
    'description_en': """Implement ViT patch embedding as an nn.Module.

Patch embedding splits an image into fixed-size patches and projects each patch into an embedding vector, forming the input sequence for Vision Transformers.

**Signature:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — image tensor (B, C, H, W)

**Returns:** patch embeddings (B, num_patches, embed_dim)

**Constraints:**
- `num_patches = (img_size / patch_size)^2`
- Store `self.num_patches` as an attribute
- Project with `nn.Linear(C * P * P, embed_dim)`""",
    'description_zh': """实现 ViT 图像块嵌入（nn.Module）。

图像块嵌入将图像分割为固定大小的块，并将每个块投影为嵌入向量，形成 Vision Transformer 的输入序列。

**签名:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 图像张量 (B, C, H, W)

**返回:** 图像块嵌入 (B, num_patches, embed_dim)

**约束:**
- `num_patches = (img_size / patch_size)^2`
- 将 `self.num_patches` 存储为属性
- 使用 `nn.Linear(C * P * P, embed_dim)` 投影""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, torch.nn as nn
pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
assert isinstance(pe, nn.Module)
out = pe(torch.randn(2, 3, 32, 32))
assert out.shape == (2, 16, 64), f'Shape: {out.shape}, expected (2, 16, 64)'""",
        },
        {
            "name": 'num_patches attribute',
            "code": """import torch
pe = {fn}(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
assert pe.num_patches == 196, f'num_patches: {pe.num_patches}'""",
        },
        {
            "name": 'Different image sizes',
            "code": """import torch
pe = {fn}(img_size=64, patch_size=16, in_channels=1, embed_dim=32)
out = pe(torch.randn(1, 1, 64, 64))
assert out.shape == (1, 16, 32), f'Shape: {out.shape}'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
x = torch.randn(1, 3, 32, 32, requires_grad=True)
pe(x).sum().backward()
assert x.grad is not None, 'x.grad is None'""",
        },
        {
            "name": 'Patch values match manual projection',
            "code": """import torch, torch.nn as nn
torch.manual_seed(7)
img_size, patch_size, in_channels, embed_dim = 4, 2, 1, 4
pe = {fn}(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
x = torch.zeros(1, 1, 4, 4)
x[0, 0, 0:2, 0:2] = 1.0
x[0, 0, 0:2, 2:4] = 2.0
x[0, 0, 2:4, 0:2] = 3.0
x[0, 0, 2:4, 2:4] = 4.0
out = pe(x)
assert out.shape == (1, 4, embed_dim), f'Shape: {out.shape}'
# Manually compute expected for each patch
proj = pe.proj
patch0 = x[0, 0, 0:2, 0:2].reshape(1, -1)  # all ones
patch1 = x[0, 0, 0:2, 2:4].reshape(1, -1)  # all twos
patch2 = x[0, 0, 2:4, 0:2].reshape(1, -1)  # all threes
patch3 = x[0, 0, 2:4, 2:4].reshape(1, -1)  # all fours
expected = torch.cat([patch0, patch1, patch2, patch3], dim=0) @ proj.weight.T + proj.bias
assert torch.allclose(out[0], expected, atol=1e-5), f'Patch projection mismatch'""",
        },
    ],
    'solution': '''class PatchEmbedding(nn.Module):
    """实现ViT Patch Embedding（ViT Patch Embedding）。"""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        # Step 1: 记录 P×P 网格规模，并建立从展平 RGB patch 到 token 的线性投影
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        # Step 2: 将 (B,C,H,W) 重排为按行扫描的 N 个连续 C×P×P patch 向量
        B, C, H, W = x.shape
        patch_size = self.patch_size
        patch_rows, patch_cols = (H // patch_size, W // patch_size)
        x = x.reshape(B, C, patch_rows, patch_size, patch_cols, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, patch_rows * patch_cols, C * patch_size ** 2)
        # Step 3: 对每个 patch 独立投影，形成 Transformer 所需的 (B,N,D) 序列
        return self.proj(x)''',
    'demo': """torch.manual_seed(42)
pe = PatchEmbedding(224, 16, 3, 768)
x = torch.randn(1, 3, 224, 224)
print('Output:', pe(x).shape)
print('Patches:', pe.num_patches)""",
    'explanation': r"""### 核心思路
把 `(B,C,H,W)` 图像无重叠地划分为 $P\times P$ 小块，每块展平成长度 $CP^2$ 的向量，再共享同一线性映射投影到 $D$ 维。token 数为 $N=(H/P)(W/P)$，因此输出是 `(B,N,D)`，与卷积核/步幅都为 $P$ 的卷积投影等价。

### 代码步骤拆解
- **Step 1（定义 patch 投影）**：`num_patches=(img_size//patch_size)^2`，`self.proj` 的输入维是 `in_channels*P*P`、输出维是 `embed_dim`。
- **Step 2（空间网格重排）**：先 reshape 为 `(B,C,H/P,P,W/P,P)`，再 permute 成 `(B,H/P,W/P,C,P,P)`，最后合并网格和 patch 内维。
- **Step 3（生成视觉 token）**：`self.proj` 并行处理 `(B,N,CP^2)` 中的每块，得到 `(B,N,D)`，token 顺序按 patch 行优先扫描。

### 重点与易错点分析
- **注意点 1（整除条件）**：`H`、`W` 必须被 `P` 整除；当前整数除法不会自动报错，尺寸不整除时后续 reshape 才失败，不能把余边静默当 patch。
- **注意点 2（内存布局）**：reshape 后必须 permute，才能把同一空间块的所有通道和像素放进连续特征维；只 reshape 会跨 patch 混合元素。
- **注意点 3（与卷积实现对比）**：`Linear(CP^2,D)` 权重可重排为 `(D,C,P,P)` 的 Conv2d 核；两者参数和乘加量相同，但输出布局分别是序列和特征图。

### 复杂度分析
- **时间复杂度**：投影 $BN$ 个长度 $CP^2$ 的向量到 $D$ 维，耗时 $O(BNCP^2D)=O(BHWC D)$。
- **空间复杂度**：输出占 $O(BND)$；patch 展平视图在重排不连续时可能物化 $O(BHWC)$ 的中间张量。""",
}

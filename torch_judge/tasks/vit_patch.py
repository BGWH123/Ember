"""ViT Patch Embedding task."""

TASK = {
    "title": "ViT Patch Embedding",
    "title_zh": "ViT Patch Embedding",
    "difficulty": "Medium",
    "category": "Transformer组件",
    "description_en": (
        "Implement ViT patch embedding as an nn.Module.\n\nPatch embedding splits an image into fixed-size patches and projects each patch into an embedding vector, forming the input sequence for Vision Transformers.\n\n**Signature:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)` (nn.Module)\n\n**Forward:** `forward(x) -> Tensor`\n- `x` — image tensor (B, C, H, W)\n\n**Returns:** patch embeddings (B, num_patches, embed_dim)\n\n**Constraints:**\n- `num_patches = (img_size / patch_size)^2`\n- Store `self.num_patches` as an attribute\n- Project with `nn.Linear(C * P * P, embed_dim)`"
    ),
    "description_zh": (
        "实现 ViT 图像块嵌入（nn.Module）。\n\n图像块嵌入将图像分割为固定大小的块，并将每个块投影为嵌入向量，形成 Vision Transformer 的输入序列。\n\n**签名:** `PatchEmbedding(img_size, patch_size, in_channels, embed_dim)`（nn.Module）\n\n**前向传播:** `forward(x) -> Tensor`\n- `x` — 图像张量 (B, C, H, W)\n\n**返回:** 图像块嵌入 (B, num_patches, embed_dim)\n\n**约束:**\n- `num_patches = (img_size / patch_size)^2`\n- 将 `self.num_patches` 存储为属性\n- 使用 `nn.Linear(C * P * P, embed_dim)` 投影"
    ),
    "function_name": "PatchEmbedding",
    "hint": "`(B,C,H,W)` → reshape to `(B, n_h, P, n_w, P, C)` → permute → `(B, num_patches, C*P*P)` → `nn.Linear(C*P*P, embed_dim)`.",
    "hint_zh": "`(B,C,H,W)` → reshape 为 `(B, n_h, P, n_w, P, C)` → permute → `(B, num_patches, C*P*P)` → `nn.Linear(C*P*P, embed_dim)`。",
    "tests": [
        {
            "name": "Output shape",
            "code": """









import torch, torch.nn as nn
pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
assert isinstance(pe, nn.Module)
out = pe(torch.randn(2, 3, 32, 32))
assert out.shape == (2, 16, 64), f'Shape: {out.shape}, expected (2, 16, 64)'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "num_patches attribute",
            "code": """









import torch
pe = {fn}(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
assert pe.num_patches == 196, f'num_patches: {pe.num_patches}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Different image sizes",
            "code": """









import torch
pe = {fn}(img_size=64, patch_size=16, in_channels=1, embed_dim=32)
out = pe(torch.randn(1, 1, 64, 64))
assert out.shape == (1, 16, 32), f'Shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
pe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
x = torch.randn(1, 3, 32, 32, requires_grad=True)
pe(x).sum().backward()
assert x.grad is not None, 'x.grad is None'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Patch values match manual projection",
            "code": """









import torch, torch.nn as nn
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
assert torch.allclose(out[0], expected, atol=1e-5), f'Patch projection mismatch'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








class PatchEmbedding(nn.Module):  # 继承 nn.Module，注册为可训练模块
    """
    ViT Patch Embedding module.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):  # 初始化: 定义模型结构和参数
        # Initialize layers and parameters
        super().__init__()  # 调用父类 nn.Module 初始化，注册所有子模块
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)  # Linear projection

    def forward(self, x):  # 前向传播: 定义数据流
        # Compute forward pass
        B, C, H, W = x.shape
        p = self.patch_size
        n_h, n_w = H // p, W // p
        x = x.reshape(B, C, n_h, p, n_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * p * p)
        return self.proj(x)
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








pe = PatchEmbedding(224, 16, 3, 768)
x = torch.randn(1, 3, 224, 224)
print('Output:', pe(x).shape)
print('Patches:', pe.num_patches)
    
    
    
    
    
    
    
    
    ''',
}

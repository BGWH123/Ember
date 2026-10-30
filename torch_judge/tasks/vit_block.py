"""ViT Transformer Block task."""

TASK = {
    'title': 'ViT Transformer Block',
    'title_zh': 'ViT Transformer Block',
    'difficulty': 'Hard',
    'category': 'Transformer组件',
    'function_name': 'ViTBlock',
    'starter_code': """class ViTBlock(nn.Module):
    pass""",
    'hint': """MHA: `nn.Linear(d, 3d)` → split Q/K/V → reshape `(B, H, N, d_h)` → scaled dot-product → `proj`
GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`
Block: `x = x + MHA(norm1(x))` → `x = x + MLP(norm2(x))`""",
    'hint_zh': """MHA：`nn.Linear(d, 3d)` → 拆分 Q/K/V → reshape `(B, H, N, d_h)` → 缩放点积 → `proj`
GELU：`x * 0.5 * (1 + erf(x / sqrt(2)))`
块结构：`x = x + MHA(norm1(x))` → `x = x + MLP(norm2(x))`""",
    'description_en': """Implement a Vision Transformer (ViT) block as an nn.Module.

A ViT block uses pre-norm architecture with multi-head self-attention and an MLP, both wrapped in residual connections.

**Signature:** `ViTBlock(d_model, num_heads)` (nn.Module)

**Forward:** `forward(x) -> Tensor`
- `x` — input tensor (B, N, d_model), where N = num_patches + 1 (includes CLS token)

**Returns:** output tensor (B, N, d_model)

**Architecture:**
```
x = x + MHA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Constraints:**
- `nn.Linear` and `nn.LayerNorm` are allowed as building blocks
- MHA must be hand-implemented (no `nn.MultiheadAttention`)
- MLP: `Linear(d, 4d) -> GELU -> Linear(4d, d)`
- GELU must be hand-implemented: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- No `F.*` or `nn.functional.*` calls anywhere""",
    'description_zh': """将 Vision Transformer（ViT）块实现为 nn.Module。

ViT 块使用 pre-norm 架构，包含多头自注意力和 MLP，两者都有残差连接。

**签名:** `ViTBlock(d_model, num_heads)`（nn.Module）

**前向传播:** `forward(x) -> Tensor`
- `x` — 输入张量 (B, N, d_model)，其中 N = 图像块数 + 1（含 CLS token）

**返回:** 输出张量 (B, N, d_model)

**架构:**
```
x = x + MHA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**约束:**
- `nn.Linear` 和 `nn.LayerNorm` 可作为构建块使用
- MHA 必须手动实现（不得使用 `nn.MultiheadAttention`）
- MLP：`Linear(d, 4d) -> GELU -> Linear(4d, d)`
- GELU 必须手动实现：`x * 0.5 * (1 + erf(x / sqrt(2)))`
- 任何地方都不得调用 `F.*` 或 `nn.functional.*`""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'ViTBlock must inherit from nn.Module'
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'""",
        },
        {
            "name": 'Residual connections active',
            "code": """import torch
torch.manual_seed(1)
block = {fn}(d_model=32, num_heads=4)
x1 = torch.randn(1, 5, 32)
x2 = x1 + 0.5
out1 = block(x1)
out2 = block(x2)
assert not torch.allclose(out1, out2, atol=1e-4), 'Output unchanged when input changed — residuals may be broken'""",
        },
        {
            "name": 'Pre-norm: has norm1 and norm2 as LayerNorm',
            "code": """import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'norm1') and isinstance(block.norm1, nn.LayerNorm), 'Need self.norm1 = nn.LayerNorm'
assert hasattr(block, 'norm2') and isinstance(block.norm2, nn.LayerNorm), 'Need self.norm2 = nn.LayerNorm'""",
        },
        {
            "name": 'Gradient flows to all parameters',
            "code": """import torch
torch.manual_seed(2)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 6, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params received gradients'""",
        },
        {
            "name": 'Works with different num_heads',
            "code": """import torch
torch.manual_seed(3)
for num_heads in [1, 2, 8]:
    block = {fn}(d_model=64, num_heads=num_heads)
    x = torch.randn(2, 9, 64)
    out = block(x)
    assert out.shape == (2, 9, 64), f'Shape mismatch with num_heads={num_heads}: {out.shape}'""",
        },
        {
            "name": 'Numerical correctness via manual forward pass',
            "code": """import torch, torch.nn as nn, math
torch.manual_seed(0)
B, S, d_model, num_heads = 1, 4, 16, 2
block = {fn}(d_model=d_model, num_heads=num_heads)
block.eval()
x = torch.randn(B, S, d_model)

# --- manual MHA ---
d_h = d_model // num_heads
normed1 = block.norm1(x)
qkv = block.qkv(normed1).reshape(B, S, 3, num_heads, d_h).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
scale = d_h ** -0.5
attn_w = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
attn_out = (attn_w @ v).transpose(1, 2).reshape(B, S, d_model)
mha_out = block.proj(attn_out)
x_after_attn = x + mha_out

# --- manual MLP with hand-implemented GELU ---
normed2 = block.norm2(x_after_attn)
h = block.fc1(normed2)
gelu_h = h * 0.5 * (1.0 + torch.erf(h / math.sqrt(2.0)))
mlp_out = block.fc2(gelu_h)
expected = x_after_attn + mlp_out

out = block(x)
assert torch.allclose(out, expected, atol=1e-4), f'Max diff: {(out - expected).abs().max().item()}'""",
        },
    ],
    'solution': '''class ViTBlock(nn.Module):
    """实现ViT Transformer Block（ViT Transformer Block）。"""

    def __init__(self, d_model, num_heads):
        # Step 1: 建立联合 QKV 多头投影、双 pre-norm 与四倍通道 MLP
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _mha(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        scale = self.d_h ** (-0.5)
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def forward(self, x):
        # Step 2: 在归一化 patch 序列上计算缩放点积注意力并加入第一条残差
        x = x + self._mha(self.norm1(x))
        x = x + self.fc2(self._gelu(self.fc1(self.norm2(x))))
        # Step 3: 对第二次归一化结果执行精确 GELU-MLP 并加入第二条残差
        return x''',
    'demo': """torch.manual_seed(42)
batch, num_patches, d_model, num_heads = (2, 16, 64, 4)
block = ViTBlock(d_model, num_heads)
x = torch.randn(batch, num_patches, d_model)
out = block(x)
print('Input shape: ', x.shape)
print('Output shape:', out.shape)
assert out.shape == x.shape, 'Shape mismatch!'
print('Shape preserved: True')""",
    'explanation': r"""### 核心思路
ViT 将 CLS 与 patch token 作为长度 $N$ 的序列，块内执行 $X_1=X+\operatorname{MSA}(\operatorname{LN}_1X)$、$X_2=X_1+\operatorname{MLP}(\operatorname{LN}_2X_1)$。每头注意力为 $\operatorname{softmax}(QK^\top/\sqrt{d_h})V$，MLP 用 $D\to4D\to D$ 在通道维变换。

### 代码步骤拆解
- **Step 1（投影模块构造）**：`qkv` 一次输出 `3D`，`num_heads` 将其划成头维 `d_h=D/H`；两套 LayerNorm 分别服务于注意力和 MLP。
- **Step 2（全局 patch 注意力）**：`(B,N,3D)` 重排为 Q/K/V 的 `(B,H,N,d_h)`，形成 `(B,H,N,N)` 权重，聚合 V 后拼头并经 `proj` 加回输入。
- **Step 3（通道 MLP 残差）**：第二次 pre-norm 后经 `fc1`、erf 形式精确 GELU、`fc2`，结果与 `X_1` 相加，形状保持 `(B,N,D)`。

### 重点与易错点分析
- **注意点 1（头维缩放）**：缩放必须是 $1/\sqrt{d_h}$；使用 $1/\sqrt D$ 会让多头数变化时注意力温度错误，且 `D` 必须整除 `H`。
- **注意点 2（重排顺序）**：QKV 需先按 `(B,N,3,H,d_h)` 拆分再移到 `(3,B,H,N,d_h)`；错误地直接 view 容易把 token、头和 QKV 元素交错。
- **注意点 3（视觉 token 无因果性）**：图像 patch 与 CLS 之间应全连接注意，不使用 causal mask；计算随 patch 数平方增长，是高分辨率 ViT 的主要瓶颈。

### 复杂度分析
- **时间复杂度**：注意力为 $O(BN^2D)$，QKV/输出投影和四倍 MLP 为 $O(BND^2)$。
- **空间复杂度**：注意力矩阵占 $O(BHN^2)$，QKV 与 MLP 激活占 $O(BND)$（常数含头数和 4 倍扩张）。""",
}

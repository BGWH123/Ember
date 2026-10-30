"""2D Convolution task."""

TASK = {
    'title': '2D Convolution',
    'title_zh': '二维卷积',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'my_conv2d',
    'starter_code': """def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    pass""",
    'hint': """1. Zero-pad input manually (no F.pad)
2. `x.unfold(2, kH, stride).unfold(3, kW, stride)` → patches `(B, C, H_out, W_out, kH, kW)`
3. `(patches * weight).sum((-3,-2,-1)) + bias`""",
    'hint_zh': """1. 手动对输入进行零填充（不用 F.pad）
2. `x.unfold(2, kH, stride).unfold(3, kW, stride)` → 块 `(B, C, H_out, W_out, kH, kW)`
3. `(patches * weight).sum((-3,-2,-1)) + bias`""",
    'description_en': """Implement 2D convolution from scratch.

Convolution slides a kernel over a 2D input, computing weighted sums at each position. It is the fundamental operation in CNNs.

**Signature:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`

**Parameters:**
- `x` — input tensor (B, C_in, H, W)
- `weight` — kernel tensor (C_out, C_in, kH, kW)
- `bias` — optional bias (C_out,)
- `stride`, `padding` — integer stride and zero-padding

**Returns:** convolved output tensor

**Constraints:**
- Must match `F.conv2d` numerically
- Support stride and padding parameters""",
    'description_zh': """从零实现二维卷积。

卷积将卷积核在二维输入上滑动，在每个位置计算加权和，是 CNN 的基本操作。

**签名:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`

**参数:**
- `x` — 输入张量 (B, C_in, H, W)
- `weight` — 卷积核张量 (C_out, C_in, kH, kW)
- `bias` — 可选偏置 (C_out,)
- `stride`, `padding` — 整数步幅和零填充

**返回:** 卷积输出张量

**约束:**
- 必须与 `F.conv2d` 数值一致
- 支持 stride 和 padding 参数""",
    "tests": [
        {
            "name": 'Output shape',
            "code": """import torch
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
out = {fn}(x, w)
assert out.shape == (1, 16, 6, 6), f'Shape: {out.shape}'""",
        },
        {
            "name": 'Matches F.conv2d',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 8, 8)
w = torch.randn(4, 3, 3, 3)
b = torch.randn(4)
out = {fn}(x, w, b)
ref = torch.nn.functional.conv2d(x, w, b)
assert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'""",
        },
        {
            "name": 'With padding',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(1, 1, 5, 5)
w = torch.randn(1, 1, 3, 3)
out = {fn}(x, w, padding=1)
ref = torch.nn.functional.conv2d(x, w, padding=1)
assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Padding mismatch'""",
        },
        {
            "name": 'With stride',
            "code": """import torch
torch.manual_seed(0)
x = torch.randn(1, 1, 8, 8)
w = torch.randn(1, 1, 3, 3)
out = {fn}(x, w, stride=2)
ref = torch.nn.functional.conv2d(x, w, stride=2)
assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Stride mismatch'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
x = torch.randn(1, 1, 4, 4, requires_grad=True)
w = torch.randn(2, 1, 3, 3, requires_grad=True)
{fn}(x, w).sum().backward()
assert x.grad is not None and w.grad is not None, 'Missing gradients'""",
        },
    ],
    'solution': '''def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    """二维卷积展开实现 (2D Convolution with Unfold)。"""
    # Step 1: 按需在高宽两侧添加零填充
    if padding > 0:
        B, C, H, W = x.shape
        x_padded = torch.zeros(
            B, C, H + 2 * padding, W + 2 * padding,
            dtype=x.dtype, device=x.device
        )
        x_padded[
            :, :, padding:padding + H, padding:padding + W
        ] = x
        x = x_padded

    # Step 2: 使用 unfold 提取所有滑动窗口
    B, C_in, H, W = x.shape
    C_out, _, k_h, k_w = weight.shape
    H_out = (H - k_h) // stride + 1
    W_out = (W - k_w) // stride + 1
    patches = x.unfold(2, k_h, stride).unfold(3, k_w, stride)

    # Step 3: 对窗口与卷积核收缩并添加通道偏置
    out = torch.einsum("bihwjk,oijk->bohw", patches, weight)
    if bias is not None:
        out = out + bias.view(1, C_out, 1, 1)
    return out.reshape(B, C_out, H_out, W_out)''',
    'demo': """torch.manual_seed(42)
B, C, H, W = 1, 3, 8, 8
x = torch.randn(B, C, H, W)
weight = torch.randn(4, C, 3, 3)
bias = torch.randn(4)
out = my_conv2d(x, weight, bias, padding=1)
print("Output shape:", out.shape)
print("Output mean:", out.mean().item())
print("Sample values:", out[0, 0, :2, :2])""",
    'explanation': r"""### 核心思路
先把输入滑动窗口展开为视图，再用爱因斯坦求和同时完成输入通道和卷积核空间维的收缩。

### 代码步骤拆解
- **Step 1（零填充）**：构造扩展后的 `(B, C, H+2p, W+2p)` 张量并写入原输入。
- **Step 2（窗口展开）**：两次 `unfold` 得到 `(B, C_in, H_out, W_out, k_h, k_w)`。
- **Step 3（核收缩）**：`einsum` 与 `(C_out, C_in, k_h, k_w)` 权重相乘，再广播偏置。

### 重点与易错点分析
- **注意点 1（输出尺寸）**：应使用 $\lfloor(H+2p-k_h)/s\rfloor+1$，宽度同理。
- **注意点 2（维度标签）**：`einsum` 必须同时收缩输入通道与两个核维，保留批次、输出通道和位置维。
- **注意点 3（梯度链）**：窗口提取与张量收缩均为可微操作，不能把中间量转成 NumPy。

### 复杂度分析
- **时间复杂度**：$O(BC_{out}C_{in}H_{out}W_{out}k_hk_w)$。
- **空间复杂度**：$O(BC_{in}H_{out}W_{out}k_hk_w)$，展开窗口视图及算子中间量占主要空间。""",
}

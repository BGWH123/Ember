"""2D Convolution task."""

TASK = {
    "title": "2D Convolution",
    "title_zh": "二维卷积",
    "difficulty": "Medium",
    "category": "基础网络组件",
    "description_en": (
        "Implement 2D convolution from scratch.\n\nConvolution slides a kernel over a 2D input, computing weighted sums at each position. It is the fundamental operation in CNNs.\n\n**Signature:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor (B, C_in, H, W)\n- `weight` — kernel tensor (C_out, C_in, kH, kW)\n- `bias` — optional bias (C_out,)\n- `stride`, `padding` — integer stride and zero-padding\n\n**Returns:** convolved output tensor\n\n**Constraints:**\n- Must match `F.conv2d` numerically\n- Support stride and padding parameters"
    ),
    "description_zh": (
        "从零实现二维卷积。\n\n卷积将卷积核在二维输入上滑动，在每个位置计算加权和，是 CNN 的基本操作。\n\n**签名:** `my_conv2d(x, weight, bias=None, stride=1, padding=0) -> Tensor`\n\n**参数:**\n- `x` — 输入张量 (B, C_in, H, W)\n- `weight` — 卷积核张量 (C_out, C_in, kH, kW)\n- `bias` — 可选偏置 (C_out,)\n- `stride`, `padding` — 整数步幅和零填充\n\n**返回:** 卷积输出张量\n\n**约束:**\n- 必须与 `F.conv2d` 数值一致\n- 支持 stride 和 padding 参数"
    ),
    "function_name": "my_conv2d",
    "hint": (
        "1. Zero-pad input manually (no F.pad)\n2. `x.unfold(2, kH, stride).unfold(3, kW, stride)` → patches `(B, C, H_out, W_out, kH, kW)`\n3. `(patches * weight).sum((-3,-2,-1)) + bias`\n"
    ),
    "hint_zh": (
        "1. 手动对输入进行零填充（不用 F.pad）\n2. `x.unfold(2, kH, stride).unfold(3, kW, stride)` → 块 `(B, C, H_out, W_out, kH, kW)`\n3. `(patches * weight).sum((-3,-2,-1)) + bias`"
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """









import torch
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
out = {fn}(x, w)
assert out.shape == (1, 16, 6, 6), f'Shape: {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Matches F.conv2d",
            "code": """









import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 8, 8)
w = torch.randn(4, 3, 3, 3)
b = torch.randn(4)
out = {fn}(x, w, b)
ref = torch.nn.functional.conv2d(x, w, b)
assert torch.allclose(out, ref, atol=1e-4), f'Max diff: {(out-ref).abs().max():.6f}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "With padding",
            "code": """









import torch
torch.manual_seed(0)
x = torch.randn(1, 1, 5, 5)
w = torch.randn(1, 1, 3, 3)
out = {fn}(x, w, padding=1)
ref = torch.nn.functional.conv2d(x, w, padding=1)
assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Padding mismatch'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "With stride",
            "code": """









import torch
torch.manual_seed(0)
x = torch.randn(1, 1, 8, 8)
w = torch.randn(1, 1, 3, 3)
out = {fn}(x, w, stride=2)
ref = torch.nn.functional.conv2d(x, w, stride=2)
assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-4), 'Stride mismatch'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
x = torch.randn(1, 1, 4, 4, requires_grad=True)
w = torch.randn(2, 1, 3, 3, requires_grad=True)
{fn}(x, w).sum().backward()
assert x.grad is not None and w.grad is not None, 'Missing gradients'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        B, C, H, W = x.shape
        x_pad = torch.zeros(B, C, H + 2*padding, W + 2*padding, dtype=x.dtype, device=x.device)
        x_pad[:, :, padding:padding+H, padding:padding+W] = x
        x = x_pad
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
    out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print('Output:', my_conv2d(x, w).shape)
print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
    
    
    
    
    
    
    
    
    ''',
}

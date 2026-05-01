"""Max Pooling 2D task."""

TASK = {
    "title": "Max Pooling 2D",
    "title_zh": "二维最大池化",
    "difficulty": "Easy",
    "category": "基础网络组件",
    "description_en": (
        "Implement 2D max pooling from scratch.\n\nMax pooling slides a window over the spatial dimensions of a feature map and takes the maximum value in each window.\n\n**Signature:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`\n\n**Parameters:**\n- `x` — input tensor of shape (B, C, H, W)\n- `kernel_size` — size of the pooling window (square)\n- `stride` — step size between windows; defaults to `kernel_size` if `None`\n\n**Returns:** tensor of shape (B, C, H_out, W_out) where\n- `H_out = (H - kernel_size) // stride + 1`\n- `W_out = (W - kernel_size) // stride + 1`\n\n**Constraints:**\n- Must match `torch.nn.functional.max_pool2d` numerically\n- Do not call any `F.*` or `nn.functional.*` functions"
    ),
    "description_zh": (
        "从零实现二维最大池化。\n\n最大池化在特征图的空间维度上滑动窗口，取每个窗口内的最大值。\n\n**签名:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`\n\n**参数:**\n- `x` — 输入张量，形状为 (B, C, H, W)\n- `kernel_size` — 池化窗口大小（正方形）\n- `stride` — 窗口步长；若为 `None` 则默认等于 `kernel_size`\n\n**返回:** 形状为 (B, C, H_out, W_out) 的张量，其中\n- `H_out = (H - kernel_size) // stride + 1`\n- `W_out = (W - kernel_size) // stride + 1`\n\n**约束:**\n- 结果必须与 `torch.nn.functional.max_pool2d` 在数值上一致\n- 不得调用任何 `F.*` 或 `nn.functional.*` 函数"
    ),
    "function_name": "max_pool2d",
    "hint": "`x.unfold(2, k, stride).unfold(3, k, stride)` → `(B, C, H_out, W_out, k, k)` → `.flatten(-2).max(dim=-1).values`.",
    "hint_zh": "`x.unfold(2, k, stride).unfold(3, k, stride)` → `(B, C, H_out, W_out, k, k)` → `.flatten(-2).max(dim=-1).values`。",
    "tests": [
        {
            "name": "Output shape (kernel=2, stride=2)",
            "code": """









import torch
x = torch.randn(2, 3, 8, 8)
out = {fn}(x, kernel_size=2, stride=2)
assert out.shape == (2, 3, 4, 4), f'Expected (2,3,4,4), got {out.shape}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Matches F.max_pool2d numerically",
            "code": """









import torch
import torch.nn.functional as F
torch.manual_seed(42)
x = torch.randn(2, 3, 8, 8)
out = {fn}(x, kernel_size=2, stride=2)
ref = F.max_pool2d(x, kernel_size=2, stride=2)
assert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out - ref).abs().max().item()}'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "stride=None defaults to kernel_size",
            "code": """









import torch
import torch.nn.functional as F
torch.manual_seed(7)
x = torch.randn(1, 2, 6, 6)
out_none = {fn}(x, kernel_size=3, stride=None)
out_explicit = {fn}(x, kernel_size=3, stride=3)
ref = F.max_pool2d(x, kernel_size=3, stride=3)
assert torch.allclose(out_none, out_explicit, atol=1e-6), 'stride=None should equal stride=kernel_size'
assert torch.allclose(out_none, ref, atol=1e-5), 'stride=None result does not match reference'

            
            
            
            
            
            
            
            
            """,
        },
        {
            "name": "Gradient flow",
            "code": """









import torch
x = torch.randn(1, 1, 4, 4, requires_grad=True)
out = {fn}(x, kernel_size=2, stride=2)
out.sum().backward()
assert x.grad is not None, 'Gradient did not flow back to input'
assert x.grad.shape == x.shape, f'Grad shape mismatch: {x.grad.shape}'

            
            
            
            
            
            
            
            
            """,
        },
    ],
    "solution": '''








def max_pool2d(x, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    # unfold extracts sliding local blocks
    # after two unfolds: (B, C, H_out, W_out, kernel_size, kernel_size)
    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return patches.flatten(-2).max(dim=-1).values  # 求最大值
    
    
    
    
    
    
    
    
    ''',
    "demo": '''








x = torch.randn(1, 1, 4, 4)
out = max_pool2d(x, kernel_size=2, stride=2)
ref = F.max_pool2d(x, kernel_size=2, stride=2)
print("Output shape:", out.shape)
print("Matches F.max_pool2d:", torch.allclose(out, ref))
    
    
    
    
    
    
    
    
    ''',
}

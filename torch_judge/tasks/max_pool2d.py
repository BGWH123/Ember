"""Max Pooling 2D task."""

TASK = {
    'title': 'Max Pooling 2D',
    'title_zh': '二维最大池化',
    'difficulty': 'Easy',
    'category': '基础网络组件',
    'function_name': 'max_pool2d',
    'starter_code': """def max_pool2d(x, kernel_size, stride=None):
    pass""",
    'hint': """`x.unfold(2, k, stride).unfold(3, k, stride)` → `(B, C, H_out, W_out, k, k)` → `.flatten(-2).max(dim=-1).values`.""",
    'hint_zh': """`x.unfold(2, k, stride).unfold(3, k, stride)` → `(B, C, H_out, W_out, k, k)` → `.flatten(-2).max(dim=-1).values`。""",
    'description_en': """Implement 2D max pooling from scratch.

Max pooling slides a window over the spatial dimensions of a feature map and takes the maximum value in each window.

**Signature:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`

**Parameters:**
- `x` — input tensor of shape (B, C, H, W)
- `kernel_size` — size of the pooling window (square)
- `stride` — step size between windows; defaults to `kernel_size` if `None`

**Returns:** tensor of shape (B, C, H_out, W_out) where
- `H_out = (H - kernel_size) // stride + 1`
- `W_out = (W - kernel_size) // stride + 1`

**Constraints:**
- Must match `torch.nn.functional.max_pool2d` numerically
- Do not call any `F.*` or `nn.functional.*` functions""",
    'description_zh': """从零实现二维最大池化。

最大池化在特征图的空间维度上滑动窗口，取每个窗口内的最大值。

**签名:** `max_pool2d(x, kernel_size, stride=None) -> Tensor`

**参数:**
- `x` — 输入张量，形状为 (B, C, H, W)
- `kernel_size` — 池化窗口大小（正方形）
- `stride` — 窗口步长；若为 `None` 则默认等于 `kernel_size`

**返回:** 形状为 (B, C, H_out, W_out) 的张量，其中
- `H_out = (H - kernel_size) // stride + 1`
- `W_out = (W - kernel_size) // stride + 1`

**约束:**
- 结果必须与 `torch.nn.functional.max_pool2d` 在数值上一致
- 不得调用任何 `F.*` 或 `nn.functional.*` 函数""",
    "tests": [
        {
            "name": 'Output shape (kernel=2, stride=2)',
            "code": """import torch
x = torch.randn(2, 3, 8, 8)
out = {fn}(x, kernel_size=2, stride=2)
assert out.shape == (2, 3, 4, 4), f'Expected (2,3,4,4), got {out.shape}'""",
        },
        {
            "name": 'Matches F.max_pool2d numerically',
            "code": """import torch
import torch.nn.functional as F
torch.manual_seed(42)
x = torch.randn(2, 3, 8, 8)
out = {fn}(x, kernel_size=2, stride=2)
ref = F.max_pool2d(x, kernel_size=2, stride=2)
assert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out - ref).abs().max().item()}'""",
        },
        {
            "name": 'stride=None defaults to kernel_size',
            "code": """import torch
import torch.nn.functional as F
torch.manual_seed(7)
x = torch.randn(1, 2, 6, 6)
out_none = {fn}(x, kernel_size=3, stride=None)
out_explicit = {fn}(x, kernel_size=3, stride=3)
ref = F.max_pool2d(x, kernel_size=3, stride=3)
assert torch.allclose(out_none, out_explicit, atol=1e-6), 'stride=None should equal stride=kernel_size'
assert torch.allclose(out_none, ref, atol=1e-5), 'stride=None result does not match reference'""",
        },
        {
            "name": 'Gradient flow',
            "code": """import torch
x = torch.randn(1, 1, 4, 4, requires_grad=True)
out = {fn}(x, kernel_size=2, stride=2)
out.sum().backward()
assert x.grad is not None, 'Gradient did not flow back to input'
assert x.grad.shape == x.shape, f'Grad shape mismatch: {x.grad.shape}'""",
        },
    ],
    'solution': '''def max_pool2d(x, kernel_size, stride=None):
    """从零实现二维最大池化 (2D Max Pooling)。"""
    # Step 1: 将缺省步长设为池化窗口大小
    if stride is None:
        stride = kernel_size

    # Step 2: 沿高和宽提取滑动局部窗口
    patches = x.unfold(2, kernel_size, stride).unfold(
        3, kernel_size, stride
    )

    # Step 3: 合并窗口维并取每个窗口最大值
    out = patches.flatten(-2).max(dim=-1).values
    return out''',
    'demo': """torch.manual_seed(42)
x = torch.randn(1, 2, 6, 6)
out = max_pool2d(x, kernel_size=2, stride=2)
reference = F.max_pool2d(x, kernel_size=2, stride=2)
print("Output shape:", out.shape)
print("Matches PyTorch:", torch.allclose(out, reference))
print("Sample values:", out[0, 0, 0])""",
    'explanation': r"""### 核心思路
二维最大池化通过两次 `unfold` 把输入变成 `(B, C, H_out, W_out, K, K)` 窗口视图，再对最后两个窗口维取最大值。输出空间大小遵循 $\lfloor(H-K)/stride
floor+1$。

### 代码步骤拆解
- **Step 1（步长解析）**：`stride=None` 时使用 `kernel_size`。
- **Step 2（窗口提取）**：沿维 2 和维 3 依次展开高、宽。
- **Step 3（最大归约）**：展平两个窗口轴并沿最后一维取最大值。

### 重点与易错点分析
- **注意点 1（维度顺序）**：第二次 `unfold` 仍应作用于维 3；第一次展开新增的窗口维位于末尾。
- **注意点 2（边界行为）**：该实现不填充，只保留完整窗口，与默认 `F.max_pool2d` 一致。
- **注意点 3（梯度分配）**：最大值梯度只流向获胜元素；并列最大值的具体梯度选择由 PyTorch 决定。

### 复杂度分析
- **时间复杂度**：$O(BCH_{out}W_{out}K^2)$，每个窗口扫描 $K^2$ 个元素。
- **空间复杂度**：$O(BCH_{out}W_{out}K^2)$ 的逻辑窗口视图，`unfold` 本身通常不复制底层数据。""",
}

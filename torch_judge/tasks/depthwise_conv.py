"""Depthwise Separable Convolution task."""

TASK = {
    'title': 'Depthwise Separable Convolution',
    'title_zh': '深度可分离卷积',
    'difficulty': 'Medium',
    'category': '基础网络组件',
    'function_name': 'depthwise_separable_conv',
    'starter_code': """def depthwise_separable_conv(x, dw_weight, pw_weight):
    pass""",
    'hint': """Depthwise: `x.unfold(2,kH,1).unfold(3,kW,1)` → `(B,C,H_out,W_out,kH,kW)` → multiply `dw_weight` → sum last 2 dims
Pointwise: `torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:,: ,0,0])`""",
    'hint_zh': """深度卷积：`x.unfold(2,kH,1).unfold(3,kW,1)` → `(B,C,H_out,W_out,kH,kW)` → 乘 `dw_weight` → 对最后 2 维求和
逐点卷积：`torch.einsum('bchw,oc->bohw', dw_out, pw_weight[:,: ,0,0])`""",
    'description_en': """Implement MobileNet-style depthwise separable convolution from primitives.

Depthwise separable convolution factorizes a standard convolution into two steps: a depthwise conv (one filter per input channel) followed by a pointwise 1x1 conv (to mix channels). This dramatically reduces parameter count and FLOPs.

**Signature:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`

**Parameters:**
- `x` — input tensor (B, C_in, H, W)
- `dw_weight` — depthwise filter (C_in, 1, kH, kW)
- `pw_weight` — pointwise filter (C_out, C_in, 1, 1)

**Returns:** output tensor (B, C_out, H-kH+1, W-kW+1)

**Constraints:**
- No padding, stride=1
- Must NOT use `F.conv2d` — implement both steps using `unfold` and `einsum`/matmul
- Depthwise step: each channel c is convolved only with `dw_weight[c, 0]`
- Pointwise step: 1x1 conv mixes channels via matrix multiply""",
    'description_zh': """从基本原语实现 MobileNet 风格的深度可分离卷积。

深度可分离卷积将标准卷积分解为两步：深度卷积（每个输入通道一个滤波器）和逐点 1x1 卷积（混合通道）。这大幅减少了参数量和计算量。

**签名:** `depthwise_separable_conv(x, dw_weight, pw_weight) -> Tensor`

**参数:**
- `x` — 输入张量 (B, C_in, H, W)
- `dw_weight` — 深度滤波器 (C_in, 1, kH, kW)
- `pw_weight` — 逐点滤波器 (C_out, C_in, 1, 1)

**返回:** 输出张量 (B, C_out, H-kH+1, W-kW+1)

**约束:**
- 无填充，步长=1
- 不得使用 `F.conv2d`——使用 `unfold` 和 `einsum`/matmul 实现两步
- 深度步骤：通道 c 仅与 `dw_weight[c, 0]` 卷积
- 逐点步骤：1x1 卷积通过矩阵乘法混合通道""",
    "tests": [
        {
            "name": 'Output shape is correct',
            "code": """import torch
torch.manual_seed(0)
B, C_in, H, W = 2, 4, 8, 8
kH, kW, C_out = 3, 3, 6
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.randn(C_out, C_in, 1, 1)
out = {fn}(x, dw_weight, pw_weight)
expected_shape = (B, C_out, H - kH + 1, W - kW + 1)
assert out.shape == expected_shape, f'Shape: {out.shape}, expected {expected_shape}'""",
        },
        {
            "name": 'Depthwise step: channel independence',
            "code": """import torch
torch.manual_seed(1)
B, C_in, H, W = 1, 3, 6, 6
kH, kW = 3, 3
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.eye(C_in).unsqueeze(-1).unsqueeze(-1)  # identity pointwise
out1 = {fn}(x, dw_weight, pw_weight)
x2 = x.clone()
x2[:, 1] = x2[:, 1] * 0  # zero out channel 1
out2 = {fn}(x2, dw_weight, pw_weight)
# channel 0 output should be unchanged
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Channel 0 output changed when channel 1 was zeroed — depthwise not independent'""",
        },
        {
            "name": 'Matches reference loop implementation',
            "code": """import torch
torch.manual_seed(2)
B, C_in, H, W = 1, 2, 5, 5
kH, kW, C_out = 3, 3, 3
x = torch.randn(B, C_in, H, W)
dw_weight = torch.randn(C_in, 1, kH, kW)
pw_weight = torch.randn(C_out, C_in, 1, 1)
H_out, W_out = H - kH + 1, W - kW + 1
# Reference: loop-based depthwise then pointwise
dw_ref = torch.zeros(B, C_in, H_out, W_out)
for c in range(C_in):
    for i in range(H_out):
        for j in range(W_out):
            dw_ref[:, c, i, j] = (x[:, c, i:i+kH, j:j+kW] * dw_weight[c, 0]).sum(dim=(-2, -1))
ref = torch.einsum('bchw,oc->bohw', dw_ref, pw_weight[:, :, 0, 0])
out = {fn}(x, dw_weight, pw_weight)
assert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out - ref).abs().max().item()}'""",
        },
        {
            "name": 'Gradient flows',
            "code": """import torch
torch.manual_seed(3)
x = torch.randn(1, 2, 6, 6, requires_grad=True)
dw_weight = torch.randn(2, 1, 3, 3, requires_grad=True)
pw_weight = torch.randn(4, 2, 1, 1, requires_grad=True)
out = {fn}(x, dw_weight, pw_weight)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert dw_weight.grad is not None, 'dw_weight.grad is None'
assert pw_weight.grad is not None, 'pw_weight.grad is None'""",
        },
    ],
    'solution': '''def depthwise_separable_conv(x, dw_weight, pw_weight):
    """用基础张量操作实现深度可分离卷积 (Depthwise Separable Convolution)。"""
    B, C, H, W = x.shape
    kernel_h, kernel_w = dw_weight.shape[-2:]

    # Step 1: 展开空间滑窗，得到逐位置局部块
    patches = x.unfold(2, kernel_h, 1).unfold(3, kernel_w, 1)
    # (B, C, H_out, W_out, kernel_h, kernel_w)

    # Step 2: 每个输入通道独立执行深度卷积
    kernels = dw_weight[:, 0].view(1, C, 1, 1, kernel_h, kernel_w)
    depthwise_out = (patches * kernels).sum(dim=(-2, -1))

    # Step 3: 用 1x1 权重沿通道维执行逐点混合
    pointwise_weight = pw_weight[:, :, 0, 0]
    out = torch.einsum("bchw,oc->bohw", depthwise_out, pointwise_weight)
    return out''',
    'demo': """torch.manual_seed(42)
B, C, H, W = 2, 4, 8, 8
x = torch.randn(B, C, H, W)
dw_weight = torch.randn(C, 1, 3, 3)
pw_weight = torch.randn(8, C, 1, 1)
out = depthwise_separable_conv(x, dw_weight, pw_weight)
print("Output shape:", out.shape)
print("Sample values:", out[0, 0, :2, :2])""",
    'explanation': r"""### 核心思路
深度可分离卷积先让每个输入通道独立与自己的空间核做卷积，再用 $1\times1$ 逐点权重在线性通道空间中混合结果。

### 代码步骤拆解
- **Step 1（提取滑窗）**：`unfold` 将 `(B, C, H, W)` 变为 `(B, C, H_o, W_o, k_h, k_w)`。
- **Step 2（深度卷积）**：广播 `kernels` 后在最后两个核维求和，得到 `(B, C, H_o, W_o)`。
- **Step 3（逐点卷积）**：通过 `einsum` 将 `(C_out, C)` 权重与通道维缩并，输出 `(B, C_out, H_o, W_o)`。

### 重点与易错点分析
- **注意点 1（通道独立）**：深度阶段不能提前混合通道，每个 `dw_weight[c, 0]` 只作用于 `x[:, c]`。
- **注意点 2（卷积约定）**：实现的是深度学习常用的互相关，无需翻转卷积核，且题目规定无 padding、步长为 1。
- **注意点 3（内存折中）**：`unfold` 逻辑上展开所有滑窗，代码清晰但可能产生较大中间视图或后续广播开销。

### 复杂度分析
- **时间复杂度**：$O(BH_oW_o(Ck_hk_w+C C_{out}))$，深度和逐点两阶段共同构成开销。
- **空间复杂度**：$O(BCH_oW_ok_hk_w)$，展开后的滑窗是主要中间激活。""",
}
